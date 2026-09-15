"""Forward / Backward トラッキング結果の照合と track_id の連鎖割り当て.

## 考え方

区間 s の backward プロンプトと、区間 s+1 の forward プロンプトは
**同じアンカーフレームの同じ Detection2D ボックス**なので、
境界での照合は不要（構造的に同一）。照合が必要なのは区間内の F×B だけ。

ノードを ``(アンカー, プロンプト index)`` とし、区間ごとの F×B マッチングを
エッジとして連鎖させると、先頭アンカーのプロンプト数だけ採番した
track_id がシーン全体へ伝播する。マッチしなかった backward は
「区間途中で現れた新規」として採番する。

## 設計上の判断

- **時空間 IoU で集計する**: フレームごとに独立で照合すると、
  1 フレーム形が崩れた瞬間に別トラック扱いになる。区間全体で見れば安定する。
- **分母は「どちらかが存在するフレーム数」**: 存在期間のずれもペナルティに
  なるので、途中出現の B が全区間ある F へ誤って吸い込まれるのを防げる。
- **マッチングは Hungarian**: 貪欲より速く（n=60 で 27 倍）かつ最適。
  ただし必ず最大マッチングを作るので、**割り当て後に閾値未満を落とす**。
- **box IoU が既定**: mask IoU の総当たりは n=30 で 11 秒かかり実用外。
  mask を選んだ場合は box IoU で候補を絞ってから計算する。
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from app.core.logging import get_logger
from common.mask_rle import decode_rle

logger = get_logger(__name__)

IOU_METHOD_BOX = "box"
IOU_METHOD_MASK = "mask"

IOU_LABEL_MATCH_LABEL = "label"
IOU_LABEL_MATCH_CATEGORY_GROUP = "category_group"
IOU_LABEL_MATCH_NONE = "none"

# mask IoU を計算する候補を絞るときの box IoU の下限。
# ここを 0 にすると総当たりと変わらなくなる
MASK_CANDIDATE_BOX_IOU = 1e-6


def _mask_iou_pairs(
    forward_instances: Sequence[dict[str, Any]],
    backward_instances: Sequence[dict[str, Any]],
    candidate_pairs: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """候補ペアの mask IoU をまとめて計算する.

    素朴に ``mask_iou(rle_a, rle_b)`` をペアごとに呼ぶと、同じマスクを
    何度も全画面デコードすることになり n=30 で 2 秒かかる。

    2 点で速くしている:
      - マスクのデコードは **1 フレームにつき 1 回**（結果を使い回す）
      - 交差判定は **外接矩形の重なり部分だけ**を切り出して行う
        （全画面の bool 演算は 1 回 1ms 程度かかる）
    """
    decoded_f: dict[int, np.ndarray] = {}
    decoded_b: dict[int, np.ndarray] = {}
    results: dict[tuple[int, int], float] = {}

    for i, j in candidate_pairs:
        instance_f = forward_instances[i]
        instance_b = backward_instances[j]

        # 重なり領域（外接矩形の共通部分）だけを見る
        x0 = max(instance_f["xmin"], instance_b["xmin"])
        y0 = max(instance_f["ymin"], instance_b["ymin"])
        x1 = min(instance_f["xmax"], instance_b["xmax"])
        y1 = min(instance_f["ymax"], instance_b["ymax"])
        if x1 <= x0 or y1 <= y0:
            results[(i, j)] = 0.0
            continue

        if i not in decoded_f:
            decoded_f[i] = decode_rle(instance_f["mask_rle"])
        if j not in decoded_b:
            decoded_b[j] = decode_rle(instance_b["mask_rle"])

        mask_f, mask_b = decoded_f[i], decoded_b[j]
        rows = slice(int(y0), int(y1))
        cols = slice(int(x0), int(x1))
        intersection = int(np.count_nonzero(
            mask_f[rows, cols] & mask_b[rows, cols]
        ))
        if intersection == 0:
            results[(i, j)] = 0.0
            continue

        # 面積は全画面で数える（重なり領域の外にも画素がある）
        area_f = int(np.count_nonzero(mask_f))
        area_b = int(np.count_nonzero(mask_b))
        union = area_f + area_b - intersection
        results[(i, j)] = intersection / union if union > 0 else 0.0

    return results


def _box_iou_matrix(
    forward_boxes: np.ndarray, backward_boxes: np.ndarray
) -> np.ndarray:
    """``(N, 4)`` と ``(M, 4)`` の box IoU 行列 ``(N, M)``.

    1 ペアずつ回すと n=60 で 53ms かかる。ベクトル化すれば 1ms で済む。
    """
    if forward_boxes.shape[0] == 0 or backward_boxes.shape[0] == 0:
        return np.zeros((forward_boxes.shape[0], backward_boxes.shape[0]))

    x0 = np.maximum(forward_boxes[:, None, 0], backward_boxes[None, :, 0])
    y0 = np.maximum(forward_boxes[:, None, 1], backward_boxes[None, :, 1])
    x1 = np.minimum(forward_boxes[:, None, 2], backward_boxes[None, :, 2])
    y1 = np.minimum(forward_boxes[:, None, 3], backward_boxes[None, :, 3])

    intersection = np.clip(x1 - x0, 0.0, None) * np.clip(y1 - y0, 0.0, None)
    area_f = (
        (forward_boxes[:, 2] - forward_boxes[:, 0])
        * (forward_boxes[:, 3] - forward_boxes[:, 1])
    )
    area_b = (
        (backward_boxes[:, 2] - backward_boxes[:, 0])
        * (backward_boxes[:, 3] - backward_boxes[:, 1])
    )
    union = area_f[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def label_compatibility(
    forward: Sequence[dict[str, Any]],
    backward: Sequence[dict[str, Any]],
    *,
    label_match: str = IOU_LABEL_MATCH_LABEL,
    label_to_category_group: dict[str, str] | None = None,
) -> np.ndarray:
    """IoU を計算してよいペアを示す bool 行列 ``(N, M)``.

    label_match に応じて、ラベル一致・カテゴリグループ一致・無条件を切り替える。
    """
    n, m = len(forward), len(backward)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=bool)
    if label_match == IOU_LABEL_MATCH_NONE:
        return np.ones((n, m), dtype=bool)

    if label_match == IOU_LABEL_MATCH_CATEGORY_GROUP:
        mapping = label_to_category_group or {}
        # グループが引けないラベルは、ラベル名そのものをグループ扱いにする
        keys_f = [mapping.get(i.get("label"), i.get("label")) for i in forward]
        keys_b = [mapping.get(i.get("label"), i.get("label")) for i in backward]
    else:
        keys_f = [i.get("label") for i in forward]
        keys_b = [i.get("label") for i in backward]

    return np.array([[kf == kb for kb in keys_b] for kf in keys_f], dtype=bool)


def temporal_iou_matrix(
    forward_tracks: Sequence[dict[int, dict[str, Any]]],
    backward_tracks: Sequence[dict[int, dict[str, Any]]],
    *,
    iou_method: str = IOU_METHOD_BOX,
    label_match: str = IOU_LABEL_MATCH_LABEL,
    label_to_category_group: dict[str, str] | None = None,
) -> np.ndarray:
    """時空間 IoU 行列 ``(N, M)`` を返す.

    Args:
        forward_tracks: forward の各トラックの ``{frame_index: instance}``。
            instance は ``xmin/ymin/xmax/ymax``（と mask 時は ``mask_rle``）、
            ``label`` を持つ
        backward_tracks: backward 側の同じ形

    集計は「両者の IoU の総和 ÷ どちらかが存在するフレーム数」。
    存在期間がずれていれば分母だけ増えてスコアが下がる。
    """
    n, m = len(forward_tracks), len(backward_tracks)
    if n == 0 or m == 0:
        return np.zeros((n, m))

    # ラベル条件は代表インスタンス（最初のフレーム）で判定する。
    # 同じトラック内でラベルが変わることはない前提
    def representative(track: dict[int, dict[str, Any]]) -> dict[str, Any]:
        return track[min(track)] if track else {}

    compatible = label_compatibility(
        [representative(t) for t in forward_tracks],
        [representative(t) for t in backward_tracks],
        label_match=label_match,
        label_to_category_group=label_to_category_group,
    )

    frames = sorted(
        {k for t in forward_tracks for k in t}
        | {k for t in backward_tracks for k in t}
    )
    total = np.zeros((n, m))
    denominator = np.zeros((n, m))

    for frame in frames:
        present_f = np.array([frame in t for t in forward_tracks])
        present_b = np.array([frame in t for t in backward_tracks])
        # どちらかが存在するフレームを分母に数える
        denominator += (present_f[:, None] | present_b[None, :]).astype(float)

        active_f = np.flatnonzero(present_f)
        active_b = np.flatnonzero(present_b)
        if active_f.size == 0 or active_b.size == 0:
            continue

        boxes_f = np.array([
            [forward_tracks[i][frame][k] for k in ("xmin", "ymin", "xmax", "ymax")]
            for i in active_f
        ], dtype=np.float64)
        boxes_b = np.array([
            [backward_tracks[j][frame][k] for k in ("xmin", "ymin", "xmax", "ymax")]
            for j in active_b
        ], dtype=np.float64)
        box_scores = _box_iou_matrix(boxes_f, boxes_b)

        if iou_method == IOU_METHOD_MASK:
            # box IoU が 0 のペアは mask IoU も 0。総当たりを避けるための絞り込み
            instances_f = [forward_tracks[int(i)][frame] for i in active_f]
            instances_b = [backward_tracks[int(j)][frame] for j in active_b]
            candidates = [
                (int(local_i), int(local_j))
                for local_i, local_j in zip(
                    *np.nonzero(box_scores > MASK_CANDIDATE_BOX_IOU)
                )
                if compatible[int(active_f[local_i]), int(active_b[local_j])]
            ]
            pair_scores = _mask_iou_pairs(instances_f, instances_b, candidates)
            scores = np.zeros_like(box_scores)
            for (local_i, local_j), value in pair_scores.items():
                scores[local_i, local_j] = value
        else:
            scores = box_scores

        # ラベル条件を満たさないペアは 0 のまま
        sub_compatible = compatible[np.ix_(active_f, active_b)]
        total[np.ix_(active_f, active_b)] += np.where(sub_compatible, scores, 0.0)

    return np.divide(
        total, denominator, out=np.zeros_like(total), where=denominator > 0
    )


def match_forward_backward(
    score: np.ndarray, iou_threshold: float
) -> dict[int, int]:
    """Hungarian で 1 対 1 に割り当て、閾値未満を落とす.

    Returns:
        ``{forward index: backward index}``

    Hungarian は必ず最大マッチングを作るため、
    **割り当て後に閾値で切る**必要がある。これで「マッチしない」を表現する。
    """
    if score.size == 0:
        return {}
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-score)
    return {
        int(i): int(j)
        for i, j in zip(rows, cols)
        if score[i, j] >= iou_threshold
    }


def assign_chain_track_ids(
    anchors: Sequence[int],
    prompt_counts: dict[int, int],
    segment_matches: dict[tuple[int, int], dict[int, int]],
    *,
    start_track_id: int = 0,
) -> tuple[dict[tuple[int, int], int], int]:
    """アンカーのプロンプトを連鎖させて track_id を割り当てる.

    Args:
        anchors: プロンプトを与えたフレーム index（昇順）
        prompt_counts: ``{anchor: プロンプト数}``
        segment_matches: ``{(a0, a1): {forward index: backward index}}``

    Returns:
        ``({(anchor, prompt index): track_id}, 次に使う track_id)``

    先頭アンカーのプロンプト数だけ採番し、マッチした先へ id を渡す。
    マッチしなかった backward は区間途中の新規として採番する。
    """
    if not anchors:
        return {}, start_track_id

    assigned: dict[tuple[int, int], int] = {}
    next_id = start_track_id

    for index in range(prompt_counts.get(anchors[0], 0)):
        assigned[(anchors[0], index)] = next_id
        next_id += 1

    for a0, a1 in zip(anchors, anchors[1:]):
        matched = segment_matches.get((a0, a1), {})
        for forward_index, backward_index in matched.items():
            source = assigned.get((a0, forward_index))
            if source is None:
                # forward 側に id が無い（前区間で新規だった等）場合は採番する
                source = next_id
                assigned[(a0, forward_index)] = source
                next_id += 1
            assigned[(a1, backward_index)] = source

        for index in range(prompt_counts.get(a1, 0)):
            if (a1, index) not in assigned:
                # どの forward ともマッチしなかった = 区間途中で出現
                assigned[(a1, index)] = next_id
                next_id += 1

    return assigned, next_id


def select_direction(
    frame: int, anchor_start: int, anchor_end: int
) -> str:
    """そのフレームで採用する伝播方向を返す.

    プロンプトフレームに近い方を採る。伝播のドリフトは距離に比例して
    増えるため、区間の前半は forward、後半は backward が有利。
    同距離なら forward を優先する（境界の扱いを一定にするため）。
    """
    return "forward" if (frame - anchor_start) <= (anchor_end - frame) else "backward"
