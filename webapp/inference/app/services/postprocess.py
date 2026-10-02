"""検出結果の後処理.

同一クラス内の NMS はグループごとの推論の中で完結するが、
**クラスをまたぐ NMS はフレーム単位でしか適用できない**。
全グループの推論が終わってから、そのフレームの全ボックスに対して行う。

例: 同じ物体が "vehicle" グループで car、"two_wheeler" グループで
motorcycle として二重に検出されることがある。
"""
from __future__ import annotations

from typing import Any, Sequence

from common.box_ops import box_iou as iou


def cross_class_nms(
    boxes: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """クラスを問わず重なりの大きいボックスを抑制する.

    スコアの高い順に残し、閾値以上に重なる後続を落とす。
    同一クラス同士も対象になるが、そちらは既にグループ内 NMS で
    処理済みなので実質的な影響はない。
    """
    if iou_threshold >= 1.0 or len(boxes) <= 1:
        return list(boxes)

    ordered = sorted(boxes, key=lambda b: b.get("score", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for box in ordered:
        if all(iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def resolve_sublabels(
    boxes: list[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    """モデルが返した語（サブラベル）をラベルへ畳み込む.

    元の語は sublabel に残す。NMS より**前**に呼ぶこと。
    後で畳み込むと、van と car が別クラス扱いのまま NMS を通り、
    同じ物体に 2 つのボックスが残る。
    """
    resolved: list[dict[str, Any]] = []
    for box in boxes:
        sublabel = box.get("label", "")
        resolved.append({
            **box,
            "sublabel": sublabel,
            # 未知の語はそのままラベルとして扱う（設定漏れで落とさない）
            "label": mapping.get(sublabel, sublabel),
        })
    return resolved


def same_class_nms(
    boxes: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """同じ label 同士でのみ NMS をかける.

    サブラベルではなくラベルで比較する（resolve_sublabels を先に通す前提）。
    """
    if iou_threshold >= 1.0 or len(boxes) <= 1:
        return list(boxes)

    ordered = sorted(boxes, key=lambda b: b.get("score", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for box in ordered:
        same = [k for k in kept if k["label"] == box["label"]]
        if all(iou(box, k) < iou_threshold for k in same):
            kept.append(box)
    return kept


# ── 内包関係の重複検出を落とす（IoS）────────────────────────────────────

DELETE_SMALL = "small"
DELETE_BIG = "big"


def box_ios(a: dict[str, Any], b: dict[str, Any]) -> float:
    """2 つのボックスの IoS（小さい方の面積に対する重なり率）.

    IoU ではなく IoS を使う。大きさが違うと、内包していても IoU は
    小さくなってしまう（面積比がそのまま上限になる）。
    """
    left = max(float(a["xmin"]), float(b["xmin"]))
    top = max(float(a["ymin"]), float(b["ymin"]))
    right = min(float(a["xmax"]), float(b["xmax"]))
    bottom = min(float(a["ymax"]), float(b["ymax"]))
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    smaller = min(box_area(a), box_area(b))
    return float(intersection / smaller) if smaller > 0 else 0.0


def box_area(box: dict[str, Any]) -> float:
    return max(0.0, float(box["xmax"]) - float(box["xmin"])) * max(
        0.0, float(box["ymax"]) - float(box["ymin"])
    )


def suppress_contained_boxes(
    boxes: Sequence[dict[str, Any]],
    *,
    threshold: float,
    direction_by_label: dict[str, str],
) -> int:
    """同じラベルで内包関係にあるボックスの片方へ削除印を付ける.

    Args:
        direction_by_label: ``{ラベル: "small" | "big"}``。
            ここに無いラベルは判定しない
        threshold: IoS の下限

    Returns:
        削除印を付けた数。``boxes`` は **その場で書き換える**。

    ## なぜ総当たりなのか

    1 対 1 の割り当て（Hungarian）では、**1 つの大きなボックスに
    2 つが内包される**場合に片方しか処理できない。同ラベル内の全対を
    見る必要がある。

    ## 1 パスで判定する

    削除印は元の集合に対して付け、途中で対象から外さない。
    入れ子が 3 段（A ⊃ B ⊃ C）のとき、
      small … B と C が落ちて A が残る
      big   … A と B が落ちて C が残る
    となり、どちらも意図どおり。消しながら進めると順序に依存する。

    面積が同じ場合は index で順序を決めるので、入力の並びを変えても
    結果は変わらない。
    """
    if not boxes or threshold <= 0:
        return 0

    by_label: dict[str, list[int]] = {}
    for index, box in enumerate(boxes):
        if box.get("is_deleted"):
            # 再判定で既に落ちているものは対象外
            continue
        label = box.get("label")
        if label in direction_by_label:
            by_label.setdefault(label, []).append(index)

    marked = 0
    for label, indices in by_label.items():
        direction = direction_by_label[label]
        areas = {i: box_area(boxes[i]) for i in indices}
        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                if box_ios(boxes[i], boxes[j]) < threshold:
                    continue
                # 面積が同じときも決まるように index を第 2 キーにする
                big, small = (
                    (i, j) if (areas[i], i) >= (areas[j], j) else (j, i)
                )
                target = small if direction == DELETE_SMALL else big
                if not boxes[target].get("is_deleted"):
                    boxes[target]["is_deleted"] = True
                    boxes[target]["deleted_by"] = f"ios_{direction}"
                    marked += 1
    return marked
