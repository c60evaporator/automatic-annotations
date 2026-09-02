"""2D ボックスの基本演算.

webapp（手修正の引き継ぎ）と inference（NMS・トラック照合）の
両方で使うため common に置く。
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

BoxLike = dict[str, Any]


def box_iou(a: BoxLike, b: BoxLike) -> float:
    """2つのボックスの IoU.

    ボックスは ``{"xmin":..., "ymin":..., "xmax":..., "ymax":...}`` の dict。
    """
    ix1 = max(a["xmin"], b["xmin"])
    iy1 = max(a["ymin"], b["ymin"])
    ix2 = min(a["xmax"], b["xmax"])
    iy2 = min(a["ymax"], b["ymax"])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, a["xmax"] - a["xmin"]) * max(0, a["ymax"] - a["ymin"])
    area_b = max(0, b["xmax"] - b["xmin"]) * max(0, b["ymax"] - b["ymin"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    sources: Sequence[BoxLike],
    targets: Sequence[BoxLike],
    *,
    iou_threshold: float,
    iou_fn: Callable[[BoxLike, BoxLike], float] = box_iou,
    compatible: Callable[[BoxLike, BoxLike], bool] | None = None,
) -> dict[int, int]:
    """IoU の降順で 1 対 1 の貪欲マッチングを行う.

    用途:
      - 再実行時に手修正ボックスを推論ボックスへ対応づける（webapp）
      - プロンプト区間の境界で track_id を引き継ぐ（inference）

    Args:
        iou_fn: IoU の計算方法。マスク IoU を使う場合に差し替える
        compatible: ラベル一致などの追加条件。False を返す組は候補から外す

    Returns:
        ``{sources の index: targets の index}``。
        閾値に届かなかった source は含まれない。

    NOTE: 1 対 1 にすること。単純に「閾値超えなら対応」とすると、
    1 つの target が複数の source にマッチして重複が生まれる。
    貪欲なので組全体では最適にならないが、挙動が追いやすく決定的。
    """
    pairs: list[tuple[float, int, int]] = []
    for s_index, source in enumerate(sources):
        for t_index, target in enumerate(targets):
            if compatible is not None and not compatible(source, target):
                continue
            iou = iou_fn(source, target)
            if iou >= iou_threshold:
                pairs.append((iou, s_index, t_index))

    # IoU 降順。同点時は index 順にして結果を決定的にする
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    matched: dict[int, int] = {}
    used_targets: set[int] = set()
    for _iou, s_index, t_index in pairs:
        if s_index in matched or t_index in used_targets:
            continue
        matched[s_index] = t_index
        used_targets.add(t_index)
    return matched
