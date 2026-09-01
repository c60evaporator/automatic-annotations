"""検出結果の後処理.

同一クラス内の NMS はグループごとの推論の中で完結するが、
**クラスをまたぐ NMS はフレーム単位でしか適用できない**。
全グループの推論が終わってから、そのフレームの全ボックスに対して行う。

例: 同じ物体が "vehicle" グループで car、"two_wheeler" グループで
motorcycle として二重に検出されることがある。
"""
from __future__ import annotations

from typing import Any


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """2つのボックスの IoU."""
    ix1 = max(a["xmin"], b["xmin"])
    iy1 = max(a["ymin"], b["ymin"])
    ix2 = min(a["xmax"], b["xmax"])
    iy2 = min(a["ymax"], b["ymax"])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, a["xmax"] - a["xmin"]) * max(0, a["ymax"] - a["ymin"])
    area_b = max(0, b["xmax"] - b["xmin"]) * max(0, b["ymax"] - b["ymin"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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


def same_class_nms(
    boxes: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """同じ label 同士でのみ NMS をかける."""
    if iou_threshold >= 1.0 or len(boxes) <= 1:
        return list(boxes)

    ordered = sorted(boxes, key=lambda b: b.get("score", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for box in ordered:
        same = [k for k in kept if k["label"] == box["label"]]
        if all(iou(box, k) < iou_threshold for k in same):
            kept.append(box)
    return kept
