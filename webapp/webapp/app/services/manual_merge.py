"""再実行時に手修正ボックスを引き継ぐロジック.

考え方:
手で直したボックスは、モデルの出力より信頼できる。
再実行のたびに手修正が失われると、同じ作業を繰り返すことになるため、
新しい推論結果へ引き継ぐ。

規則（sample_data 単位で適用する）:
  1. 手修正ボックスごとに、IoU が最大の推論ボックスを1つ選ぶ
  2. IoU が閾値以上なら、その推論ボックスを手修正ボックスで**置き換える**
  3. 閾値未満・相手がいない手修正ボックスは、そのまま**追加**する
     （モデルが見落とした物体を人が足したケース）
  4. 一度マッチした推論ボックスは、以降の照合から外す（1対1）

照合はカテゴリグループを跨いで行う。
「car と誤検出された歩行者を pedestrian に直す」という修正がグループを
またぐため、グループ内に限定すると同じ物体に 2 つ残ってしまう。

スコープ外:
「誤検出を削除した」という修正は、行が無いことでしか表現できないため
引き継げない。再実行すると同じ誤検出が復活する。
"""
from __future__ import annotations

from typing import Any, Iterable

BoxDict = dict[str, Any]


def box_iou(a: BoxDict, b: BoxDict) -> float:
    """2つの 2D ボックスの IoU."""
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


def _sort_key(box: BoxDict) -> tuple:
    """マッチング順を決める.

    スコアの高い手修正ボックスから処理する。
    同点のときに結果がぶれないよう、座標も含めて完全に決定的にする。
    """
    score = box.get("score")
    return (
        -(score if score is not None else 1.0),
        box["xmin"], box["ymin"], box["xmax"], box["ymax"],
        str(box.get("label", "")),
    )


def merge_manual_boxes(
    predicted: Iterable[BoxDict],
    manual: Iterable[BoxDict],
    iou_threshold: float,
) -> list[BoxDict]:
    """1フレーム分の推論結果に手修正ボックスを反映する.

    Args:
        predicted: 今回の推論結果
        manual: 引き継ぐ手修正ボックス
        iou_threshold: これ以上重なっていれば「同じ物体」とみなす

    Returns:
        マージ後のボックス。手修正由来のものは manually_modified=True。
    """
    predicted_list = list(predicted)
    manual_list = sorted(manual, key=_sort_key)

    used_indices: set[int] = set()
    kept_manual: list[BoxDict] = []

    for manual_box in manual_list:
        best_index = -1
        best_iou = 0.0
        for index, pred_box in enumerate(predicted_list):
            if index in used_indices:
                continue
            iou = box_iou(manual_box, pred_box)
            if iou > best_iou:
                best_iou, best_index = iou, index

        if best_index >= 0 and best_iou >= iou_threshold:
            # 置き換え: 対応する推論ボックスを捨て、手修正ボックスを残す
            used_indices.add(best_index)
        # 閾値未満でもマッチ相手なしでも、手修正ボックスは残す
        kept = dict(manual_box)
        kept["manually_modified"] = True
        kept_manual.append(kept)

    merged = [
        box for index, box in enumerate(predicted_list)
        if index not in used_indices
    ]
    merged.extend(kept_manual)
    return merged


def merge_manual_boxes_by_frame(
    predicted_by_frame: dict[str, list[BoxDict]],
    manual_by_frame: dict[str, list[BoxDict]],
    iou_threshold: float,
) -> dict[str, list[BoxDict]]:
    """sample_data 単位でマージする.

    手修正だけが存在するフレーム（今回の推論対象外だった sample など）も
    結果に含める。sample_interval を変えて再実行したときに、
    前回の手修正が消えないようにするため。
    """
    merged: dict[str, list[BoxDict]] = {}
    for token in set(predicted_by_frame) | set(manual_by_frame):
        merged[token] = merge_manual_boxes(
            predicted_by_frame.get(token, []),
            manual_by_frame.get(token, []),
            iou_threshold,
        )
    return merged
