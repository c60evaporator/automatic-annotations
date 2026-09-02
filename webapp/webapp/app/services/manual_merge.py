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

from common.box_ops import greedy_match

BoxDict = dict[str, Any]


def _sort_key(box: BoxDict) -> tuple:
    """手修正ボックスの並び順.

    出力の並びを安定させるために使う。座標まで含めて完全に決定的にする。
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

    # 1 対 1 の貪欲マッチング（common と共有）。
    # ラベルは見ない: 「car と誤検出された歩行者を pedestrian に直す」
    # という修正はグループを跨ぐため、ラベル一致を条件にすると効かない
    matched = greedy_match(
        manual_list, predicted_list, iou_threshold=iou_threshold
    )
    used_indices = set(matched.values())

    kept_manual: list[BoxDict] = []
    for manual_box in manual_list:
        # 置き換えでも追加でも、手修正ボックスは必ず残す
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
