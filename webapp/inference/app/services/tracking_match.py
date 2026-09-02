"""プロンプト区間をまたぐ track_id の引き継ぎ.

SAM2 には sample_interval ごとにプロンプト（Detection2D のボックス）を与える。
区間の境界となる sample には、
  - 前の区間から**伝播してきた**インスタンス
  - 今回のプロンプトから**新たに得た**インスタンス
の両方が存在する。これらを貪欲マッチングし、IoU が閾値以上なら
前の区間の track_id を引き継ぐ。閾値未満なら新しい track_id を発番する。

これをしないと、区間ごとに track_id が振り直され、
シーン全体を通したトラックにならない。
"""
from __future__ import annotations

from typing import Any, Sequence

from common.box_ops import box_iou, greedy_match
from common.mask_rle import mask_iou

# IoU の計算方法
IOU_METHOD_BOX = "box"
IOU_METHOD_MASK = "mask"

# ラベル一致の要求レベル
LABEL_MATCH_LABEL = "label"
LABEL_MATCH_CATEGORY_GROUP = "category_group"
LABEL_MATCH_NONE = "none"


def instance_iou(
    a: dict[str, Any], b: dict[str, Any], method: str
) -> float:
    """設定に応じて外接矩形 IoU かマスク IoU を返す.

    マスク IoU はデコードを伴うぶん重いが、
    細長い物体や重なりの多い場面では矩形より素直に効く。
    """
    if method == IOU_METHOD_MASK and a.get("mask_rle") and b.get("mask_rle"):
        return mask_iou(a["mask_rle"], b["mask_rle"])
    return box_iou(a, b)


def labels_compatible(
    a: dict[str, Any],
    b: dict[str, Any],
    label_match: str,
    label_to_category_group: dict[str, str] | None = None,
) -> bool:
    """ラベル一致の条件を満たすか.

    'none' を選ぶと、検出のラベルがフレーム間で揺れても
    トラックが途切れなくなる代わりに、別物体を取り違える余地が増える。
    """
    if label_match == LABEL_MATCH_NONE:
        return True
    if label_match == LABEL_MATCH_LABEL:
        return a.get("label") == b.get("label")
    if label_match == LABEL_MATCH_CATEGORY_GROUP:
        mapping = label_to_category_group or {}
        return mapping.get(a.get("label")) == mapping.get(b.get("label"))
    raise ValueError(f"unknown label_match: {label_match}")


def match_instances(
    propagated: Sequence[dict[str, Any]],
    detected: Sequence[dict[str, Any]],
    *,
    iou_threshold: float,
    iou_method: str = IOU_METHOD_BOX,
    label_match: str = LABEL_MATCH_LABEL,
    label_to_category_group: dict[str, str] | None = None,
) -> dict[int, int]:
    """伝播インスタンスと新規検出インスタンスを1対1で貪欲マッチングする.

    Args:
        propagated: 前の区間から伝播してきたインスタンス
        detected: 今回のプロンプトで得たインスタンス

    Returns:
        {detected の index: propagated の index}。
        マッチしなかった detected は含まれない（＝新規 track_id を発番する）。

    貪欲の順序は IoU の降順。組全体で最適にはならないが、
    区間境界の照合では十分で、実装も挙動も追いやすい。
    """
    return greedy_match(
        detected,
        propagated,
        iou_threshold=iou_threshold,
        iou_fn=lambda a, b: instance_iou(a, b, iou_method),
        compatible=lambda a, b: labels_compatible(
            a, b, label_match, label_to_category_group
        ),
    )


def assign_track_ids(
    propagated: Sequence[dict[str, Any]],
    detected: Sequence[dict[str, Any]],
    *,
    next_track_id: int,
    iou_threshold: float,
    iou_method: str = IOU_METHOD_BOX,
    label_match: str = LABEL_MATCH_LABEL,
    label_to_category_group: dict[str, str] | None = None,
) -> tuple[list[str], int]:
    """新規検出インスタンスに track_id を割り当てる.

    Returns:
        (detected と同じ長さの track_id リスト, 次に使う track_id)
    """
    matched = match_instances(
        propagated, detected,
        iou_threshold=iou_threshold,
        iou_method=iou_method,
        label_match=label_match,
        label_to_category_group=label_to_category_group,
    )

    track_ids: list[str] = []
    for d_index in range(len(detected)):
        p_index = matched.get(d_index)
        if p_index is not None:
            track_ids.append(str(propagated[p_index]["track_id"]))
        else:
            track_ids.append(str(next_track_id))
            next_track_id += 1
    return track_ids, next_track_id
