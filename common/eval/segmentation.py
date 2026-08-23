import numpy as np

from ..schemas import Instance2D

def calc_mask_ious(
    instances1: list[Instance2D],
    instances2: list[Instance2D],
    match_label: bool = True,
) -> np.ndarray:
    """
    2つの Instance2D リスト間で pairwise mask IoU を計算する。

    各 Instance2D.mask は全画面マスクではなく、mask_region で示される
    画像領域のみを保持していることを前提とする。

    Args:
        instances1:
            Instance2D のリスト。
        instances2:
            Instance2D のリスト。
        match_label:
            True の場合、box.label が一致するペアのみ IoU を計算する。
            ラベルが一致しない場合は IoU = 0 とする。
            False の場合、ラベルに関係なく IoU を計算する。

    Returns:
        np.ndarray:
            shape (len(instances1), len(instances2)) の IoU 行列。
            dtype は np.float64。
    """
    ious = np.zeros(
        (len(instances1), len(instances2)),
        dtype=np.float64,
    )

    # 各マスクの面積はペアごとに変わらないので事前計算
    areas1 = [
        np.count_nonzero(instance.mask)
        if instance.mask is not None
        else 0
        for instance in instances1
    ]
    areas2 = [
        np.count_nonzero(instance.mask)
        if instance.mask is not None
        else 0
        for instance in instances2
    ]

    for i, instance1 in enumerate(instances1):
        if instance1.mask is None or instance1.mask_region is None:
            continue

        x0_1, y0_1, x1_1, y1_1 = instance1.mask_region

        for j, instance2 in enumerate(instances2):
            if instance2.mask is None or instance2.mask_region is None:
                continue

            if match_label and instance1.box.label != instance2.box.label:
                continue

            x0_2, y0_2, x1_2, y1_2 = instance2.mask_region

            # mask_region 同士の重複領域
            overlap_x0 = max(x0_1, x0_2)
            overlap_y0 = max(y0_1, y0_2)
            overlap_x1 = min(x1_1, x1_2)
            overlap_y1 = min(y1_1, y1_2)

            # bbox 領域自体が重なっていなければ mask IoU も 0
            if overlap_x0 >= overlap_x1 or overlap_y0 >= overlap_y1:
                continue

            # 重複領域を各ローカル mask 座標へ変換
            mask1_overlap = instance1.mask[
                overlap_y0 - y0_1 : overlap_y1 - y0_1,
                overlap_x0 - x0_1 : overlap_x1 - x0_1,
            ]
            mask2_overlap = instance2.mask[
                overlap_y0 - y0_2 : overlap_y1 - y0_2,
                overlap_x0 - x0_2 : overlap_x1 - x0_2,
            ]

            intersection = np.count_nonzero(
                np.logical_and(mask1_overlap, mask2_overlap)
            )

            union = areas1[i] + areas2[j] - intersection

            if union > 0:
                ious[i, j] = intersection / union

    return ious
