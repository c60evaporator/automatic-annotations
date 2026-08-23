import numpy as np

from ..schemas import Box2D

def calc_box_ious(
    boxes1: list[Box2D],
    boxes2: list[Box2D],
    match_label: bool = True,
) -> np.ndarray:
    """2Dボックス間のIoUを計算する。

    Args:
        boxes1: N個のBox2D。
        boxes2: M個のBox2D。
        match_label:
            True の場合、box.label が一致するペアのみIoUを計算する。
            ラベルが一致しない場合はIoU=0とする。
            False の場合、ラベルに関係なくIoUを計算する。

    Returns:
        shape (N, M) のIoU行列。
    """
    def get_coordinates(boxes: list[Box2D]) -> np.ndarray:
        if not boxes:
            return np.empty((0, 4), dtype=np.float64)

        return np.asarray(
            [box.xyxy for box in boxes],
            dtype=np.float64,
        )

    coordinates1 = get_coordinates(boxes1)
    coordinates2 = get_coordinates(boxes2)

    sizes1 = np.maximum(
        coordinates1[:, 2:] - coordinates1[:, :2],
        0.0,
    )
    sizes2 = np.maximum(
        coordinates2[:, 2:] - coordinates2[:, :2],
        0.0,
    )

    areas1 = np.prod(sizes1, axis=1)
    areas2 = np.prod(sizes2, axis=1)

    intersection_top_left = np.maximum(
        coordinates1[:, None, :2],
        coordinates2[None, :, :2],
    )
    intersection_bottom_right = np.minimum(
        coordinates1[:, None, 2:],
        coordinates2[None, :, 2:],
    )

    intersection_sizes = np.maximum(
        intersection_bottom_right - intersection_top_left,
        0.0,
    )
    intersections = np.prod(intersection_sizes, axis=-1)

    unions = (
        areas1[:, None]
        + areas2[None, :]
        - intersections
    )

    ious = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )

    if match_label:
        labels1 = np.asarray([box.label for box in boxes1], dtype=object)
        labels2 = np.asarray([box.label for box in boxes2], dtype=object)

        label_matches = labels1[:, None] == labels2[None, :]
        ious = np.where(label_matches, ious, 0.0)

    return ious
