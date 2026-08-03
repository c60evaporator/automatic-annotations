import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

from .nuscenes_utils import (
    make_box_corners_ego,
    project_camera_points,
    transform_ego_to_camera,
)
from .schemas import Box3D


_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def plot_bboxes_on_image(
    image: Image.Image,
    bboxes: list[Box3D],
    camera_translation: np.ndarray | list[float],
    camera_rotation: np.ndarray | list[float],
    camera_intrinsic: np.ndarray,
    ax: Axes | None = None,
    color: str | tuple[float, float, float] = "lime",
    line_width: int = 2,
    near_plane: float = 0.1,
) -> Axes:
    """ego座標系の3D bounding boxを画像へ投影して描画する。

    Args:
        image: Matplotlib上に表示する画像。
        bboxes: ego座標系の3D bounding boxのリスト。
        camera_translation: カメラ原点のego座標 ``(x, y, z)``。
        camera_rotation: camera座標からego座標への回転クォータニオン
            ``(w, x, y, z)``。
        camera_intrinsic: shape ``(3, 3)`` のカメラ内部パラメータ。
        ax: 描画先のMatplotlib Axes。Noneの場合は新しく生成する。
        color: bboxの描画色。
        line_width: bboxの線幅[px]。
        near_plane: カメラ前方の点とみなすz座標の閾値[m]。

    Returns:
        画像とbboxを描画したMatplotlib ``Axes``。
    """
    if line_width <= 0:
        raise ValueError(f"line_width must be positive, got {line_width}")

    if ax is None:
        _, ax = plt.subplots()

    image_width, image_height = image.size
    ax.imshow(image)
    ax.set_axis_off()

    for box in bboxes:
        corners_ego = make_box_corners_ego(box)
        corners_camera = transform_ego_to_camera(
            corners_ego,
            camera_translation,
            camera_rotation,
        )
        corners_uv, valid = project_camera_points(
            corners_camera,
            camera_intrinsic,
            image_width,
            image_height,
            near_plane,
            filter_outside_image=False,
        )

        # project_camera_pointsはvalidな点のみ返すため、元の頂点indexへ戻す。
        projected_by_index = {
            int(corner_index): tuple(map(float, corners_uv[:, projected_index]))
            for projected_index, corner_index in enumerate(np.flatnonzero(valid))
        }
        for start_index, end_index in _BOX_EDGES:
            if start_index not in projected_by_index or end_index not in projected_by_index:
                continue
            start = projected_by_index[start_index]
            end = projected_by_index[end_index]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=line_width,
            )

    # 画面外の投影点によるautoscaleを抑え、線を画像境界でclipさせる。
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)

    return ax
