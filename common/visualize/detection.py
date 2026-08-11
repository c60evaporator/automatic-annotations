import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

from ..schemas import Box3D, Box2D
from ..geometry.detection import (
    make_box_corners_ego,
    BOX_EDGES,
)
from ..geometry.pointcloud import (
    project_camera_points,
    transform_ego_to_camera,
)

def plot_3d_boxes_on_image(
    image: Image.Image,
    boxes_3d_ego: list[Box3D],
    camera_translation: np.ndarray | list[float],
    camera_rotation: np.ndarray | list[float],
    camera_intrinsic: np.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
    color: str | tuple[float, float, float] = "lime",
    line_width: int = 2,
    near_plane: float = 0.1,
) -> Axes:
    """ego座標系の3D bounding boxを画像へ投影して描画する。

    Args:
        image: Matplotlib上に表示する画像。
        boxes_3d_ego: ego座標系の3D bounding boxのリスト。
        camera_translation: カメラ原点のego座標 ``(x, y, z)``。
        camera_rotation: camera座標からego座標への回転クォータニオン
            ``(w, x, y, z)``。
        camera_intrinsic: shape ``(3, 3)`` のカメラ内部パラメータ。
        ax: 描画先のMatplotlib Axes。Noneの場合はplt.gca()を使用する。
        title: Axesのタイトル。Noneの場合は設定しない。
        color: bboxの描画色。
        line_width: bboxの線幅[px]。
        near_plane: カメラ前方の点とみなすz座標の閾値[m]。

    Returns:
        画像とbboxを描画したMatplotlib ``Axes``。
    """
    if line_width <= 0:
        raise ValueError(f"line_width must be positive, got {line_width}")

    if ax is None:
        ax = plt.gca()

    image_width, image_height = image.size
    ax.imshow(image)
    ax.set_axis_off()

    for box in boxes_3d_ego:
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
            int(corner_index): tuple(map(float, corners_uv[projected_index]))
            for projected_index, corner_index in enumerate(np.flatnonzero(valid))
        }
        for start_index, end_index in BOX_EDGES:
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

    if title is not None:
        ax.set_title(title)


def plot_2d_boxes(
    boxes_2d: list[Box2D],
    ax: Axes | None = None,
    color: str | tuple[float, float, float] | dict[str, str] | dict[str, tuple[float, float, float]] = "lime",
    line_width: int = 2,
) -> Axes:
    """2D bounding boxを描画する。

    Args:
        boxes_2d: 2D bounding boxのリスト。
        ax: 描画先のMatplotlib Axes。Noneの場合はplt.gca()を使用する。
        color: bboxの描画色。dictの場合は、bboxのlabelに応じて色を変える。
        line_width: bboxの線幅[px]。

    Returns:
        画像とbboxを描画したMatplotlib ``Axes``。
    """
    if line_width <= 0:
        raise ValueError(f"line_width must be positive, got {line_width}")

    if ax is None:
        ax = plt.gca()

    # Plot each box
    for box in boxes_2d:
        if isinstance(color, dict) and box.label is None:
            raise ValueError("box.label must be provided when color is a dict")
        x1, y1, x2, y2 = box.xyxy
        color_to_use = color if not isinstance(color, dict) else color.get(box.label, "lime")
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color_to_use, 
                linewidth=line_width, label=box.label if isinstance(color, dict) else None)


def plot_2d_boxes_on_image(
    image: Image.Image,
    boxes_2d: list[Box2D],
    ax: Axes | None = None,
    title: str | None = None,
    color: str | tuple[float, float, float] | dict[str, str] | dict[str, tuple[float, float, float]] = "lime",
    line_width: int = 2,
) -> Axes:
    """2D bounding boxを画像と一緒に描画する。

    Args:
        image: Matplotlib上に表示する画像。
        boxes_2d: 2D bounding boxのリスト。
        ax: 描画先のMatplotlib Axes。Noneの場合はplt.gca()を使用する。
        title: Axesのタイトル。Noneの場合は設定しない。
        color: bboxの描画色。dictの場合は、bboxのlabelに応じて色を変える。
        line_width: bboxの線幅[px]。

    Returns:
        画像とbboxを描画したMatplotlib ``Axes``。
    """
    if ax is None:
        ax = plt.gca()

    image_width, image_height = image.size
    ax.imshow(image)
    ax.set_axis_off()

    # Plot the boxes
    plot_2d_boxes(boxes_2d, ax=ax, color=color, line_width=line_width)

    # Add legend if color is a dict
    if isinstance(color, dict):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    # 画面外の投影点によるautoscaleを抑え、線を画像境界でclipさせる。
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)

    if title is not None:
        ax.set_title(title)
