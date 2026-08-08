import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.axes import Axes
from PIL import Image

from .nuscenes_utils import (
    make_box_corners_ego,
    project_camera_points,
    transform_ego_to_camera,
    BOX_EDGES
)
from .schemas import Box3D, Box2D, Instance2D
from .geometry import paste_cropped_mask

TABLEAU10_NAMES: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
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


def plot_instance_masks_with_prompt(
    instances: list[Instance2D],
    image_height: int,
    image_width: int,
    plot_instance_boxes: bool = False,
    input_points: list[tuple[int, int]] | None = None,
    input_labels: list[int] | None = None,
    input_boxes: list[Box2D] | None = None,
    ax: Axes | None = None,
    color: str | tuple[float, float, float] = "lime",
    prompt_box_color: str | tuple[float, float, float] = "red",
    pos_prompt_color: str | tuple[float, float, float] = "yellow",
    neg_prompt_color: str | tuple[float, float, float] = "navy",
    line_width: int = 2,
) -> Axes:
    """2D instance maskを描画し、プロンプトを重ねる（SAM, SAM2を想定）。

    Args:
        instances: A list of 2D instance mask objects.
        image_height: 元画像の高さ[px]。
        image_width: 元画像の幅[px]。
        plot_instance_boxes: Trueの場合は、instance_masksのboxを描画する。Falseの場合は描画しない。
        input_points: プロンプトの点 ``(x, y)`` のリスト。Noneの場合は描画しない。
        input_labels: プロンプトの点のラベル。1はforeground、0はbackground。Noneの場合は描画しない。
        input_boxes: プロンプトのバウンディングボックスのリスト。Noneの場合は描画しない。
        ax: 描画先のMatplotlib Axes。Noneの場合はplt.gca()を使用する。
        color: マスクとインスタンスBoxの描画色。
        prompt_box_color: str | tuple[float, float, float] = "red",
        pos_prompt_color: foregroundプロンプトの点の描画色。
        neg_prompt_color: backgroundプロンプトの点の描画色。
        line_width: マスクの線幅[px]。

    Returns:
        画像とマスクを描画したMatplotlib ``Axes``。
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"image_height and image_width must be positive, got {image_height}, {image_width}")

    if line_width <= 0:
        raise ValueError(f"line_width must be positive, got {line_width}")

    if ax is None:
        ax = plt.gca()

    if (input_points is None) ^ (input_labels is None):
        raise ValueError("Both input_labels and input_points should be provided")
    if input_points is None:
        input_points = []
    if input_labels is None:
        input_labels = []
    if input_boxes is None:
        input_boxes = []

    # Create a RGBA from the color input
    if isinstance(color, str):
        color_rgb = to_rgb(color)
    else:
        color_rgb = color

    for instance in instances:
        # Create a mask with the same size as the original image
        global_mask = paste_cropped_mask(
            cropped_mask=instance.mask,
            original_height=image_height,
            original_width=image_width,
            crop_xyxy=instance.mask_region,
        )
        # Plot the mask
        mask_overlay = np.zeros((image_height, image_width, 4), dtype=np.float32)
        mask_overlay[global_mask > 0] = [*color_rgb, 0.5]
        ax.imshow(mask_overlay, interpolation="none")

    # Plot the prompt points
    for input_point, input_label in zip(input_points, input_labels):
        ax.scatter(
            input_point[0],
            input_point[1],
            color=pos_prompt_color if input_label == 1 else neg_prompt_color,
            marker="*" if input_label == 1 else "o",
            s=250,
            edgecolor="black",
        )

    # Plot the prompt boxes
    for input_box in input_boxes:
        x1, y1, x2, y2 = input_box.xyxy
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=line_width,
            edgecolor=prompt_box_color,
            facecolor="none",
        )
        ax.add_patch(rect)

    # Plot the instance box if requested
    if plot_instance_boxes:
        for instance in instances:
            x1, y1, x2, y2 = instance.box.xyxy
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=line_width,
                edgecolor=color_rgb,
                facecolor="none",
            )
            ax.add_patch(rect)


def plot_instance_masks_on_image(
    instances: list[Instance2D],
    image: Image.Image,
    plot_instance_boxes: bool = True,
    input_points: list[tuple[int, int]] | None = None,
    input_labels: list[int] | None = None,
    input_boxes: list[Box2D] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    color: str | tuple[float, float, float] = "lime",
    prompt_box_color: str | tuple[float, float, float] = "red",
    pos_prompt_color: str | tuple[float, float, float] = "yellow",
    neg_prompt_color: str | tuple[float, float, float] = "navy",
    line_width: int = 2,
) -> Axes:
    """2D instance maskを描画し、プロンプトを重ねる（SAM, SAM2を想定）。

    Args:
        instances: A list of 2D instance mask objects.
        image: Matplotlib上に表示する画像。
        plot_instance_boxes: Trueの場合は、instance_masksのboxを描画する。Falseの場合は描画しない。
        input_points: プロンプトの点 ``(x, y)`` のリスト。Noneの場合は描画しない。
        input_labels: プロンプトの点のラベル。1はforeground、0はbackground。Noneの場合は描画しない。
        input_boxes: プロンプトのバウンディングボックスのリスト。Noneの場合は描画しない。
        ax: 描画先のMatplotlib Axes。Noneの場合はpl
        title: Axesのタイトル。Noneの場合は設定しない。
        color: マスクとインスタンスBoxの描画色。
        prompt_box_color: プロンプトのBoxの描画色。
        pos_prompt_color: foregroundプロンプトの点の描画色。
        neg_prompt_color: backgroundプロンプトの点の描画色。
        line_width: マスクの線幅[px]。

    Returns:
        画像とマスクを描画したMatplotlib ``Axes``。
    """
    if ax is None:
        ax = plt.gca()

    # Plot the image
    image_width, image_height = image.size
    ax.imshow(image)
    ax.set_axis_off()

    # Plot instance masks with prompts
    plot_instance_masks_with_prompt(
        instances=instance_masks,
        image_height=image.height,
        image_width=image.width,
        plot_instance_boxes=plot_instance_boxes,
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        ax=ax,
        color=color,
        prompt_box_color=prompt_box_color,
        pos_prompt_color=pos_prompt_color,
        neg_prompt_color=neg_prompt_color,
        line_width=line_width,
    )

    # 画面外の投影点によるautoscaleを抑え、線を画像境界でclipさせる。
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.axis("off")

    if title is not None:
        ax.set_title(title)

    return ax
