import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.axes import Axes
from PIL import Image

from ..schemas import Box2D, Instance2D
from ..geometry.crop_resize import (
    paste_cropped_mask,
)

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
    ax.imshow(image)
    ax.set_axis_off()

    # Plot instance masks with prompts
    image_width, image_height = image.size
    plot_instance_masks_with_prompt(
        instances=instances,
        image_height=image_height,
        image_width=image_width,
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
