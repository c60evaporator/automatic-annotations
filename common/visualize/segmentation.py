import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.axes import Axes
from PIL import Image

from ..schemas import Box2D, Instance2D
from ..geometry.crop_resize import (
    paste_cropped_mask,
)

def plot_instance_mask(
    instance: Instance2D,
    image_height: int,
    image_width: int,
    ax: Axes | None = None,
    color: str | tuple[float, float, float] = "lime",
) -> Axes:
    """2D instance maskを描画する。

    Args:
        instance: 2D instance mask object.
        image_height: 元画像の高さ[px]。
        image_width: 元画像の幅[px]。
        ax: 描画先のMatplotlib Axes。Noneの場合はplt.gca()を使用する。
        color: マスクとインスタンスBoxの描画色。
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"image_height and image_width must be positive, got {image_height}, {image_width}")

    if ax is None:
        ax = plt.gca()

    # Create a RGBA from the color input
    if isinstance(color, str):
        color_rgb = to_rgb(color)
    elif isinstance(color, tuple) and len(color) == 3:
        color_rgb = color
    else:
        raise ValueError(f"Invalid color value: {color}. Must be a string or a tuple of 3 floats.")

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


def plot_instance_mask_with_prompt(
    instance: Instance2D,
    image_height: int,
    image_width: int,
    prompt_points: list[tuple[int, int]] | None = None,
    prompt_labels: list[int] | None = None,
    prompt_box: Box2D | None = None,
    ax: Axes | None = None,
    color: str | tuple[float, float, float] = "lime",
    prompt_box_color: str | tuple[float, float, float] = "red",
    pos_prompt_color: str | tuple[float, float, float] = "yellow",
    neg_prompt_color: str | tuple[float, float, float] = "navy",
    line_width: int = 2,
) -> Axes:
    """2D instance maskを描画し、プロンプトを重ねる（SAM, SAM2を想定）。

    Args:
        instances: 2D instance mask object.
        image_height: 元画像の高さ[px]。
        image_width: 元画像の幅[px]。
        prompt_points: プロンプトの点 ``(x, y)`` のリスト。Noneの場合は描画しない。
        prompt_labels: プロンプトの点のラベル。1はforeground、0はbackground。Noneの場合は描画しない。
        prompt_box: プロンプトのバウンディングボックス。Noneの場合は描画しない。
        ax: 描画先のMatplotlib Axes。Noneの場合はplt.gca()を使用する。
        color: マスクとインスタンスBoxの描画色。
        prompt_box_color: str | tuple[float, float, float] = "red",
        pos_prompt_color: foregroundプロンプトの点の描画色。
        neg_prompt_color: backgroundプロンプトの点の描画色。
        line_width: バウンディングボックスの線幅[px]。

    Returns:
        画像とマスクを描画したMatplotlib ``Axes``。
    """
    if (prompt_points is None) ^ (prompt_labels is None):
        raise ValueError("Both prompt_labels and prompt_points should be provided")
    if prompt_points is None:
        prompt_points = []
    if prompt_labels is None:
        prompt_labels = []

    if ax is None:
        ax = plt.gca()

    # Create a RGBA from the color input
    if isinstance(color, str):
        color_rgb = to_rgb(color)
    elif isinstance(color, tuple) and len(color) == 3:
        color_rgb = color
    else:
        raise ValueError(f"Invalid color value: {color}. Must be a string or a tuple of 3 floats.")

    # Plot the instance mask
    plot_instance_mask(
        instance=instance,
        image_height=image_height,
        image_width=image_width,
        ax=ax,
        color=color_rgb,
    )

    # Plot the prompt points
    for prompt_point, prompt_label in zip(prompt_points, prompt_labels):
        ax.scatter(
            prompt_point[0],
            prompt_point[1],
            color=pos_prompt_color if prompt_label == 1 else neg_prompt_color,
            marker="*" if prompt_label == 1 else "o",
            s=250,
            edgecolor="black",
        )

    # Plot the prompt box
    if prompt_box is not None:
        x1, y1, x2, y2 = prompt_box.xyxy
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=line_width,
            edgecolor=prompt_box_color,
            facecolor="none",
        )
        ax.add_patch(rect)


def plot_instance_mask_on_image(
    instance: Instance2D,
    image: Image.Image,
    prompt_points: list[tuple[int, int]] | None = None,
    prompt_labels: list[int] | None = None,
    prompt_box: Box2D | None = None,
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
        instance: 2D instance maskオブジェクト。
        image: Matplotlib上に表示する画像。
        prompt_points: プロンプトの点 ``(x, y)`` のリスト。Noneの場合は描画しない。
        prompt_labels: プロンプトの点のラベル。1はforeground、0はbackground。Noneの場合は描画しない。
        prompt_box: プロンプトのバウンディングボックス。Noneの場合は描画しない。
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
    plot_instance_mask_with_prompt(
        instance=instance,
        image_height=image_height,
        image_width=image_width,
        prompt_points=prompt_points,
        prompt_labels=prompt_labels,
        prompt_box=prompt_box,
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
