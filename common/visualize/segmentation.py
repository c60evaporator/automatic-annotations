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
    alpha: float = 0.6,
    text_shown_attr: str | None = None,
) -> Axes:
    """2D instance maskを描画する。

    Args:
        instance: 2D instance mask object.
        image_height: 元画像の高さ[px]。
        image_width: 元画像の幅[px]。
        ax: 描画先のMatplotlib Axes。Noneの場合はplt.gca()を使用する。
        color: マスクとインスタンスBoxの描画色。
        alpha: マスクの透明度。0.0は完全に透明、1.0は完全に不透明。
        text_shown_attr: テキストとして描画するinstance.boxの属性。"label"または"track_id"を指定可能。Noneの場合は描画しない。
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"image_height and image_width must be positive, got {image_height}, {image_width}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
    if text_shown_attr is not None and getattr(instance.box, text_shown_attr, None) is None:
            raise ValueError(f"box.{text_shown_attr} must be provided when text_shown_attr is not None")

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
    mask_overlay[global_mask > 0] = [*color_rgb, alpha]
    ax.imshow(mask_overlay, interpolation="none")

    # Plot the text for the instance box
    if text_shown_attr is not None:
        text_value = getattr(instance.box, text_shown_attr)
        if text_value is not None:
            x1, y1, x2, y2 = instance.box.xyxy
            ax.text(
                x1,
                y1 - 5,
                str(text_value),
                color=color_rgb,
                fontsize=12,
                fontweight="bold",
                verticalalignment="bottom",
                horizontalalignment="left",
            )


def get_rgb_color(color, instance, color_attr, allow_dict=True):
    if isinstance(color, dict):
        if not allow_dict:
            raise ValueError("Dictionaries are not allowed for color")
        if getattr(instance.box, color_attr, None) is None:
            raise ValueError(f"box.{color_attr} must be provided when color is a dict")
        if getattr(instance.box, color_attr) not in color:
            raise ValueError(f"box.{color_attr} value {getattr(instance.box, color_attr)} not found in color dict")
        color = color[getattr(instance.box, color_attr)]
    if isinstance(color, str):
        color_rgb = to_rgb(color)
    elif isinstance(color, tuple) and len(color) == 3:
        color_rgb = color
    else:
        raise ValueError(f"Invalid color value: {color}. Must be a string or a tuple of 3 floats.")
    return color_rgb


def plot_instance_mask_with_prompt(
    instance: Instance2D,
    image_height: int,
    image_width: int,
    prompt_points: list[tuple[int, int]] | None = None,
    prompt_labels: list[int] | None = None,
    prompt_box: Box2D | None = None,
    ax: Axes | None = None,
    alpha: float = 0.6,
    color: str | tuple[float, float, float] | dict[str, str] | dict[tuple[float, float, float], str] = "lime",
    prompt_box_color: str | tuple[float, float, float] | dict[str, str] | dict[tuple[float, float, float], str] = "red",
    color_attr: str = "label",
    pos_prompt_color: str | tuple[float, float, float] = "yellow",
    neg_prompt_color: str | tuple[float, float, float] = "navy",
    line_width: int = 2,
    text_shown_attr: str | None = None,
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
        alpha: マスクの透明度。0.0は完全に透明、1.0は完全に不透明。
        color: マスクの描画色。dictの場合は、``color_attr``で指定した``instances.box``の属性に応じて色を変える。
        prompt_box_color: プロンプトのBoxの描画色。dictの場合は、``color_attr``で指定した``instances.box``の属性に応じて色を変える。
        color_attr: ``color``、``prompt_box_color``がdictの場合に使用する``instances.box``の属性名。"label"または"track_id"を指定可能。
        pos_prompt_color: foregroundプロンプトの点の描画色。
        neg_prompt_color: backgroundプロンプトの点の描画色。
        line_width: バウンディングボックスの線幅[px]。
        text_shown_attr: テキストとして描画するinstance.boxの属性。"label"または"track_id"を指定可能。Noneの場合は描画しない。

    Returns:
        画像とマスクを描画したMatplotlib ``Axes``。
    """
    if (prompt_points is None) ^ (prompt_labels is None):
        raise ValueError("Both prompt_labels and prompt_points should be provided")
    if prompt_points is None:
        prompt_points = []
    if prompt_labels is None:
        prompt_labels = []
    if text_shown_attr is not None and getattr(instance.box, text_shown_attr, None) is None:
        raise ValueError(f"box.{text_shown_attr} must be provided when text_shown_attr is not None")

    if ax is None:
        ax = plt.gca()

    # Create a RGBA from the color input
    color_rgb = get_rgb_color(color, instance, color_attr, allow_dict=True)
    prompt_box_color_rgb = get_rgb_color(prompt_box_color, instance, color_attr, allow_dict=True)
    pos_prompt_color_rgb = get_rgb_color(pos_prompt_color, instance, color_attr, allow_dict=False)
    neg_prompt_color_rgb = get_rgb_color(neg_prompt_color, instance, color_attr, allow_dict=False)

    # Plot the instance mask
    plot_instance_mask(
        instance=instance,
        image_height=image_height,
        image_width=image_width,
        ax=ax,
        color=color_rgb,
        alpha=alpha,
        text_shown_attr=text_shown_attr,
    )

    # Plot the prompt points
    for prompt_point, prompt_label in zip(prompt_points, prompt_labels):
        ax.scatter(
            prompt_point[0],
            prompt_point[1],
            color=pos_prompt_color_rgb if prompt_label == 1 else neg_prompt_color_rgb,
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
            edgecolor=prompt_box_color_rgb,
            facecolor="none",
        )
        ax.add_patch(rect)


def plot_instance_masks_on_image(
    instances: list[Instance2D],
    image: Image.Image,
    prompt_points: list[tuple[int, int]] | None = None,
    prompt_labels: list[int] | None = None,
    prompt_boxes: list[Box2D] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    image_alpha: float = 1.0,
    mask_alpha: float = 0.6,
    color: str | tuple[float, float, float] | dict[str, str] | dict[tuple[float, float, float], str] = "lime",
    prompt_box_color: str | tuple[float, float, float] | dict[str, str] | dict[tuple[float, float, float], str] = "red",
    color_attr: str = "label",
    pos_prompt_color: str | tuple[float, float, float] = "yellow",
    neg_prompt_color: str | tuple[float, float, float] = "navy",
    line_width: int = 2,
    text_shown_attr: str | None = None,
) -> Axes:
    """2D instance maskを描画し、プロンプトを重ねる（SAM, SAM2を想定）。

    Args:
        instances: 2D instance maskオブジェクトのリスト。
        image: Matplotlib上に表示する画像。
        prompt_points: プロンプトの点 ``(x, y)`` のリスト。Noneの場合は描画しない。
        prompt_labels: プロンプトの点のラベル。1はforeground、0はbackground。Noneの場合は描画しない。
        prompt_boxes: プロンプトのバウンディングボックスのリスト。instancesとインデックスが対応している必要がある。Noneの場合は描画しない。
        ax: 描画先のMatplotlib Axes。Noneの場合はplt.gca()を使用する。
        title: Axesのタイトル。Noneの場合は設定しない。
        image_alpha: 画像の透明度。0.0は完全に透明、1.0は完全に不透明。
        mask_alpha: マスクの透明度。0.0は完全に透明、1.0は完全に不透明。
        color: マスクの描画色。dictの場合は、``color_attr``で指定した``instances.box``の属性に応じて色を変える。
        prompt_box_color: プロンプトのBoxの描画色。dictの場合は、``color_attr``で指定した``instances.box``の属性に応じて色を変える。
        color_attr: ``color``、``prompt_box_color``がdictの場合に使用する``instances.box``の属性名。"label"または"track_id"を指定可能。
        pos_prompt_color: foregroundプロンプトの点の描画色。
        neg_prompt_color: backgroundプロンプトの点の描画色。
        line_width: マスクの線幅[px]。
        text_shown_attr: テキストとして描画するinstance.boxの属性。"label"または"track_id"を指定可能。Noneの場合は描画しない。

    Returns:
        画像とマスクを描画したMatplotlib ``Axes``。
    """
    if ax is None:
        ax = plt.gca()

    # Plot the image
    ax.imshow(image, alpha=image_alpha)
    ax.set_axis_off()

    for i_instance in range(len(instances)):
        # Plot instance mask with prompts
        image_width, image_height = image.size
        plot_instance_mask_with_prompt(
            instance=instances[i_instance],
            image_height=image_height,
            image_width=image_width,
            prompt_points=prompt_points,
            prompt_labels=prompt_labels,
            prompt_box=prompt_boxes[i_instance] if prompt_boxes is not None else None,
            ax=ax,
            alpha=mask_alpha,
            color=color,
            prompt_box_color=prompt_box_color,
            color_attr=color_attr,
            pos_prompt_color=pos_prompt_color,
            neg_prompt_color=neg_prompt_color,
            line_width=line_width,
            text_shown_attr=text_shown_attr,
        )

    # 画面外の投影点によるautoscaleを抑え、線を画像境界でclipさせる。
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.axis("off")

    if title is not None:
        ax.set_title(title)

    return ax
