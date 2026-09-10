"""3D バウンディングボックスをカメラ画像へ投影して描く.

参考実装（matplotlib 版 plot_3d_boxes_on_image）と同じ考え方だが、
このアプリは画像描画を PIL で行っているので PIL へ移植してある。
投影そのものは既存の geometry モジュール
（transform_ego_to_camera / project_camera_points）を使う。

ボックスは **ego 座標**で受け取る。Box Fitting の結果も GT の変換結果も
ego 座標なので、そのまま渡せる。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from app.services.geometry.detection import BOX_EDGES, make_box_corners
from app.services.geometry.pointcloud import (
    project_camera_points,
    transform_ego_to_camera,
)
from app.services.geometry.transform import yaw_to_quaternion
from app.streamlit.components.det2d_viewer import (
    BOX_TEXT_MARGIN,
    BOX_TEXT_SIZE,
    BOX_TEXT_STROKE,
    BOX_TEXT_STROKE_COLOR,
    _font,
)
from app.streamlit.components.det2d_viewer import color_for as color_for_label
from app.streamlit.components.instance_tracking_viewer import (
    COLOR_MODE_TRACK,
    TEXT_MODE_LABEL,
    TEXT_MODE_NONE,
    color_for_track,
)

# 3D ボックスの線幅。12 辺あるので細いと画像上で追いにくい
BOX3D_LINE_WIDTH = 5
# カメラ前方と見なす z の下限 [m]
NEAR_PLANE = 0.1


def project_box_to_image(
    box: dict[str, Any],
    calibrated_sensor: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[int, tuple[float, float]]:
    """ego 座標のボックスを画像座標へ投影する.

    Args:
        box: ``{"center_ego", "size_wlh", "yaw_ego"}``

    Returns:
        ``{頂点index: (u, v)}``。カメラ前方に無い頂点は含まれない。

    NOTE: 画像範囲外の点も残す（filter_outside_image=False）。
    画面外で切ると、はみ出した辺が描けなくなる。
    """
    corners_ego = make_box_corners(
        box["center_ego"], box["size_wlh"],
        yaw_to_quaternion(box.get("yaw_ego") or 0.0),
    )
    corners_camera = transform_ego_to_camera(
        corners_ego,
        calibrated_sensor["translation"],
        calibrated_sensor["rotation"],
    )
    corners_uv, valid = project_camera_points(
        corners_camera,
        np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float64),
        image_width, image_height,
        near_plane=NEAR_PLANE,
        filter_outside_image=False,
    )
    # project_camera_points は有効な点だけを返すので、元の頂点 index へ戻す
    return {
        int(corner_index): (float(corners_uv[order][0]), float(corners_uv[order][1]))
        for order, corner_index in enumerate(np.flatnonzero(valid))
    }


def box_center_in_fov(
    box: dict[str, Any],
    calibrated_sensor: dict[str, Any],
    image_width: int,
    image_height: int,
) -> bool:
    """ボックス中心がカメラの画角内にあるか.

    中心で判定するのは、参考実装の filter_boxes_in_camera_fov と同じ方針。
    隣のカメラに写っている物体が端に描かれるのを防ぐ。
    """
    center = np.asarray(box["center_ego"], dtype=np.float64).reshape(1, 3)
    center_camera = transform_ego_to_camera(
        center,
        calibrated_sensor["translation"],
        calibrated_sensor["rotation"],
    )
    _, valid = project_camera_points(
        center_camera,
        np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float64),
        image_width, image_height,
        near_plane=NEAR_PLANE,
        filter_outside_image=True,
    )
    return bool(valid[0])


def color_for_box(box: dict[str, Any], color_mode: str) -> str:
    if color_mode == COLOR_MODE_TRACK:
        return color_for_track(str(box.get("track_id", "")))
    return color_for_label(str(box.get("label", "")))


def legend_key(box: dict[str, Any], color_mode: str) -> str:
    if color_mode == COLOR_MODE_TRACK:
        return str(box.get("track_id", ""))
    return str(box.get("label", ""))


def draw_boxes_3d(
    image: Image.Image,
    boxes: Sequence[dict[str, Any]],
    calibrated_sensor: dict[str, Any],
    *,
    color_mode: str = "Label",
    text_mode: str = TEXT_MODE_NONE,
    fixed_color: str | None = None,
    line_width: int = BOX3D_LINE_WIDTH,
) -> Image.Image:
    """ボックスを投影して重ねた新しい画像を返す.

    Args:
        fixed_color: 指定するとすべて同じ色で描く（GT を一色にしたい場合）

    元画像はキャッシュで共有されているので、必ずコピーしてから描く。
    """
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(BOX_TEXT_SIZE) if text_mode != TEXT_MODE_NONE else None
    width, height = image.size

    for box in boxes:
        projected = project_box_to_image(box, calibrated_sensor, width, height)
        if len(projected) < 2:
            continue

        color = fixed_color or color_for_box(box, color_mode)
        for start_index, end_index in BOX_EDGES:
            # 片方の頂点がカメラ背後なら、その辺は描けない
            if start_index not in projected or end_index not in projected:
                continue
            draw.line(
                [projected[start_index], projected[end_index]],
                fill=color, width=line_width,
            )

        if font is not None:
            text = (
                str(box.get("label", "")) if text_mode == TEXT_MODE_LABEL
                else str(box.get("track_id", ""))
            )
            if not text:
                continue
            # 見えている頂点のうち一番上に置く
            anchor = min(projected.values(), key=lambda uv: uv[1])
            _, top, _, bottom = draw.textbbox((0, 0), text, font=font)
            y = anchor[1] - (bottom - top) - BOX_TEXT_MARGIN
            if y < 0:
                y = anchor[1] + BOX_TEXT_MARGIN
            draw.text(
                (anchor[0], y), text, font=font, fill=color,
                stroke_width=BOX_TEXT_STROKE, stroke_fill=BOX_TEXT_STROKE_COLOR,
            )

    return canvas


def _render_one(
    item: dict[str, Any],
    boxes: Sequence[dict[str, Any]],
    caption: str,
    *,
    color_mode: str,
    text_mode: str,
    fixed_color: str | None = None,
) -> None:
    image = item.get("image")
    if image is None:
        st.warning(f"{item['channel']}: 画像が見つかりません")
        return

    rendered = draw_boxes_3d(
        image, boxes, item["calibrated_sensor"],
        color_mode=color_mode, text_mode=text_mode, fixed_color=fixed_color,
    )
    st.image(rendered, caption=f"{caption}  {len(boxes)} boxes", width="stretch")


def render_box3d_grid(
    items: list[dict[str, Any]],
    *,
    columns: int = 2,
    color_mode: str = "Label",
    text_mode: str = TEXT_MODE_NONE,
) -> None:
    """カメラ画像を格子状に並べ、推定ボックスを重ねる.

    items の各要素:
        ``{"channel", "image", "calibrated_sensor", "boxes"}``
    """
    cols = st.columns(columns)
    for index, item in enumerate(items):
        with cols[index % columns]:
            _render_one(
                item, item.get("boxes") or [], item["channel"],
                color_mode=color_mode, text_mode=text_mode,
            )


def render_box3d_comparison_grid(
    items: list[dict[str, Any]],
    *,
    color_mode: str = "Label",
    text_mode: str = TEXT_MODE_NONE,
    left_label: str = "GT",
    right_label: str = "Fitted",
) -> None:
    """1 カメラにつき 1 行、左に GT・右に推定を並べる.

    items の各要素に ``"gt_boxes"`` と ``"boxes"`` を持たせる。
    """
    for item in items:
        left_col, right_col = st.columns(2)
        with left_col:
            _render_one(
                item, item.get("gt_boxes") or [],
                f"{item['channel']} [{left_label}]",
                color_mode=color_mode, text_mode=text_mode,
            )
        with right_col:
            _render_one(
                item, item.get("boxes") or [],
                f"{item['channel']} [{right_label}]",
                color_mode=color_mode, text_mode=text_mode,
            )
