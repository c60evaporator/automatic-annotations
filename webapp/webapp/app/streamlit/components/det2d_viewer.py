"""カメラ画像上に 2D bounding box を描画するコンポーネント."""
from __future__ import annotations

from typing import Any, Sequence

import streamlit as st
from PIL import Image, ImageDraw

# ラベルごとの色。未知のラベルはハッシュで決める
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def color_for(label: str) -> str:
    return PALETTE[hash(label) % len(PALETTE)]


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[dict[str, Any]],
    *,
    width: int = 3,
    show_score: bool = True,
    min_score: float = 0.0,
) -> Image.Image:
    """画像に BBox を描いた新しい画像を返す.

    元画像はキャッシュ（cache_resource）で共有されているので、
    必ず copy してから描く。直接描くとキャッシュ内の画像が汚染され、
    再描画のたびにボックスが重なって増えていく。
    """
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for box in boxes:
        if box.get("score", 1.0) < min_score:
            continue
        color = color_for(box["label"])
        xy = (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
        draw.rectangle(xy, outline=color, width=width)

        text = box["label"]
        if show_score and box.get("score") is not None:
            text += f" {box['score']:.2f}"
        # ラベル背景を敷いて可読性を上げる
        tl, tt, tr, tb = draw.textbbox((0, 0), text)
        pad = 2
        bx, by = box["xmin"], max(0, box["ymin"] - (tb - tt) - pad * 2)
        draw.rectangle(
            (bx, by, bx + (tr - tl) + pad * 2, by + (tb - tt) + pad * 2), fill=color
        )
        draw.text((bx + pad, by + pad), text, fill="white")

    return canvas


def render_camera_grid(
    items: list[dict[str, Any]],
    *,
    columns: int = 2,
    min_score: float = 0.0,
    show_boxes: bool = True,
) -> None:
    """カメラ画像をグリッド表示する.

    items の各要素:
        {"channel": str, "image": PIL.Image | None,
         "boxes": [...], "pending": bool}
    """
    cols = st.columns(columns)
    for i, item in enumerate(items):
        with cols[i % columns]:
            image = item.get("image")
            boxes = item.get("boxes") or []
            caption = item["channel"]
            if item.get("pending"):
                caption += "（推論待ち）"
            elif show_boxes:
                shown = [b for b in boxes if b.get("score", 1.0) >= min_score]
                caption += f"  {len(shown)} boxes"

            if image is None:
                st.warning(f"{item['channel']}: 画像が見つかりません")
                continue

            if show_boxes and boxes:
                image = draw_boxes(image, boxes, min_score=min_score)
            st.image(image, caption=caption, width="stretch")
