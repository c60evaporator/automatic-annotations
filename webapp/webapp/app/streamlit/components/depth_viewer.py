"""深度マップの可視化.

深度は単なる float の配列なので、そのままでは見て判断できない。
カラーマップを当てて画像にし、インスタンスマスクを重ねられるようにする。

matplotlib は webapp の依存に入れていない（numpy を巻き上げる事故を避けるため）。
turbo 相当のカラーマップを制御点の線形補間で自前に持つ。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

# turbo カラーマップの制御点（近い＝青、遠い＝赤の並びで使う）
_TURBO_STOPS = np.array([
    [0.19, 0.07, 0.23], [0.27, 0.31, 0.71], [0.10, 0.72, 0.81],
    [0.31, 0.92, 0.44], [0.85, 0.92, 0.20], [0.98, 0.56, 0.16],
    [0.80, 0.16, 0.09],
])


def colorize_depth(
    depth: np.ndarray,
    *,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> Image.Image:
    """深度マップをカラー画像にする.

    範囲を指定しない場合はフレームごとの最小・最大で正規化する。
    フレーム間で色を比較したいときは範囲を明示すること
    （自動だと近景だけの画像と遠景だけの画像が同じ色域になる）。
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8))

    low = float(np.min(depth[valid])) if min_depth is None else float(min_depth)
    high = float(np.max(depth[valid])) if max_depth is None else float(max_depth)
    if high <= low:
        high = low + 1e-3

    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, len(_TURBO_STOPS))
    rgb = np.stack([
        np.interp(normalized, positions, _TURBO_STOPS[:, channel])
        for channel in range(3)
    ], axis=-1)
    rgb[~valid] = 0.0
    return Image.fromarray((rgb * 255).astype(np.uint8))


def render_depth_grid(
    items: list[dict[str, Any]],
    *,
    columns: int = 2,
    draw_fn=None,
) -> None:
    """深度画像をグリッド表示する.

    items の各要素:
        {"channel": str, "image": PIL.Image | None,
         "instances": [...], "caption_suffix": str}

    Args:
        draw_fn: マスクを重ねる関数。None ならそのまま表示する。
            instance_tracking_viewer.draw_instances を想定
    """
    cols = st.columns(columns)
    for index, item in enumerate(items):
        with cols[index % columns]:
            image = item.get("image")
            if image is None:
                st.warning(f"{item['channel']}: 深度マップがありません")
                continue

            instances = item.get("instances") or []
            if draw_fn is not None and instances:
                image = draw_fn(image, instances)

            caption = item["channel"] + item.get("caption_suffix", "")
            st.image(image, caption=caption, width="stretch")
