"""ボックス周辺の切り出し.

再判定（zero-shot 分類）はボックス内だけを見ると当たらない。
路面や隣接物が少し写っていないと「柵か縁石か」「絵か実車か」の区別が
付かないため、マージンを付けて切り出す。
"""
from __future__ import annotations

import math
from typing import Any

from PIL import Image


def crop_image_with_margin(
    image: Image.Image,
    box: dict[str, Any],
    margin_ratio: float = 0.1,
) -> Image.Image:
    """ボックスを margin_ratio ぶん広げて切り出す.

    Args:
        box: ``xmin/ymin/xmax/ymax``（画素座標）
        margin_ratio: 各方向へ広げる比率（幅・高さに対する割合）

    Returns:
        切り出した画像。画像範囲でクリップするため、
        端の物体では指定より狭くなる。

    NOTE: 切り出し後の幅・高さが 0 にならないよう、最低 1 画素は残す。
    ボックスが画像の外を指している場合に PIL が空画像を返すのを防ぐ。
    """
    width, height = image.size
    x0, y0 = float(box["xmin"]), float(box["ymin"])
    x1, y1 = float(box["xmax"]), float(box["ymax"])

    margin_x = (x1 - x0) * margin_ratio
    margin_y = (y1 - y0) * margin_ratio

    left = max(0, math.floor(x0 - margin_x))
    top = max(0, math.floor(y0 - margin_y))
    right = min(width, math.ceil(x1 + margin_x))
    bottom = min(height, math.ceil(y1 + margin_y))

    if right <= left:
        left = min(max(0, left), max(0, width - 1))
        right = left + 1
    if bottom <= top:
        top = min(max(0, top), max(0, height - 1))
        bottom = top + 1

    return image.crop((left, top, right, bottom))
