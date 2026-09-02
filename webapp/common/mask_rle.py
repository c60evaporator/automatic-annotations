"""インスタンスマスクの RLE 表現.

形式は COCO の「非圧縮 RLE」に合わせる:
    {"size": [height, width], "counts": [run, run, ...]}
counts は**列優先（Fortran order）**の連長で、最初の run は 0（背景）の個数。
pycocotools がそのまま解釈できる形式なので、後で圧縮 RLE へ移行する場合も
互換を保ちやすい。

inference が encode し、webapp が decode して描画するため common に置く。
pycocotools に依存せず numpy だけで読み書きできるようにしてあるのは、
webapp に pycocotools を足すと numpy のバージョンを巻き上げる事故に
つながるため（推論サーバー側で実際に踏んだ）。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """bool の 2D 配列を RLE dict にする.

    Args:
        mask: shape ``(H, W)`` の bool 配列
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")

    height, width = mask.shape
    # 列優先に並べ替えてから連長圧縮する（COCO の定義に合わせる）
    flat = mask.flatten(order="F")
    if flat.size == 0:
        return {"size": [height, width], "counts": []}

    # NOTE: 画素ごとの Python ループにしないこと。
    # 900x1600 で 144 万回まわり、1 マスクあたり数百 ms かかる。
    # 値が変わる位置だけ求めて差分を取れば numpy 側で完結する
    change_positions = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], change_positions, [flat.size]))
    runs = np.diff(boundaries)

    counts = runs.tolist()
    # counts は「0 の run」から始まる約束。先頭が True なら 0 を足して辻褄を合わせる
    if flat[0]:
        counts = [0] + counts

    return {"size": [height, width], "counts": counts}


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    """RLE dict を bool の 2D 配列に戻す."""
    height, width = rle["size"]
    counts = rle["counts"]

    flat = np.zeros(height * width, dtype=bool)
    position = 0
    value = False
    for run in counts:
        if value and run:
            flat[position:position + run] = True
        position += run
        value = not value

    return flat.reshape((height, width), order="F")


def rle_area(rle: dict[str, Any]) -> int:
    """RLE のまま面積（True の画素数）を数える.

    デコードせずに済むので、IoU 判定の前段で軽く弾きたいときに使う。
    """
    counts = rle["counts"]
    return int(sum(counts[1::2]))


def rle_bbox(rle: dict[str, Any]) -> tuple[int, int, int, int]:
    """マスクの外接矩形 ``(xmin, ymin, xmax, ymax)`` を返す.

    空マスクの場合は ``(0, 0, 0, 0)``。
    """
    mask = decode_rle(rle)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    # xmax/ymax は排他（幅 = xmax - xmin）にそろえる
    return (int(xmin), int(ymin), int(xmax) + 1, int(ymax) + 1)


def mask_iou(rle_a: dict[str, Any], rle_b: dict[str, Any]) -> float:
    """2つのマスクの IoU."""
    if rle_a["size"] != rle_b["size"]:
        raise ValueError(
            f"mask size mismatch: {rle_a['size']} vs {rle_b['size']}"
        )
    mask_a = decode_rle(rle_a)
    mask_b = decode_rle(rle_b)
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return intersection / union if union else 0.0


def rle_from_box(
    xmin: int, ymin: int, xmax: int, ymax: int, height: int, width: int
) -> dict[str, Any]:
    """矩形からマスク RLE を作る（スタブ用）."""
    mask = np.zeros((height, width), dtype=bool)
    y0, y1 = max(0, ymin), min(height, ymax)
    x0, x1 = max(0, xmin), min(width, xmax)
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = True
    return encode_rle(mask)
