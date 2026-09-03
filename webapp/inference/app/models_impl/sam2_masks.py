"""SAM2 の出力マスクを API のインスタンス形式へ変換する.

参考実装（Instance2D）はマスクを bbox 内に切り出して保持していたが、
このサーバーは COCO 非圧縮 RLE の全画面マスクを返す。
RLE は空白部分が 1 要素にまとまるため、全画面でも保持コストは
「マスクの輪郭の複雑さ」に比例し、bbox 切り出しと大差ない。

torch に依存させていないのは、GPU 無しでも単体テストできるようにするため。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from common.mask_rle import encode_rle


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """bool マスクを最近傍補間で ``(height, width)`` にリサイズする."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if height < 1 or width < 1:
        raise ValueError(f"size must be >= 1, got {width}x{height}")
    h, w = mask.shape
    rows = ((np.arange(height) + 0.5) * h / height).astype(np.int64).clip(0, h - 1)
    cols = ((np.arange(width) + 0.5) * w / width).astype(np.int64).clip(0, w - 1)
    return np.ascontiguousarray(mask[rows][:, cols], dtype=np.bool_)


def masks_to_instances(
    masks: np.ndarray,
    image_height: int,
    image_width: int,
    *,
    local_ids: Sequence[int],
    labels: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    detection_ids: Sequence[str | None] | None = None,
    is_prompt_frame: bool = False,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """マスク配列 ``(C, H, W)`` をインスタンスの dict リストへ変換する.

    Args:
        masks: 0/1 または logits>0 で二値化済みの配列
        local_ids: 各マスクに対応する区間内の仮 ID（SAM2 の obj_id）。
            シーン全体の track_id は呼び出し側が付け替える
        threshold: 二値化の閾値

    Returns:
        空でないマスクだけを、入力順で返す。

    外接矩形は argmax を使って一括で求める。
    マスクごとに np.where を呼ぶと、インスタンス数ぶんの Python ループで
    1600x900 の走査が繰り返されて遅い。
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (C, H, W), got {masks.shape}")
    if masks.shape[0] == 0:
        return []

    if masks.shape[-2:] != (image_height, image_width):
        masks = np.stack([
            resize_mask_nearest(m, image_height, image_width) for m in masks
        ])

    binary = masks > threshold  # (C, H, W)

    rows = binary.any(axis=2)   # (C, H)
    cols = binary.any(axis=1)   # (C, W)
    nonempty = rows.any(axis=1)

    # argmax は最初の True の位置。反転して引くと「終端 + 1」（排他）になる
    y0 = rows.argmax(axis=1)
    y1 = rows.shape[1] - np.flip(rows, axis=1).argmax(axis=1)
    x0 = cols.argmax(axis=1)
    x1 = cols.shape[1] - np.flip(cols, axis=1).argmax(axis=1)

    instances: list[dict[str, Any]] = []
    for index in range(binary.shape[0]):
        if not nonempty[index]:
            continue
        mask = np.ascontiguousarray(binary[index])
        rle = encode_rle(mask)
        instances.append({
            "local_id": int(local_ids[index]),
            "label": None if labels is None else labels[index],
            "score": None if scores is None else (
                None if scores[index] is None else float(scores[index])
            ),
            "mask_rle": rle,
            "mask_area": int(mask.sum()),
            "xmin": int(x0[index]), "ymin": int(y0[index]),
            "xmax": int(x1[index]), "ymax": int(y1[index]),
            "detection_2d_id": (
                None if detection_ids is None else detection_ids[index]
            ),
            "is_prompt_frame": is_prompt_frame,
        })
    return instances
