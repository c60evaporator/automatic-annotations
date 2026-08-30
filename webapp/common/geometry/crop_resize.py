import numpy as np


###### Crop & Resize helpers ######
def rect_to_original(
    xyxy: np.ndarray,
    original_width: int,
    original_height: int,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    resized_width: float | None = None,
    resized_height: float | None = None,
    ) -> np.ndarray:
    """BBox等の矩形を元画像座標系に変換する。以下の順で変換された前提とする

    original_image_size → crop_xyxyの範囲で切出 → resized_width/heightにリサイズ

    Args:
        original_width: 元画像の幅[px]。
        original_height: 元画像の高さ[px]。
        crop_xyxy: crop した場合の crop 前の元画像座標系での切り出し範囲。None の場合は crop なし。
        resized_width: リサイズ後の画像幅[px]。None の場合はリサイズなし。
        resized_height: リサイズ後の画像高さ[px]。None の場合はリサイズなし。

    Returns:
        元画像基準の box。shape は入力と同じ、``np.float64``。
    """
    cropped_width = crop_xyxy[2] - crop_xyxy[0] if crop_xyxy is not None else original_width
    cropped_height = crop_xyxy[3] - crop_xyxy[1] if crop_xyxy is not None else original_height
    if resized_width is None:
        resized_width = cropped_width
    if resized_height is None:
        resized_height = cropped_height
    # モデル入力画像のピクセル座標を元画像のピクセル座標に変換
    if cropped_width != resized_width or cropped_height != resized_height:
        xyxy = xyxy / np.array([resized_width / cropped_width, resized_height / cropped_height, resized_width / cropped_width, resized_height / cropped_height], dtype=np.float64)
    # crop された場合は元画像座標系に戻す
    if crop_xyxy is not None:
        xyxy += np.array([crop_xyxy[0], crop_xyxy[1], crop_xyxy[0], crop_xyxy[1]], dtype=np.float64)
    return xyxy


def crop_intrinsic(intrinsic: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """crop 後の画像に対応する内部パラメータ ``K`` を返す（主点を平行移動）。

    画素座標の原点は左上（``x`` 右・``y`` 下）。``(x0, y0)`` を新しい原点にする
    切り出しでは、主点 ``(cx, cy)`` が同じだけ左上へ移動する（焦点距離は不変）。
    元画像上の点 ``(u, v)`` は crop 後に ``(u - x0, v - y0)`` に写る。

    Args:
        intrinsic: shape ``(3, 3)`` のピンホール内部パラメータ。
        x0: crop 左端の x 座標[px]。
        y0: crop 上端の y 座標[px]。

    Returns:
        shape ``(3, 3)``、``np.float64`` の新しい ``K``。**入力は破壊しない。**
    """
    result = np.array(intrinsic, dtype=np.float64, copy=True)
    result[0, 2] -= x0
    result[1, 2] -= y0
    return result


def scale_intrinsic(intrinsic: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """リサイズ後の画像に対応する内部パラメータ ``K`` を返す（焦点距離と主点をスケール）。

    画像を ``(scale_x, scale_y)`` 倍にリサンプルすると、焦点距離 ``(fx, fy)`` と
    主点 ``(cx, cy)`` が同じ倍率でスケールする。実際の画素のリサンプリングは行わない
    （それは画像処理ライブラリの責務であり core は持たない）。``K`` の更新のみを担う。

    Args:
        intrinsic: shape ``(3, 3)`` のピンホール内部パラメータ。
        scale_x: x 方向の拡大率（現サイズ / 元サイズ）。
        scale_y: y 方向の拡大率。

    Returns:
        shape ``(3, 3)``、``np.float64`` の新しい ``K``。**入力は破壊しない。**
    """
    result = np.array(intrinsic, dtype=np.float64, copy=True)
    result[0, 0] *= scale_x
    result[0, 2] *= scale_x
    result[1, 1] *= scale_y
    result[1, 2] *= scale_y
    return result


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """bool マスクを最近傍補間で ``(height, width)`` にリサイズする。

    Args:
        mask: リサイズ前の入力boolマスク。
        height: 出力マスクの高さ[px]（1 以上）。
        width: 出力マスクの幅[px]（1 以上）。

    Returns:
        shape ``(height, width)``、``np.bool_`` の C 連続配列
    """
    if mask.ndim != 2:
        raise ValueError(f"resize_mask_nearest expects a 2-D mask, got shape {mask.shape}")
    if height < 1 or width < 1:
        raise ValueError(f"resize_mask_nearest size must be >= 1, got {width}x{height}")
    h, w = mask.shape
    rows = ((np.arange(height) + 0.5) * h / height).astype(np.int64).clip(0, h - 1)
    cols = ((np.arange(width) + 0.5) * w / width).astype(np.int64).clip(0, w - 1)
    resized = mask[rows][:, cols]
    return np.ascontiguousarray(resized, dtype=np.bool_)

def paste_cropped_mask(cropped_mask: np.ndarray, original_height: int, original_width: int, crop_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Cropされた bool マスクを元画像座標系に貼り戻す。

    Args:
        cropped_mask: crop後座標の bool マスク。
        original_height: crop前の元画像の高さ[px]。
        original_width: crop前の元画像の幅[px]。
        crop_xyxy: crop範囲の元画像座標系での矩形 ``(x0, y0, x1, y1)``。x1/y1 は排他。

    Returns:
        shape ``(original_height, original_width)``、``np.bool_`` の **C 連続**配列。**入力は破壊しない。**
    """
    x0, y0, x1, y1 = crop_xyxy
    canvas = np.zeros((original_height, original_width), dtype=np.bool_)
    # キャンバスに収まる範囲へクリップ（負座標・サイズ超過を切り落とす）。
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, original_width), min(y1, original_height)
    if cx1 <= cx0 or cy1 <= cy0:
        return canvas  # 完全に画像外。
    canvas[cy0:cy1, cx0:cx1] = cropped_mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    return canvas
