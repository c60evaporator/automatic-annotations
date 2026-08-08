"""座標変換・回転表現のヘルパ。

- ego 座標系は右手系で ``x`` = 前方、``y`` = 左方、``z`` = 上方。
- 変換行列はすべて 4x4 同次変換行列（``np.float64``）。
- 変換行列の変数名・引数名は必ず ``<from>_to_<to>`` の向きを明示する。
  「extrinsic」のような向きの曖昧な名前を単体で使わない。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_quaternion",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "quaternion_to_yaw",
    "yaw_to_quaternion",
    "make_transform",
    "invert_transform",
    "transform_points",
    "crop_intrinsic",
    "scale_intrinsic",
    "rect_to_original",
    "resize_mask_nearest",
    "paste_cropped_mask",
]

###### 3D 座標変換・回転表現のヘルパ ######
def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """クォータニオンを正規化する。

    Args:
        quaternion: shape ``(4,)``、``(w, x, y, z)`` の順。

    Returns:
        shape ``(4,)``、``np.float64`` の単位クォータニオン。

    Raises:
        ValueError: shape が不正、またはゼロクォータニオンの場合。
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("quaternion must not be the zero quaternion")
    return quaternion / norm


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` クォータニオンの共役を返す。"""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """2つの ``(w, x, y, z)`` クォータニオンの Hamilton 積を返す。"""
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """クォータニオンを 3x3 回転行列に変換する。

    Args:
        quaternion: shape ``(4,)``、``(w, x, y, z)`` の順。正規化されている前提。

    Returns:
        shape ``(3, 3)`` の回転行列（``np.float64``）。
    """
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_to_yaw(quaternion: np.ndarray) -> float:
    """クォータニオンから z 軸まわりの yaw 角[rad]を取り出す。

    x 軸ベクトルを回転させて水平面に射影する方式で算出する
    （nuScenes devkit の ``quaternion_yaw`` と同じ定義）。
    ロール・ピッチを含む姿勢に対しても「車両が向いている方位」として
    直感に合う値を返すため、オイラー角分解より本用途に適する。

    Args:
        quaternion: shape ``(4,)``、``(w, x, y, z)`` の順。

    Returns:
        yaw 角[rad]。範囲は ``(-pi, pi]``。ego 座標系では ``0`` が前方（+x）、
        正の値が左回り（+y 方向へ向く）。
    """
    rotation = quaternion_to_rotation_matrix(quaternion)
    forward = rotation @ np.array([1.0, 0.0, 0.0])
    return float(np.arctan2(forward[1], forward[0]))


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    """z 軸まわりの yaw 角[rad]をクォータニオンに変換する。

    yaw のみを扱うモデル（mmdetection3d 系など）の出力を
    標準スキーマの ``Box3D.rotation`` に載せる際に使う。

    Args:
        yaw: yaw 角[rad]。

    Returns:
        shape ``(4,)``、``(w, x, y, z)`` の順のクォータニオン。
    """
    half = float(yaw) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def make_transform(
    quaternion: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """クォータニオンと並進から4x4同次変換行列を作る。

    Args:
        quaternion: from座標系からto座標系への回転。shape ``(4,)``、
            ``(w, x, y, z)`` の順。関数内で正規化する。
        translation: from座標系の原点をto座標系で表した位置。shape ``(3,)``。

    Returns:
        shape ``(4, 4)``、``np.float64`` のfrom-to変換行列。
    """
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {translation.shape}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(
        normalize_quaternion(quaternion)
    )
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """4x4 同次変換行列の逆変換を返す。

    回転部が直交行列であることを利用するため、一般の逆行列計算より安定かつ高速。

    Args:
        transform: shape ``(4, 4)`` の同次変換行列。

    Returns:
        shape ``(4, 4)`` の逆変換行列。``a_to_b`` を渡すと ``b_to_a`` が返る。
    """
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ translation
    return inverted


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """点群に 4x4 同次変換を適用する。

    先頭 3 列（x, y, z）のみを変換し、4 列目以降（intensity, ring など）は
    そのまま保持する。入力の dtype を維持する。

    Args:
        points: shape ``(N, C)``、``C >= 3``。先頭 3 列が x, y, z。
        transform: shape ``(4, 4)`` の同次変換行列。

    Returns:
        変換後の点群。shape・dtype は入力と同じ。
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, C>=3), got {points.shape}")

    transform = np.asarray(transform, dtype=np.float64)
    xyz = points[:, :3].astype(np.float64)
    transformed = xyz @ transform[:3, :3].T + transform[:3, 3]

    result = points.copy()
    result[:, :3] = transformed.astype(points.dtype)
    return result


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


###### 2D画像の座標変換・マスク後処理のヘルパ ######
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
