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
]


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
