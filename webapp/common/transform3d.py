"""3D 座標変換の最小セット（inference 側で使う）.

webapp には app/services/geometry/transform.py に同等のものがあるが、
inference からは import できないため共通化した。
webapp 側も将来こちらへ寄せられるが、既存ページが広く依存しているので
今は触らない（同じ式であることをテストで確認している）。
"""
from __future__ import annotations

import numpy as np


def quaternion_to_rotation_matrix(quaternion) -> np.ndarray:
    """``(w, x, y, z)`` クォータニオンを 3x3 回転行列にする."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("quaternion must not be the zero quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def make_transform(quaternion, translation) -> np.ndarray:
    """クォータニオンと並進から 4x4 同次変換行列を作る."""
    t = np.asarray(translation, dtype=np.float64)
    if t.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {t.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    transform[:3, 3] = t
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """4x4 同次変換行列の逆変換を返す.

    回転部が直交行列であることを利用するため、
    一般の逆行列計算より安定かつ高速。
    """
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverted


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """点群に 4x4 同次変換を適用する（先頭 3 列のみ）."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, C>=3), got {points.shape}")
    if points.shape[0] == 0:
        return points
    transform = np.asarray(transform, dtype=np.float64)
    return points[:, :3] @ transform[:3, :3].T + transform[:3, 3]


def camera_to_ego(points_camera: np.ndarray, translation, quaternion) -> np.ndarray:
    """カメラ座標の点群を ego 座標へ変換する.

    CalibratedSensor は camera → ego の pose なので、そのまま適用する
    （ego → camera の逆変換ではない）。
    """
    return transform_points(points_camera, make_transform(quaternion, translation))


def scale_intrinsic(intrinsic, scale_x: float, scale_y: float) -> np.ndarray:
    """リサイズ後の画像に対応する内部パラメータを返す（入力は破壊しない）.

    画像を (scale_x, scale_y) 倍にリサンプルすると、
    焦点距離と主点が同じ倍率でスケールする。
    """
    result = np.array(intrinsic, dtype=np.float64, copy=True)
    result[0, 0] *= scale_x
    result[0, 2] *= scale_x
    result[1, 1] *= scale_y
    result[1, 2] *= scale_y
    return result
