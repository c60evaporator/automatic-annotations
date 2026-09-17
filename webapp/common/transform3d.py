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


def ego_to_ego(
    points_ego: np.ndarray,
    source_ego_pose: dict,
    target_ego_pose: dict,
) -> np.ndarray:
    """ある時刻の ego 座標を、別の時刻の ego 座標へ移す.

    Args:
        source_ego_pose / target_ego_pose: ``{"translation", "rotation"}``
            （ego → global の姿勢）

    カメラごとに sample_data のタイムスタンプが違うため、
    「ego 座標」の基準そのものがカメラ間でずれている。
    カメラを跨いで点群を比べる前に、必ず共通の基準へ揃えること。
    自車 10 m/s・時刻差 25 ms で 0.25 m ずれ、対象物体の移動分も加わる。

    どちらかの姿勢が欠けている場合は変換せず、そのまま返す
    （変換できないのに黙って別座標系の点を混ぜるより安全）。
    """
    if not source_ego_pose or not target_ego_pose:
        return np.asarray(points_ego, dtype=np.float64)

    source_to_global = make_transform(
        source_ego_pose["rotation"], source_ego_pose["translation"]
    )
    target_to_global = make_transform(
        target_ego_pose["rotation"], target_ego_pose["translation"]
    )
    return transform_points(
        points_ego, invert_transform(target_to_global) @ source_to_global
    )


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
