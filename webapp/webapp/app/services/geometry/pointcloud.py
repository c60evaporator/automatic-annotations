"""点群・点の座標系変換とカメラ投影."""
from __future__ import annotations

import numpy as np

from app.services.geometry.transform import (
    invert_transform,
    make_transform,
    transform_points,
)


def transform_global_to_ego(
    points_global: np.ndarray,
    ego_translation: np.ndarray | list[float],
    ego_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """global 座標の点群を ego 座標へ変換する.

    Args:
        points_global: shape ``(N, 3)``
        ego_translation: ego 原点の global 座標 ``[x, y, z]``（EgoPose.translation）
        ego_quaternion: ego → global の回転 ``[w, x, y, z]``（EgoPose.rotation）
    """
    points_global = np.asarray(points_global, dtype=np.float64)
    if points_global.ndim != 2 or points_global.shape[1] != 3:
        raise ValueError(
            f"points_global must have shape (N, 3), got {points_global.shape}"
        )

    ego_to_global = make_transform(ego_quaternion, ego_translation)
    return transform_points(points_global, invert_transform(ego_to_global))


def transform_ego_to_global(
    points_ego: np.ndarray,
    ego_translation: np.ndarray | list[float],
    ego_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """ego 座標の点群を global 座標へ変換する."""
    points_ego = np.asarray(points_ego, dtype=np.float64)
    if points_ego.ndim != 2 or points_ego.shape[1] != 3:
        raise ValueError(f"points_ego must have shape (N, 3), got {points_ego.shape}")

    return transform_points(
        points_ego, make_transform(ego_quaternion, ego_translation)
    )


def transform_ego_to_camera(
    points_ego: np.ndarray,
    camera_translation: np.ndarray | list[float],
    camera_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """ego 座標の点群をカメラ座標へ変換する.

    Args:
        camera_translation: カメラ原点の ego 座標（CalibratedSensor.translation）
        camera_quaternion: camera → ego の回転（CalibratedSensor.rotation）
    """
    points_ego = np.asarray(points_ego, dtype=np.float64)
    if points_ego.ndim != 2 or points_ego.shape[1] != 3:
        raise ValueError(f"points_ego must have shape (N, 3), got {points_ego.shape}")

    camera_to_ego = make_transform(camera_quaternion, camera_translation)
    return transform_points(points_ego, invert_transform(camera_to_ego))


def transform_lidar_to_ego(
    points_lidar: np.ndarray,
    lidar_translation: np.ndarray | list[float],
    lidar_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """LiDAR 座標の点群を ego 座標へ変換する（Depth/Box Fitting で使う）."""
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if points_lidar.ndim != 2 or points_lidar.shape[1] < 3:
        raise ValueError(
            f"points_lidar must have shape (N, C>=3), got {points_lidar.shape}"
        )
    return transform_points(
        points_lidar, make_transform(lidar_quaternion, lidar_translation)
    )


def project_camera_points(
    points_camera: np.ndarray,
    camera_intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    near_plane: float = 0.1,
    filter_outside_image: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """カメラ座標の点を画像座標へ投影する.

    Args:
        points_camera: shape ``(N, 3)``
        camera_intrinsic: shape ``(3, 3)`` のピンホール内部パラメータ
        near_plane: カメラ前方の点とみなす z 座標の閾値 [m]
        filter_outside_image:
            ``True`` なら画像範囲外の点を ``valid=False`` として除外する。
            ``False`` なら near plane より前にあれば画像範囲外でも返す。
            3D ボックスの 2D 化では、画面外にはみ出した頂点も
            外接矩形の計算に必要なので ``False`` を使う。

    Returns:
        (points_uv, valid)
        points_uv は shape ``(M, 2)``、valid は入力 N 点に対応する bool 配列。
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "image_width and image_height must be positive, "
            f"got ({image_width}, {image_height})"
        )

    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(
            f"points_camera must have shape (N, 3), got {points_camera.shape}"
        )

    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(
            f"camera_intrinsic must have shape (3, 3), got {intrinsic.shape}"
        )

    depths = points_camera[:, 2]
    valid = depths > near_plane
    visible_points = points_camera[valid]

    if visible_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), valid

    homogeneous_image_points = visible_points @ intrinsic.T
    visible_points_uv = (
        homogeneous_image_points[:, :2] / homogeneous_image_points[:, 2:3]
    )

    if filter_outside_image:
        u, v = visible_points_uv.T
        within_image = (
            (u >= 0.0) & (u < image_width) & (v >= 0.0) & (v < image_height)
        )
        # valid は入力 N 点に対応するため、前方点の位置だけ画像内判定で更新する
        valid[valid] = within_image
        visible_points_uv = visible_points_uv[within_image]

    return visible_points_uv, valid
