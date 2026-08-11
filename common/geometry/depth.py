import numpy as np

from ..geometry.transform import (
    make_transform,
    transform_points,
)


def transform_cam_to_ego(
    points_camera: np.ndarray,
    camera_translation: np.ndarray | list[float],
    camera_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """カメラ座標の点群をego座標へ変換する。（Pseudo LiDAR点群の変換を想定）

    Args:
        points_camera:
            shape=(N, 3) のカメラ座標の点群。
        camera_translation:
            カメラ原点のego座標 [x, y, z]。
        camera_quaternion:
            camera座標からego座標への回転 [w, x, y, z]。

    Returns:
        shape=(N, 3) のego座標。
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(
            "points_camera must have shape (N, 3), "
            f"got {points_camera.shape}"
        )

    # calibration が表す camera -> ego pose をそのまま適用する。
    camera_to_ego = make_transform(camera_quaternion, camera_translation)

    return transform_points(points_camera, camera_to_ego)


def depth_map_to_point_cloud(
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Generate a pseudo-LiDAR point cloud from a depth map and camera intrinsics.

    Args:
        depth_map (np.ndarray): HxW array of depth values.
        camera_intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        mask (np.ndarray | None): HxW binary mask array. Only points where mask is True will be included. If None, all points are included.

    Returns:
        np.ndarray: Nx3 array of 3D points in the camera coordinate system. Only points where mask is True are included.
    """
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    i = i.flatten()
    j = j.flatten()
    depth = depth_map.flatten()

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    z = depth

    points = np.vstack((x, y, z)).T
    if mask is not None:
        points = points[mask.flatten()]
    return points
