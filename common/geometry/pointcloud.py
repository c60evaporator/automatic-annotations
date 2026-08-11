import numpy as np

from ..geometry.transform import (
    make_transform,
    invert_transform,
    transform_points,
)


def transform_lidar_to_ego(
    points_lidar: np.ndarray,
    lidar_translation: np.ndarray | list[float],
    lidar_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """
    LiDAR座標の点群をego座標へ変換する。

    Args:
        points_lidar:
            shape=(N, 3)
        lidar_translation:
            LiDAR原点のego座標 [x, y, z]
        lidar_quaternion:
            LiDAR座標からego座標への回転 [w, x, y, z]

    Returns:
        shape=(N, 3) のego座標
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError(f"points_lidar must have shape (N, 3), got {points_lidar.shape}")

    # calibration が表す lidar -> ego pose を使う。
    lidar_to_ego = make_transform(lidar_quaternion, lidar_translation)

    return transform_points(points_lidar, lidar_to_ego)


def transform_ego_to_global(
    points_ego: np.ndarray,
    ego_translation: np.ndarray | list[float],
    ego_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """
    ego座標の点群をglobal座標へ変換する。

    Args:
        points_ego:
            shape=(N, 3)
        ego_translation:
            ego原点のglobal座標 [x, y, z]
        ego_quaternion:
            ego座標からglobal座標への回転 [w, x, y, z]

    Returns:
        shape=(N, 3) のglobal座標
    """
    points_ego = np.asarray(points_ego, dtype=np.float64)
    if points_ego.ndim != 2 or points_ego.shape[1] != 3:
        raise ValueError(f"points_ego must have shape (N, 3), got {points_ego.shape}")

    # ego pose を使って、ego -> global transform を作る。
    ego_to_global = make_transform(ego_quaternion, ego_translation)

    return transform_points(points_ego, ego_to_global)


def transform_ego_to_camera(
    points_ego: np.ndarray,
    camera_translation: np.ndarray | list[float],
    camera_quaternion: np.ndarray | list[float],
) -> np.ndarray:
    """
    ego座標の点群をカメラ座標へ変換する。

    Args:
        points_ego:
            shape=(N, 3)
        camera_translation:
            カメラ原点のego座標 [x, y, z]
        camera_quaternion:
            camera座標からego座標への回転 [w, x, y, z]

    Returns:
        shape=(N, 3) のカメラ座標
    """
    points_ego = np.asarray(points_ego, dtype=np.float64)
    if points_ego.ndim != 2 or points_ego.shape[1] != 3:
        raise ValueError(f"points_ego must have shape (N, 3), got {points_ego.shape}")

    # calibration が表す camera -> ego pose から逆変換を作る。
    camera_to_ego = make_transform(camera_quaternion, camera_translation)
    ego_to_camera = invert_transform(camera_to_ego)

    return transform_points(points_ego, ego_to_camera)


def project_camera_points(
    points_camera: np.ndarray,
    camera_intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    near_plane: float = 0.1,
    filter_outside_image: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    カメラ座標の点を画像座標へ投影する。

    Args:
        points_camera:
            shape=(N, 3) のカメラ座標の点群
        camera_intrinsic:
            shape=(3, 3) のピンホールカメラ内部パラメータ行列
        image_width:
            画像の幅 [px]
        image_height:
            画像の高さ [px]
        near_plane:
            カメラ前方の点とみなす z 座標の閾値。デフォルトは 0.1[m]。
        filter_outside_image:
            ``True`` の場合は画像範囲外の点を ``valid=False`` として除外する。
            ``False`` の場合はnear planeより前にあれば画像範囲外でも返す。

    Returns:
        points_uv:
            shape=(M, 2) の画像座標
        valid:
            入力N点のうち投影結果に含まれる点を示すbool配列。
            ``filter_outside_image=True`` ではカメラ前方かつ画像内、``False``
            ではカメラ前方にある点を示す。
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "image_width and image_height must be positive"
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
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "image_width and image_height must be positive, "
            f"got ({image_width}, {image_height})"
        )

    depths = points_camera[:, 2]
    valid = depths > near_plane

    visible_points = points_camera[valid]

    if visible_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), valid

    homogeneous_image_points = visible_points @ intrinsic.T

    visible_points_uv = (
        homogeneous_image_points[:, :2]
        / homogeneous_image_points[:, 2:3]
    )

    if filter_outside_image:
        u, v = visible_points_uv.T
        within_image = (
            (u >= 0.0)
            & (u < image_width)
            & (v >= 0.0)
            & (v < image_height)
        )

        # validは入力N点に対応するため、前方点の位置だけ画像内判定で更新する。
        valid[valid] = within_image
        visible_points_uv = visible_points_uv[within_image]

    return visible_points_uv, valid
