import numpy as np

from .schemas import Box3D
from .geometry import (
    invert_transform,
    make_transform,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    transform_points,
)


def convert_global_bbox_to_ego(
    global_bbox: Box3D,
    ego_translation: np.ndarray | list[float],
    ego_quaternion: np.ndarray | list[float],
) -> Box3D:
    """
    Convert a global bounding box to a local bounding box in the ego vehicle's frame.

    Args:
        global_bbox (Box3D): The global bounding box.
        ego_translation (np.ndarray): The ego vehicle's translation (x, y, z).
        ego_quaternion (np.ndarray): The ego vehicle's orientation as a quaternion (w, x, y, z).

    Returns:
        Box3D: The bounding box expressed in the ego vehicle's local frame.

    Notes:
        ``ego_quaternion`` and ``ego_translation`` describe the ego pose in the
        global frame (that is, the ego-to-global transform).  Box dimensions do
        not change under this rigid transform.  A velocity, when present, is
        rotated into the local frame without applying the translation.
    """
    # Validate and normalize ego pose
    ego_quaternion = np.asarray(ego_quaternion, dtype=np.float64)
    ego_translation = np.asarray(ego_translation, dtype=np.float64)
    if ego_quaternion.shape != (4,):
        raise ValueError(f"ego_quaternion must have shape (4,), got {ego_quaternion.shape}")
    if ego_translation.shape != (3,):
        raise ValueError(f"ego_translation must have shape (3,), got {ego_translation.shape}")

    ego_quaternion = normalize_quaternion(ego_quaternion)

    # Transform the global bounding box center to the ego frame
    ego_to_global = make_transform(ego_quaternion, ego_translation)
    global_to_ego = invert_transform(ego_to_global)
    center = transform_points(
        np.asarray(global_bbox.center, dtype=np.float64).reshape(1, 3),
        global_to_ego,
    )[0]

    # Validate and normalize the global bounding box rotation
    global_rotation = np.asarray(global_bbox.rotation, dtype=np.float64)
    if global_rotation.shape != (4,):
        raise ValueError(f"global_bbox.rotation must have shape (4,), got {global_rotation.shape}")
    global_rotation = normalize_quaternion(global_rotation)
    ego_inverse_quaternion = quaternion_conjugate(ego_quaternion)
    rotation = normalize_quaternion(
        quaternion_multiply(ego_inverse_quaternion, global_rotation)
    )

    # Transform the global bounding box velocity to the ego frame, if present
    velocity = None
    if global_bbox.velocity is not None:
        global_velocity = np.asarray(global_bbox.velocity, dtype=np.float64)
        if global_velocity.shape not in ((2,), (3,)):
            raise ValueError(
                "global_bbox.velocity must have shape (2,) or (3,), "
                f"got {global_velocity.shape}"
            )
        velocity_3d = np.zeros(3, dtype=np.float64)
        velocity_3d[: global_velocity.size] = global_velocity
        velocity = (global_to_ego[:3, :3] @ velocity_3d)[: global_velocity.size]

    return Box3D(
        center=center,
        size=np.array(global_bbox.size, dtype=np.float64, copy=True),
        rotation=rotation,
        label=global_bbox.label,
        score=global_bbox.score,
        velocity=velocity,
        track_id=global_bbox.track_id,
        attributes=global_bbox.attributes.copy(),
    )


def make_box_corners_ego(box: Box3D) -> np.ndarray:
    """
    ego座標系の3D bounding boxから8頂点を生成する。

    Args:
        box: ego座標系のbounding box。``size`` は ``(length, width, height)``、
            ``rotation`` は box local から ego への回転を表す ``(w, x, y, z)``。

    Returns:
        shape=(3, 8) array of corners. Each column is a corner.
    """
    length, width, height = np.asarray(box.size, dtype=np.float64)

    # ボックスローカル座標:
    # x: 前、y: 左、z: 上
    corners_local = np.array(
        [
            [
                length / 2,
                length / 2,
                length / 2,
                length / 2,
                -length / 2,
                -length / 2,
                -length / 2,
                -length / 2,
            ],
            [
                width / 2,
                -width / 2,
                -width / 2,
                width / 2,
                width / 2,
                -width / 2,
                -width / 2,
                width / 2,
            ],
            [
                height / 2,
                height / 2,
                -height / 2,
                -height / 2,
                height / 2,
                height / 2,
                -height / 2,
                -height / 2,
            ],
        ],
        dtype=np.float64,
    )

    rotation_box_to_ego = quaternion_to_rotation_matrix(
        normalize_quaternion(box.rotation)
    )
    center_ego = np.asarray(box.center, dtype=np.float64).reshape(3, 1)

    return rotation_box_to_ego @ corners_local + center_ego

def transform_ego_to_camera(
    points_ego: np.ndarray,
    camera_translation: np.ndarray | list[float],
    camera_rotation: np.ndarray | list[float],
) -> np.ndarray:
    """
    ego座標の点群をカメラ座標へ変換する。

    Args:
        points_ego:
            shape=(3, N)
        camera_translation:
            カメラ原点のego座標 [x, y, z]
        camera_rotation:
            camera座標からego座標への回転 [w, x, y, z]

    Returns:
        shape=(3, N) のカメラ座標
    """
    points_ego = np.asarray(points_ego, dtype=np.float64)
    if points_ego.ndim != 2 or points_ego.shape[0] != 3:
        raise ValueError(f"points_ego must have shape (3, N), got {points_ego.shape}")

    # calibration が表す camera -> ego pose から逆変換を作る。
    camera_to_ego = make_transform(camera_rotation, camera_translation)
    ego_to_camera = invert_transform(camera_to_ego)

    # transform_points の標準shape (N, C) に転置して適用し、元のshapeへ戻す。
    return transform_points(points_ego.T, ego_to_camera).T

def project_camera_points(
    points_camera: np.ndarray,
    camera_intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    near_plane: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    カメラ座標の点を画像座標へ投影する。

    Args:
        points_camera:
            shape=(3, N) のカメラ座標の点群
        camera_intrinsic:
            shape=(3, 3) のピンホールカメラ内部パラメータ行列
        image_width:
            画像の幅 [px]
        image_height:
            画像の高さ [px]
        near_plane:
            カメラ前方の点とみなす z 座標の閾値。デフォルトは 0.1[m]。

    Returns:
        points_uv:
            shape=(2, M) の画像座標
        valid:
            入力N点のうちカメラ前方かつ画像内にある点を示すbool配列
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.ndim != 2 or points_camera.shape[0] != 3:
        raise ValueError(
            f"points_camera must have shape (3, N), got {points_camera.shape}"
        )

    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(
            f"camera_intrinsic must have shape (3, 3), got {intrinsic.shape}"
        )

    depths = points_camera[2]
    valid = depths > near_plane

    visible_points = points_camera[:, valid]

    if visible_points.shape[1] == 0:
        return np.empty((2, 0), dtype=np.float64), valid

    homogeneous_image_points = intrinsic @ visible_points

    points_uv = (
        homogeneous_image_points[:2]
        / homogeneous_image_points[2:3]
    )

    return points_uv, valid
