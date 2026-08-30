import numpy as np

from ..schemas import Box3D
from ..geometry.transform import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    make_transform,
    invert_transform,
    transform_points,
)
from ..geometry.pointcloud import (
    transform_ego_to_camera,
    project_camera_points,
)


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # 一方の面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 反対側の面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 両面をつなぐ辺
)


###### 3D Detection helpers ######
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
        shape=(8, 3) array of corners. Each row is a corner.
    """
    length, width, height = np.asarray(box.size, dtype=np.float64)

    # ボックスローカル座標:
    # x: 前、y: 左、z: 上
    corners_local = np.array(
        [
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
        ],
        dtype=np.float64,
    )

    rotation_box_to_ego = quaternion_to_rotation_matrix(
        normalize_quaternion(box.rotation)
    )
    center_ego = np.asarray(box.center, dtype=np.float64)

    return corners_local @ rotation_box_to_ego.T + center_ego


def filter_boxes_in_camera_fov(boxes_3d_ego: list[Box3D],
                               camera_translation: np.ndarray,
                               camera_quaternion: np.ndarray,
                               camera_intrinsic: np.ndarray,
                               image_width: int,
                               image_height: int) -> list[Box3D]:
    """
    Filter 3D bounding boxes to keep only those whose centers are in the camera's field of view.

    Args:
        boxes_3d_ego (list of Box3D): List of 3D bounding boxes in ego coordinates.
        camera_translation (np.ndarray): Camera translation in ego coordinates.
        camera_quaternion (np.ndarray): Camera rotation in ego coordinates.
        camera_intrinsic (np.ndarray): Camera intrinsic matrix.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        list of Box3D: Filtered list of 3D bounding boxes whose centers are in the camera's field of view.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "image_width and image_height must be positive"
        )
    
    # Keep only box centers in the camera's field of view.
    center_points_ego = np.asarray(
        [box.center for box in boxes_3d_ego],
        dtype=np.float64,
    ).reshape(-1, 3)
    center_points_cam = transform_ego_to_camera(center_points_ego, 
                                                camera_translation,
                                                camera_quaternion)
    _, valid = project_camera_points(center_points_cam, camera_intrinsic,
                                     image_width, image_height)
    valid_boxes_3d_ego = [box for box, v in zip(boxes_3d_ego, valid) if v]

    return valid_boxes_3d_ego
