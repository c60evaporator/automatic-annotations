import numpy as np
import cv2

from .schemas import Box3D, Box2D
from .geometry import (
    invert_transform,
    make_transform,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    transform_points,
)


# Category mapping from Nuscenes to UniAD (Based on MMdet3D v1.0.0rc6. Referenece https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc6/mmdet3d/datasets/nuscenes_dataset.py#L56)
CATEGORY_MAPPING_TO_UNIAD = {
    'vehicle.car': {'category_name': 'car', 'id': 0, 'category_group': 'vehicle'},
    'vehicle.truck': {'category_name': 'truck', 'id': 1, 'category_group': 'vehicle'},
    'vehicle.construction': {'category_name': 'construction_vehicle', 'id': 2, 'category_group': 'vehicle'},
    'vehicle.bus.bendy': {'category_name': 'bus', 'id': 3, 'category_group': 'vehicle'},
    'vehicle.bus.rigid': {'category_name': 'bus', 'id': 3, 'category_group': 'vehicle'},
    'vehicle.trailer': {'category_name': 'trailer', 'id': 4, 'category_group': 'vehicle'},
    'movable_object.barrier': {'category_name': 'barrier', 'id': 5, 'category_group': 'road_object'},
    'vehicle.motorcycle': {'category_name': 'motorcycle', 'id': 6, 'category_group': 'two_wheeled'},
    'vehicle.bicycle': {'category_name': 'bicycle', 'id': 7, 'category_group': 'two_wheeled'},
    'human.pedestrian.adult': {'category_name': 'pedestrian', 'id': 8, 'category_group': 'pedestrian'},
    'human.pedestrian.child': {'category_name': 'pedestrian', 'id': 8, 'category_group': 'pedestrian'},
    'human.pedestrian.construction_worker': {'category_name': 'pedestrian', 'id': 8, 'category_group': 'pedestrian'},
    'human.pedestrian.police_officer': {'category_name': 'pedestrian', 'id': 8, 'category_group': 'pedestrian'},
    'movable_object.trafficcone': {'category_name': 'traffic_cone', 'id': 9, 'category_group': 'road_object'},
}

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # 一方の面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 反対側の面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 両面をつなぐ辺
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


def transform_ego_to_camera(
    points_ego: np.ndarray,
    camera_translation: np.ndarray | list[float],
    camera_rotation: np.ndarray | list[float],
) -> np.ndarray:
    """
    ego座標の点群をカメラ座標へ変換する。

    Args:
        points_ego:
            shape=(N, 3)
        camera_translation:
            カメラ原点のego座標 [x, y, z]
        camera_rotation:
            camera座標からego座標への回転 [w, x, y, z]

    Returns:
        shape=(N, 3) のカメラ座標
    """
    points_ego = np.asarray(points_ego, dtype=np.float64)
    if points_ego.ndim != 2 or points_ego.shape[1] != 3:
        raise ValueError(f"points_ego must have shape (N, 3), got {points_ego.shape}")

    # calibration が表す camera -> ego pose から逆変換を作る。
    camera_to_ego = make_transform(camera_rotation, camera_translation)
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

def get_sample_data_bboxes(sample_data: dict[str, dict],
                           sample_annotations: dict[str, dict],
                           instances: dict[str, dict],
                           categories: dict[str, dict],
                           category_conversion: dict[str, str] = None,
                           delete_unspecified_categories: bool = True,
                           tracking_ids: dict[str, int] = None):
    """
    Get bounding boxes in the camera's field of view for each sample data entry.

    Args:
        sample_data (dict): Dictionary of sample data entries. The keys are sample data tokens and the values are dict from sample_data.json.
        sample_annotations (dict): Dictionary of sample annotations. The keys are sample annotation tokens and the values are dict from sample_annotation.json.
        instances (dict): Dictionary of instances. The keys are instance tokens and the values are dict from instance.json.
        categories (dict): Dictionary of categories. The keys are category tokens and the values are dict from category.json.
        category_conversion (dict): Dictionary for converting category names to desired labels. If None, the original category names will be used.
        delete_unspecified_categories (bool): If True, boxes with categories not in category_conversion will be deleted. If False, they will be kept with their original category names.
        tracking_ids (dict): Dictionary for converting instance tokens to tracking IDs. If None, tracking IDs will be set to None.
    """
    if category_conversion is None:
        category_conversion = {cat["name"]: cat["name"] for cat in categories.values()}

    # Getting bounding boxes in the sample
    annotations_in_sample = [sa for sa in sample_annotations.values() if sa["sample_token"] == sample_data["sample_token"]]
    boxes_3d = [Box3D.from_dimensions(
        center=np.array(sa["translation"]),
        length=sa["size"][1],
        width=sa["size"][0],
        height=sa["size"][2],
        rotation=np.array(sa["rotation"]),
        label=category_conversion.get(categories[instances[sa["instance_token"]]["category_token"]]["name"], None),
        track_id=tracking_ids[sa["instance_token"]] if tracking_ids is not None else None
    ) for sa in annotations_in_sample]
    print(f"Number of boxes in sample {sample_data['sample_token']}: {len(boxes_3d)}")
    # Delete boxes with unspecified categories if required
    if delete_unspecified_categories:
        boxes_3d = [box for box in boxes_3d if box.label is not None]
        print(f"Number of boxes after removing unspecified categories: {len(boxes_3d)}")

    return boxes_3d
    

def filter_boxes_in_camera_fov(boxes_3d_ego: list[Box3D],
                               camera_translation: np.ndarray,
                               camera_rotation: np.ndarray,
                               camera_intrinsic: np.ndarray,
                               image_width: int,
                               image_height: int) -> list[Box3D]:
    """
    Filter 3D bounding boxes to keep only those whose centers are in the camera's field of view.

    Args:
        boxes_3d_ego (list of Box3D): List of 3D bounding boxes in ego coordinates.
        camera_translation (np.ndarray): Camera translation in ego coordinates.
        camera_rotation (np.ndarray): Camera rotation in ego coordinates.
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
                                                camera_rotation)
    _, valid = project_camera_points(center_points_cam, camera_intrinsic,
                                     image_width, image_height)
    valid_boxes_3d_ego = [box for box, v in zip(boxes_3d_ego, valid) if v]

    return valid_boxes_3d_ego


def _clip_segment_to_near_plane(
    point0: np.ndarray,
    point1: np.ndarray,
    near_plane: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """3D線分をカメラのnear planeに対してクリップする。

    Args:
        point0: shape (3,) のカメラ座標。
        point1: shape (3,) のカメラ座標。
        near_plane: 有効とみなす最小depth。

    Returns:
        クリップ後の線分。線分全体がカメラ背後ならNone。
    """
    point0 = np.asarray(point0, dtype=np.float64)
    point1 = np.asarray(point1, dtype=np.float64)

    z0 = point0[2]
    z1 = point1[2]

    point0_is_front = z0 >= near_plane
    point1_is_front = z1 >= near_plane

    if not point0_is_front and not point1_is_front:
        return None

    if point0_is_front and point1_is_front:
        return point0.copy(), point1.copy()

    # p(t) = p0 + t * (p1 - p0)
    # p(t).z = near_plane
    t = (near_plane - z0) / (z1 - z0)
    intersection = point0 + t * (point1 - point0)

    if point0_is_front:
        return point0.copy(), intersection

    return intersection, point1.copy()


def _collect_near_clipped_box_points(
    corners_camera: np.ndarray,
    near_plane: float = 0.1,
) -> np.ndarray:
    """near planeでクリップした3Dボックスの頂点を集める。

    Args:
        corners_camera: shape (8, 3) のカメラ座標。
        near_plane: 最小depth。

    Returns:
        shape (N, 3)。ボックス全体が背後ならshape (0, 3)。
    """
    corners_camera = np.asarray(
        corners_camera,
        dtype=np.float64,
    )
    if corners_camera.shape != (8, 3):
        raise ValueError(
            "corners_camera must have shape (8, 3), "
            f"got {corners_camera.shape}"
        )

    clipped_points: list[np.ndarray] = []

    for index0, index1 in BOX_EDGES:
        clipped_segment = _clip_segment_to_near_plane(
            corners_camera[index0],
            corners_camera[index1],
            near_plane,
        )
        if clipped_segment is None:
            continue

        clipped_points.extend(clipped_segment)

    if not clipped_points:
        return np.empty((0, 3), dtype=np.float64)

    points = np.asarray(clipped_points, dtype=np.float64)

    # 同じ頂点が複数の辺から追加されるため、近似的に重複除去する。
    rounded = np.round(points, decimals=10)
    _, unique_indices = np.unique(
        rounded,
        axis=0,
        return_index=True,
    )

    return points[np.sort(unique_indices)]


def convert_3d_box_to_2d_box(box_3d_ego: Box3D, 
                             camera_translation: np.ndarray,
                             camera_rotation: np.ndarray,
                             camera_intrinsic: np.ndarray,
                             image_width: int,
                             image_height: int,
                             near_plane: float = 0.1,
                             min_size: float = 1.0) -> Box2D:
    """
    Convert 3D bounding box to 2D bounding box in the image plane.

    Args:
        box_3d_ego (Box3D): 3D bounding box in ego coordinates.
        camera_translation (np.ndarray): Camera translation in ego coordinates.
        camera_rotation (np.ndarray): Camera rotation in ego coordinates.
        camera_intrinsic (np.ndarray): Camera intrinsic matrix.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "image_width and image_height must be positive"
        )
    # Convert the 3D box to camera coordinates
    corners_ego = make_box_corners_ego(box_3d_ego)
    corners_camera = transform_ego_to_camera(
        corners_ego,
        camera_translation,
        camera_rotation,
    )

    # Clip the box to the near plane
    clipped_camera_points = _collect_near_clipped_box_points(
        corners_camera,
        near_plane=near_plane,
    )
    if clipped_camera_points.shape[0] == 0:
        return None

    # Project the clipped points to the image plane
    projected_points, _ = project_camera_points(
        clipped_camera_points,
        camera_intrinsic,
        image_width,
        image_height,
        near_plane=near_plane,
        filter_outside_image=False,
    )
    if not np.all(np.isfinite(projected_points)):
        return None

    # Create convex hull of the projected points
    projected_hull = cv2.convexHull(
        projected_points.astype(np.float32)
    ).reshape(-1, 2)
    if projected_hull.shape[0] < 3:
        return None
    # Compute the intersection of the projected hull with the image boundary
    image_polygon = np.array(
        [
            [0.0, 0.0],
            [float(image_width), 0.0],
            [float(image_width), float(image_height)],
            [0.0, float(image_height)],
        ],
        dtype=np.float32,
    )
    intersection_area, intersection_polygon = (
        cv2.intersectConvexConvex(
            projected_hull.astype(np.float32),
            image_polygon,
        )
    )
    if (
        intersection_polygon is None
        or intersection_area <= 0
    ):
        return None
    intersection_polygon = intersection_polygon.reshape(-1, 2)

    # Check if the intersection polygon is too small
    x_min = float(intersection_polygon[:, 0].min())
    y_min = float(intersection_polygon[:, 1].min())
    x_max = float(intersection_polygon[:, 0].max())
    y_max = float(intersection_polygon[:, 1].max())
    if x_max - x_min < min_size or y_max - y_min < min_size:
        return None

    # Create a Box2D object from the intersection polygon
    return Box2D(
        xyxy=np.array([x_min, y_min, x_max, y_max], dtype=np.float64),
        label=box_3d_ego.label,
        score=box_3d_ego.score,
        track_id=box_3d_ego.track_id,
        attributes=box_3d_ego.attributes.copy(),
    )
