import numpy as np
import cv2

from ..schemas import Box3D, Box2D
from ..geometry.pointcloud import (
    transform_ego_to_camera,
    project_camera_points
)
from ..geometry.detection import (
    make_box_corners_ego,
    BOX_EDGES
)


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
    box_edges: list[tuple[int, int]],
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

    for index0, index1 in box_edges:
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
                             camera_quaternion: np.ndarray,
                             camera_intrinsic: np.ndarray,
                             image_width: int,
                             image_height: int,
                             box_edges: list[tuple[int, int]] = BOX_EDGES,
                             near_plane: float = 0.1,
                             min_size: float = 1.0) -> Box2D:
    """
    Convert 3D bounding box to 2D bounding box in the image plane.

    Args:
        box_3d_ego (Box3D): 3D bounding box in ego coordinates.
        camera_translation (np.ndarray): Camera translation in ego coordinates.
        camera_quaternion (np.ndarray): Camera rotation in ego coordinates.
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
        camera_quaternion,
    )

    # Clip the box to the near plane
    clipped_camera_points = _collect_near_clipped_box_points(
        corners_camera,
        box_edges=box_edges,
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
