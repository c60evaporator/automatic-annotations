"""3D bounding box の頂点生成と 2D 投影.

参考実装は OpenCV の ``convexHull`` / ``intersectConvexConvex`` を使っていたが、
ここでは numpy だけで実装している。webapp 側に opencv を追加すると
依存が numpy のバージョンを巻き上げる事故につながるため
（推論サーバー側で実際に踏んだ）、Streamlit 側は numpy のみで完結させる。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from app.services.geometry.pointcloud import (
    project_camera_points,
    transform_ego_to_camera,
    transform_global_to_ego,
)
from app.services.geometry.transform import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)

BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # 一方の面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 反対側の面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 両面をつなぐ辺
)


def make_box_corners(
    center: Sequence[float],
    size_wlh: Sequence[float],
    rotation: Sequence[float],
) -> np.ndarray:
    """3D bounding box の 8 頂点を返す.

    Args:
        center: ボックス中心。座標系は rotation と揃っていること
        size_wlh: **nuScenes の並び ``[width, length, height]``**。
            SampleAnnotation.size がこの順で入っている
        rotation: box local → center の座標系への回転 ``(w, x, y, z)``

    Returns:
        shape ``(8, 3)`` の頂点。

    NOTE: ボックスローカル座標は x=前(length), y=左(width), z=上(height)。
    nuScenes の size は ``[width, length, height]`` で並びが違うので、
    ここで明示的に入れ替えている。取り違えると縦横が逆のボックスになる。
    """
    width, length, height = np.asarray(size_wlh, dtype=np.float64)

    corners_local = np.array(
        [
            [length / 2,  width / 2,  height / 2],
            [length / 2, -width / 2,  height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2,  width / 2, -height / 2],
            [-length / 2,  width / 2,  height / 2],
            [-length / 2, -width / 2,  height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2,  width / 2, -height / 2],
        ],
        dtype=np.float64,
    )

    rotation_matrix = quaternion_to_rotation_matrix(normalize_quaternion(rotation))
    return corners_local @ rotation_matrix.T + np.asarray(center, dtype=np.float64)


def rotate_quaternion_global_to_ego(
    box_rotation: Sequence[float], ego_quaternion: Sequence[float]
) -> np.ndarray:
    """global 座標系のボックス回転を ego 座標系の回転へ変換する."""
    ego_q = normalize_quaternion(ego_quaternion)
    return normalize_quaternion(
        quaternion_multiply(quaternion_conjugate(ego_q), box_rotation)
    )


# ── near plane クリップ ───────────────────────────────────────────────────────

def _clip_segment_to_near_plane(
    point0: np.ndarray, point1: np.ndarray, near_plane: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """3D 線分をカメラの near plane に対してクリップする.

    両端ともカメラ背後なら None。片方だけ前方なら交点で切る。
    これをしないと、カメラ背後の頂点が投影で符号反転し、
    画面の反対側に巨大なボックスが現れる。
    """
    point0 = np.asarray(point0, dtype=np.float64)
    point1 = np.asarray(point1, dtype=np.float64)

    z0, z1 = point0[2], point1[2]
    point0_is_front = z0 >= near_plane
    point1_is_front = z1 >= near_plane

    if not point0_is_front and not point1_is_front:
        return None
    if point0_is_front and point1_is_front:
        return point0.copy(), point1.copy()

    # p(t) = p0 + t * (p1 - p0) が p(t).z = near_plane となる t
    t = (near_plane - z0) / (z1 - z0)
    intersection = point0 + t * (point1 - point0)

    if point0_is_front:
        return point0.copy(), intersection
    return intersection, point1.copy()


def _collect_near_clipped_box_points(
    corners_camera: np.ndarray,
    box_edges: Sequence[tuple[int, int]] = BOX_EDGES,
    near_plane: float = 0.1,
) -> np.ndarray:
    """near plane でクリップした 3D ボックスの頂点を集める.

    Returns:
        shape ``(N, 3)``。ボックス全体が背後なら shape ``(0, 3)``。
    """
    corners_camera = np.asarray(corners_camera, dtype=np.float64)
    if corners_camera.shape != (8, 3):
        raise ValueError(
            f"corners_camera must have shape (8, 3), got {corners_camera.shape}"
        )

    clipped_points: list[np.ndarray] = []
    for index0, index1 in box_edges:
        segment = _clip_segment_to_near_plane(
            corners_camera[index0], corners_camera[index1], near_plane
        )
        if segment is None:
            continue
        clipped_points.extend(segment)

    if not clipped_points:
        return np.empty((0, 3), dtype=np.float64)

    points = np.asarray(clipped_points, dtype=np.float64)
    # 同じ頂点が複数の辺から追加されるため、近似的に重複除去する
    rounded = np.round(points, decimals=10)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


# ── 2D ポリゴン処理（numpy のみ）──────────────────────────────────────────────

def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """2D 点集合の凸包を反時計回りで返す（monotone chain）.

    cv2.convexHull の代替。点が 3 未満なら入力をそのまま返す。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return points

    order = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[order]

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in sorted_points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull if hull.shape[0] >= 3 else sorted_points


def clip_polygon_to_rect(
    polygon: np.ndarray, width: float, height: float
) -> np.ndarray:
    """凸ポリゴンを画像矩形でクリップする（Sutherland-Hodgman）.

    cv2.intersectConvexConvex の代替。
    単に外接矩形をクランプするのでは不十分で、画面の角を斜めに横切る
    細長いボックスで領域を過大評価してしまう。
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.shape[0] == 0:
        return polygon

    # (内側判定, 交点計算) を 4 辺ぶん繰り返す
    edges = (
        (lambda p: p[0] >= 0.0,     0, 0.0),      # 左
        (lambda p: p[0] <= width,   0, width),    # 右
        (lambda p: p[1] >= 0.0,     1, 0.0),      # 上
        (lambda p: p[1] <= height,  1, height),   # 下
    )

    output = polygon
    for is_inside, axis, bound in edges:
        if output.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        current = output
        clipped: list[np.ndarray] = []
        previous = current[-1]
        previous_inside = is_inside(previous)
        for point in current:
            point_inside = is_inside(point)
            if point_inside != previous_inside:
                denom = point[axis] - previous[axis]
                if denom != 0:
                    t = (bound - previous[axis]) / denom
                    clipped.append(previous + t * (point - previous))
            if point_inside:
                clipped.append(point)
            previous, previous_inside = point, point_inside
        output = (
            np.asarray(clipped, dtype=np.float64)
            if clipped else np.empty((0, 2), dtype=np.float64)
        )
    return output


# ── 3D → 2D 変換 ─────────────────────────────────────────────────────────────

def convert_3d_box_to_2d_box(
    center_ego: Sequence[float],
    size_wlh: Sequence[float],
    rotation_ego: Sequence[float],
    camera_translation: Sequence[float],
    camera_quaternion: Sequence[float],
    camera_intrinsic: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
    *,
    near_plane: float = 0.1,
    min_size: float = 1.0,
) -> tuple[int, int, int, int] | None:
    """ego 座標系の 3D ボックスを画像上の 2D 外接矩形へ変換する.

    Returns:
        ``(xmin, ymin, xmax, ymax)``。
        カメラ背後・画角外・小さすぎる場合は None。
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    corners_ego = make_box_corners(center_ego, size_wlh, rotation_ego)
    corners_camera = transform_ego_to_camera(
        corners_ego, camera_translation, camera_quaternion
    )

    clipped = _collect_near_clipped_box_points(
        corners_camera, near_plane=near_plane
    )
    if clipped.shape[0] == 0:
        return None

    # 画面外にはみ出す頂点も外接矩形の計算に要るので filter_outside_image=False
    projected, _ = project_camera_points(
        clipped,
        camera_intrinsic,
        image_width,
        image_height,
        near_plane=near_plane,
        filter_outside_image=False,
    )
    if projected.shape[0] == 0 or not np.all(np.isfinite(projected)):
        return None

    hull = convex_hull_2d(projected)
    if hull.shape[0] < 3:
        return None

    intersection = clip_polygon_to_rect(
        hull, float(image_width), float(image_height)
    )
    if intersection.shape[0] < 3:
        return None

    x_min = float(intersection[:, 0].min())
    y_min = float(intersection[:, 1].min())
    x_max = float(intersection[:, 0].max())
    y_max = float(intersection[:, 1].max())
    if x_max - x_min < min_size or y_max - y_min < min_size:
        return None

    return (
        int(round(x_min)), int(round(y_min)),
        int(round(x_max)), int(round(y_max)),
    )


def convert_global_box_to_2d_box(
    center_global: Sequence[float],
    size_wlh: Sequence[float],
    rotation_global: Sequence[float],
    ego_translation: Sequence[float],
    ego_quaternion: Sequence[float],
    camera_translation: Sequence[float],
    camera_quaternion: Sequence[float],
    camera_intrinsic: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
    **kwargs: Any,
) -> tuple[int, int, int, int] | None:
    """global 座標系のボックス（SampleAnnotation そのもの）を 2D 化する.

    DB のアノテーションは global 座標なので、
    ego → camera の 2 段変換が要る。呼び出し側の定型処理をここに寄せる。
    """
    center_ego = transform_global_to_ego(
        np.asarray(center_global, dtype=np.float64).reshape(1, 3),
        ego_translation,
        ego_quaternion,
    )[0]
    rotation_ego = rotate_quaternion_global_to_ego(rotation_global, ego_quaternion)

    return convert_3d_box_to_2d_box(
        center_ego,
        size_wlh,
        rotation_ego,
        camera_translation,
        camera_quaternion,
        camera_intrinsic,
        image_width,
        image_height,
        **kwargs,
    )
