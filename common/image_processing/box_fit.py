from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class Line2D:
    """Line represented by n . x = c."""
    normal: np.ndarray  # shape=(2,), unit normal
    c: float

    @property
    def tangent(self) -> np.ndarray:
        a, b = self.normal
        return np.array([b, -a], dtype=float)

    def distance(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        return np.abs(point @ self.normal - self.c)


@dataclass
class ConvexHullBoxFitResult:
    center: np.ndarray        # [x, y, z]
    size: np.ndarray          # [length, width, height]
    yaw: float                # yaw of the long side [rad]
    occlusion_area: float
    theta_search: float       # raw angle in [0, pi/2)
    hull_xy: np.ndarray
    corners_xy: np.ndarray
    score_curve: np.ndarray   # [[theta, score], ...]


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]

    return 0.5 * np.sum(
        x * np.roll(y, -1)
        - y * np.roll(x, -1)
    )


def _wrap_pi(angle: np.ndarray | float):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _convex_hull_clockwise(xy: np.ndarray) -> np.ndarray:
    """
    Extract convex hull.

    scipy.spatial.ConvexHull is used instead of explicitly implementing
    Graham scan. The resulting polygon is forced to clockwise order
    because the traversal below assumes clockwise hull vertices.
    """
    hull = ConvexHull(xy)
    points = xy[hull.vertices]

    # scipy returns CCW in the normal mathematical XY coordinate system.
    if _signed_area(points) > 0:
        points = points[::-1].copy()

    return points


def _boundary_indices(
    hull: np.ndarray,
    origin: np.ndarray,
) -> tuple[int, int]:
    """
    Find the two angular boundary points seen from the LiDAR.

    The paper directly uses min/max atan2. Here angles are first unwrapped
    around the object's mean bearing so that objects around +/-pi do not
    suffer from atan2 branch-cut problems.
    """
    rel = hull - origin

    azimuth = np.arctan2(
        rel[:, 1],
        rel[:, 0],
    )

    reference = np.arctan2(
        np.mean(rel[:, 1]),
        np.mean(rel[:, 0]),
    )

    delta = _wrap_pi(azimuth - reference)

    idx_l = int(np.argmin(delta))
    idx_r = int(np.argmax(delta))

    return idx_l, idx_r


def _candidate_rectangle(
    hull: np.ndarray,
    theta: float,
):
    """
    Calculate the enclosing rectangle for candidate orientation theta.

    line:
        n . x = c

    line[0]: u . x = min(u . p)
    line[1]: v . x = min(v . p)
    line[2]: u . x = max(u . p)
    line[3]: v . x = max(v . p)
    """
    u = np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    v = np.array([
        -np.sin(theta),
        np.cos(theta),
    ])

    proj_u = hull @ u
    proj_v = hull @ v

    u_min = float(np.min(proj_u))
    u_max = float(np.max(proj_u))
    v_min = float(np.min(proj_v))
    v_max = float(np.max(proj_v))

    lines = (
        Line2D(u, u_min),
        Line2D(v, v_min),
        Line2D(u, u_max),
        Line2D(v, v_max),
    )

    # Rectangle vertices corresponding to the four line intersections.
    vertices = np.stack([
        u * u_min + v * v_min,
        u * u_max + v * v_min,
        u * u_max + v * v_max,
        u * u_min + v * v_max,
    ])

    return lines, vertices, (
        u,
        v,
        u_min,
        u_max,
        v_min,
        v_max,
    )


def _ray_segment_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    eps: float = 1e-10,
):
    """
    Intersection between

        origin + t * direction, t >= 0

    and the finite segment p0--p1.
    """
    edge = p1 - p0

    denominator = _cross2(direction, edge)

    if abs(denominator) < eps:
        return None

    diff = p0 - origin

    t = _cross2(diff, edge) / denominator
    s = _cross2(diff, direction) / denominator

    if t >= -eps and -eps <= s <= 1.0 + eps:
        point = origin + t * direction
        return float(t), point

    return None


def _point_in_triangle(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-9,
) -> bool:
    s1 = _cross2(b - a, p - a)
    s2 = _cross2(c - b, p - b)
    s3 = _cross2(a - c, p - c)

    positive = (
        s1 >= -eps
        and s2 >= -eps
        and s3 >= -eps
    )

    negative = (
        s1 <= eps
        and s2 <= eps
        and s3 <= eps
    )

    return positive or negative


def _select_projection_edge(
    lines: tuple[Line2D, ...],
    vertices: np.ndarray,
    boundary_point: np.ndarray,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    origin: np.ndarray,
) -> int:
    """
    Algorithm 2: visible/projection edge selection.

    The ray from the LiDAR through the boundary hull point is intersected
    with all four rectangle edges. The first contacted rectangle edge is
    selected.

    If the first intersection is exactly a rectangle corner, choose the
    adjacent edge entering the valid triangle.
    """
    direction = boundary_point - origin
    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        raise ValueError(
            "A hull point coincides with the LiDAR origin."
        )

    direction /= norm

    hits = []

    # line i corresponds to segment vertex[i-1] -> vertex[i].
    for edge_idx in range(4):
        p0 = vertices[(edge_idx - 1) % 4]
        p1 = vertices[edge_idx]

        hit = _ray_segment_intersection(
            origin,
            direction,
            p0,
            p1,
        )

        if hit is not None:
            t, point = hit
            hits.append((t, edge_idx, point))

    if not hits:
        raise RuntimeError(
            "Boundary ray did not intersect candidate rectangle."
        )

    # "First contact" = closest intersection to LiDAR.
    min_t = min(x[0] for x in hits)

    tolerance = 1e-8 * max(
        1.0,
        abs(min_t),
    )

    nearest = [
        x for x in hits
        if abs(x[0] - min_t) <= tolerance
    ]

    if len(nearest) == 1:
        return nearest[0][1]

    # Intersection is at a rectangle vertex.
    # Select the adjacent edge that goes into the valid region.
    for _, edge_idx, intersection in nearest:
        p0 = vertices[(edge_idx - 1) % 4]
        p1 = vertices[edge_idx]

        if (
            np.linalg.norm(p0 - intersection)
            > np.linalg.norm(p1 - intersection)
        ):
            other = p0
        else:
            other = p1

        probe = (
            intersection
            + 1e-6 * (other - intersection)
        )

        if _point_in_triangle(
            probe,
            origin,
            left_boundary,
            right_boundary,
        ):
            return edge_idx

    # Numerical fallback.
    return nearest[0][1]


def _is_visible_side(
    point: np.ndarray,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    origin: np.ndarray,
    eps: float = 1e-12,
) -> bool:
    """
    Test whether point is on the same side of the boundary chord
    as the LiDAR origin.
    """
    edge = right_boundary - left_boundary

    point_sign = _cross2(
        edge,
        point - left_boundary,
    )

    origin_sign = _cross2(
        edge,
        origin - left_boundary,
    )

    return point_sign * origin_sign >= -eps


def _iteration_directions(
    hull: np.ndarray,
    idx_l: int,
    idx_r: int,
    origin: np.ndarray,
) -> tuple[int, int]:
    """
    Determine in which direction the convex hull should be traversed
    from the left/right boundary points.
    """
    n = len(hull)

    left_prev = (idx_l - 1) % n
    right_prev = (idx_r - 1) % n

    if (
        (idx_l + 1) % n == idx_r
        or _is_visible_side(
            hull[left_prev],
            hull[idx_l],
            hull[idx_r],
            origin,
        )
    ):
        dir_l = -1
    else:
        dir_l = 1

    if (
        (idx_r + 1) % n == idx_l
        or _is_visible_side(
            hull[right_prev],
            hull[idx_l],
            hull[idx_r],
            origin,
        )
    ):
        dir_r = -1
    else:
        dir_r = 1

    return dir_l, dir_r


def _steps_to(
    start: int,
    target: int,
    direction: int,
    n: int,
) -> int:
    idx = start

    for step in range(n + 1):
        if idx == target:
            return step

        idx = (idx + direction) % n

    raise RuntimeError(
        "Could not reach target on cyclic convex hull."
    )


def _calculate_or_one_side(
    hull: np.ndarray,
    projection_line: Line2D,
    start_idx: int,
    target_idx: int,
    direction: int,
    max_steps: int,
):
    """
    Algorithm 3 / CalculateOR.

    Occlusion area is integrated as a sequence of trapezoids.
    """
    if (
        max_steps <= 0
        or start_idx == target_idx
    ):
        return 0.0, 0

    area = 0.0

    idx = start_idx
    previous_projection = None
    current_idx = start_idx

    for _ in range(max_steps):
        current_idx = idx

        if idx == target_idx:
            break

        next_idx = (
            idx + direction
        ) % len(hull)

        p_current = hull[idx]
        p_next = hull[next_idx]

        hull_segment = (
            p_next - p_current
        )

        # Equation (1):
        # projection onto the visible rectangle edge.
        trapezoid_h = float(
            hull_segment @ projection_line.tangent
        )

        # Stopping criterion:
        # projection direction changed.
        if (
            idx != start_idx
            and previous_projection is not None
            and trapezoid_h * previous_projection < 0
        ):
            break

        upper = float(
            projection_line.distance(
                p_current
            )
        )

        lower = float(
            projection_line.distance(
                p_next
            )
        )

        trapezoid_area = (
            (upper + lower)
            * abs(trapezoid_h)
            / 2.0
        )

        area += trapezoid_area

        previous_projection = trapezoid_h
        idx = next_idx

    gap = _steps_to(
        current_idx,
        target_idx,
        direction,
        len(hull),
    )

    return area, gap


def _score_orientation(
    hull: np.ndarray,
    theta: float,
    idx_l: int,
    idx_r: int,
    origin: np.ndarray,
):
    lines, vertices, _ = _candidate_rectangle(
        hull,
        theta,
    )

    proj_l = _select_projection_edge(
        lines,
        vertices,
        hull[idx_l],
        hull[idx_l],
        hull[idx_r],
        origin,
    )

    proj_r = _select_projection_edge(
        lines,
        vertices,
        hull[idx_r],
        hull[idx_l],
        hull[idx_r],
        origin,
    )

    dir_l, dir_r = _iteration_directions(
        hull,
        idx_l,
        idx_r,
        origin,
    )

    # Algorithm 3:
    # start from the right boundary.
    area_r, gap_r = _calculate_or_one_side(
        hull=hull,
        projection_line=lines[proj_r],
        start_idx=idx_r,
        target_idx=idx_l,
        direction=dir_r,
        max_steps=len(hull),
    )

    # Fill remaining part from the left boundary.
    area_l, _ = _calculate_or_one_side(
        hull=hull,
        projection_line=lines[proj_l],
        start_idx=idx_l,
        target_idx=idx_r,
        direction=dir_l,
        max_steps=gap_r,
    )

    return area_l + area_r


def fit_convex_hull_moa(
    points_xyz: np.ndarray,
    angle_step_deg: float = 0.5,
    sensor_origin_xy: tuple[float, float] = (0.0, 0.0),
    z_percentiles: tuple[float, float] | None = None,
) -> ConvexHullBoxFitResult:
    """
    Convex-hull + Minimum Occlusion Area vehicle box fitting.

    Parameters
    ----------
    points_xyz:
        (N, 3) points of ONE object instance.

    angle_step_deg:
        Orientation search resolution.
        Paper experiment: 0.5 deg.

    sensor_origin_xy:
        LiDAR position in the same XY coordinate system as points_xyz.

        If points are already in the LiDAR coordinate frame:
            (0, 0)

    z_percentiles:
        None:
            Use exact min/max z (paper behavior).

        e.g. (1, 99):
            Robust alternative useful for pseudo-LiDAR / DA3.

    Returns
    -------
    ConvexHullBoxFitResult
    """
    points = np.asarray(
        points_xyz,
        dtype=float,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            "points_xyz must have shape (N, 3)."
        )

    points = points[
        np.isfinite(points).all(axis=1)
    ]

    if len(points) < 3:
        raise ValueError(
            "At least 3 valid points are required."
        )

    origin = np.asarray(
        sensor_origin_xy,
        dtype=float,
    )

    xy = points[:, :2]

    # ---------------------------------------------------------
    # 1. 3D -> BEV
    # 2. Convex hull
    # ---------------------------------------------------------
    hull = _convex_hull_clockwise(xy)

    if len(hull) < 3:
        raise ValueError(
            "At least 3 non-collinear hull points are required."
        )

    # ---------------------------------------------------------
    # 3. Boundary points from LiDAR azimuth
    # ---------------------------------------------------------
    idx_l, idx_r = _boundary_indices(
        hull,
        origin,
    )

    # ---------------------------------------------------------
    # 4. Search theta in [0, pi/2)
    # ---------------------------------------------------------
    theta_candidates = np.deg2rad(
        np.arange(
            0.0,
            90.0,
            angle_step_deg,
        )
    )

    scores = np.full(
        len(theta_candidates),
        np.inf,
    )

    for i, theta in enumerate(
        theta_candidates
    ):
        try:
            scores[i] = _score_orientation(
                hull=hull,
                theta=theta,
                idx_l=idx_l,
                idx_r=idx_r,
                origin=origin,
            )

        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            continue

    if not np.any(np.isfinite(scores)):
        raise RuntimeError(
            "No valid orientation candidate was found."
        )

    best_idx = int(
        np.nanargmin(scores)
    )

    theta_best = float(
        theta_candidates[best_idx]
    )

    # ---------------------------------------------------------
    # 5. Reconstruct final rectangle
    # ---------------------------------------------------------
    _, corners_xy, params = _candidate_rectangle(
        hull,
        theta_best,
    )

    (
        u,
        v,
        u_min,
        u_max,
        v_min,
        v_max,
    ) = params

    extent_u = u_max - u_min
    extent_v = v_max - v_min

    center_u = (
        u_min + u_max
    ) / 2.0

    center_v = (
        v_min + v_max
    ) / 2.0

    center_xy = (
        center_u * u
        + center_v * v
    )

    # ---------------------------------------------------------
    # 6. Z extent
    # ---------------------------------------------------------
    if z_percentiles is None:
        z_min = float(
            np.min(points[:, 2])
        )
        z_max = float(
            np.max(points[:, 2])
        )
    else:
        z_min, z_max = np.percentile(
            points[:, 2],
            z_percentiles,
        )

        z_min = float(z_min)
        z_max = float(z_max)

    # ---------------------------------------------------------
    # 7. Define vehicle yaw along longer side
    # ---------------------------------------------------------
    if extent_u >= extent_v:
        length = extent_u
        width = extent_v
        yaw = theta_best
    else:
        length = extent_v
        width = extent_u
        yaw = theta_best + np.pi / 2.0

    yaw = float(
        _wrap_pi(yaw)
    )

    center = np.array([
        center_xy[0],
        center_xy[1],
        (z_min + z_max) / 2.0,
    ])

    size = np.array([
        length,
        width,
        max(z_max - z_min, 1e-6),
    ])

    return ConvexHullBoxFitResult(
        center=center,
        size=size,
        yaw=yaw,
        occlusion_area=float(
            scores[best_idx]
        ),
        theta_search=theta_best,
        hull_xy=hull,
        corners_xy=corners_xy,
        score_curve=np.column_stack([
            theta_candidates,
            scores,
        ]),
    )
