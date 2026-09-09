"""Convex hull + Minimum Occlusion Area による 3D ボックス当てはめ.

提供された参照実装をほぼそのまま移植したもの。アルゴリズムの本体
（境界点の決定、可視エッジの選択、台形によるオクルージョン面積の積分）は
変更していない。変更点は次のとおり:

  - 戻り値を dataclass ではなく dict にした（API/DB へそのまま載せるため）
  - 点群が **ego 座標**で来るため、sensor_origin_xy を呼び出し側から渡す
    （センサー座標系なら (0, 0) だが、こちらは ego 原点とカメラ位置がずれる）

## 手法の考え方

BEV へ落とした凸包に対し、向き θ を刻んで外接矩形を作る。
そのとき「センサーから見て隠れている（＝観測できていない）領域」の面積を
求め、これが最小になる向きを採用する。

最小面積矩形と違い、**センサーから見える面に矩形を合わせる**ため、
片側からしか見えていない車両でも向きが安定しやすい。
そのぶんセンサー位置に依存するので、原点の指定を誤ると結果が崩れる。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Line2D:
    """``n . x = c`` で表す直線."""
    normal: np.ndarray  # shape=(2,), 単位法線
    c: float

    @property
    def tangent(self) -> np.ndarray:
        a, b = self.normal
        return np.array([b, -a], dtype=float)

    def distance(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        return np.abs(point @ self.normal - self.c)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def _wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _convex_hull_clockwise(xy: np.ndarray) -> np.ndarray:
    """凸包を時計回りで返す.

    以降の走査が時計回りを前提にしているため、向きを揃える。
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(xy)
    points = xy[hull.vertices]
    # scipy は通常の XY 座標系で反時計回りを返す
    if _signed_area(points) > 0:
        points = points[::-1].copy()
    return points


def _boundary_indices(hull: np.ndarray, origin: np.ndarray) -> tuple[int, int]:
    """センサーから見た角度方向の両端点を求める.

    atan2 の分岐（±pi）をまたぐ物体で破綻しないよう、
    物体の平均方位を基準に角度をほどいてから最小・最大を取る。
    """
    rel = hull - origin
    azimuth = np.arctan2(rel[:, 1], rel[:, 0])
    reference = np.arctan2(np.mean(rel[:, 1]), np.mean(rel[:, 0]))
    delta = _wrap_pi(azimuth - reference)
    return int(np.argmin(delta)), int(np.argmax(delta))


def _candidate_rectangle(hull: np.ndarray, theta: float):
    """向き theta の外接矩形（4 直線と 4 頂点）を作る."""
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])

    proj_u = hull @ u
    proj_v = hull @ v
    u_min, u_max = float(np.min(proj_u)), float(np.max(proj_u))
    v_min, v_max = float(np.min(proj_v)), float(np.max(proj_v))

    lines = (
        Line2D(u, u_min), Line2D(v, v_min),
        Line2D(u, u_max), Line2D(v, v_max),
    )
    vertices = np.stack([
        u * u_min + v * v_min,
        u * u_max + v * v_min,
        u * u_max + v * v_max,
        u * u_min + v * v_max,
    ])
    return lines, vertices, (u, v, u_min, u_max, v_min, v_max)


def _ray_segment_intersection(
    origin: np.ndarray, direction: np.ndarray,
    p0: np.ndarray, p1: np.ndarray, eps: float = 1e-10,
):
    """``origin + t * direction`` (t >= 0) と線分 p0--p1 の交点."""
    edge = p1 - p0
    denominator = _cross2(direction, edge)
    if abs(denominator) < eps:
        return None

    diff = p0 - origin
    t = _cross2(diff, edge) / denominator
    s = _cross2(diff, direction) / denominator
    if t >= -eps and -eps <= s <= 1.0 + eps:
        return float(t), origin + t * direction
    return None


def _point_in_triangle(p, a, b, c, eps: float = 1e-9) -> bool:
    s1 = _cross2(b - a, p - a)
    s2 = _cross2(c - b, p - b)
    s3 = _cross2(a - c, p - c)
    positive = s1 >= -eps and s2 >= -eps and s3 >= -eps
    negative = s1 <= eps and s2 <= eps and s3 <= eps
    return positive or negative


def _select_projection_edge(
    lines: tuple[Line2D, ...], vertices: np.ndarray,
    boundary_point: np.ndarray, left_boundary: np.ndarray,
    right_boundary: np.ndarray, origin: np.ndarray,
) -> int:
    """センサーから境界点へ伸ばした光線が最初に当たる矩形の辺を選ぶ."""
    direction = boundary_point - origin
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("凸包の点がセンサー原点と一致しています")
    direction = direction / norm

    hits = []
    # line i は 頂点 [i-1] -> [i] の線分に対応する
    for edge_idx in range(4):
        p0 = vertices[(edge_idx - 1) % 4]
        p1 = vertices[edge_idx]
        hit = _ray_segment_intersection(origin, direction, p0, p1)
        if hit is not None:
            hits.append((hit[0], edge_idx, hit[1]))

    if not hits:
        raise RuntimeError("境界方向の光線が候補矩形と交差しませんでした")

    min_t = min(x[0] for x in hits)
    tolerance = 1e-8 * max(1.0, abs(min_t))
    nearest = [x for x in hits if abs(x[0] - min_t) <= tolerance]
    if len(nearest) == 1:
        return nearest[0][1]

    # 交点が矩形の頂点に一致した場合は、有効領域へ入るほうの辺を選ぶ
    for _, edge_idx, intersection in nearest:
        p0 = vertices[(edge_idx - 1) % 4]
        p1 = vertices[edge_idx]
        other = p0 if np.linalg.norm(p0 - intersection) > np.linalg.norm(p1 - intersection) else p1
        probe = intersection + 1e-6 * (other - intersection)
        if _point_in_triangle(probe, origin, left_boundary, right_boundary):
            return edge_idx

    return nearest[0][1]  # 数値誤差時のフォールバック


def _is_visible_side(
    point: np.ndarray, left_boundary: np.ndarray,
    right_boundary: np.ndarray, origin: np.ndarray, eps: float = 1e-12,
) -> bool:
    """点が境界弦に対してセンサーと同じ側にあるか."""
    edge = right_boundary - left_boundary
    point_sign = _cross2(edge, point - left_boundary)
    origin_sign = _cross2(edge, origin - left_boundary)
    return point_sign * origin_sign >= -eps


def _iteration_directions(
    hull: np.ndarray, idx_l: int, idx_r: int, origin: np.ndarray
) -> tuple[int, int]:
    """左右の境界点から凸包をどちら回りに辿るかを決める."""
    n = len(hull)
    left_prev = (idx_l - 1) % n
    right_prev = (idx_r - 1) % n

    dir_l = -1 if (
        (idx_l + 1) % n == idx_r
        or _is_visible_side(hull[left_prev], hull[idx_l], hull[idx_r], origin)
    ) else 1
    dir_r = -1 if (
        (idx_r + 1) % n == idx_l
        or _is_visible_side(hull[right_prev], hull[idx_l], hull[idx_r], origin)
    ) else 1
    return dir_l, dir_r


def _steps_to(start: int, target: int, direction: int, n: int) -> int:
    idx = start
    for step in range(n + 1):
        if idx == target:
            return step
        idx = (idx + direction) % n
    raise RuntimeError("凸包上で目標点に到達できませんでした")


def _calculate_or_one_side(
    hull: np.ndarray, projection_line: Line2D,
    start_idx: int, target_idx: int, direction: int, max_steps: int,
):
    """片側のオクルージョン面積を台形の連なりとして積分する."""
    if max_steps <= 0 or start_idx == target_idx:
        return 0.0, 0

    area = 0.0
    idx = start_idx
    previous_projection = None
    current_idx = start_idx

    for _ in range(max_steps):
        current_idx = idx
        if idx == target_idx:
            break

        next_idx = (idx + direction) % len(hull)
        p_current, p_next = hull[idx], hull[next_idx]
        hull_segment = p_next - p_current

        # 可視エッジ上への射影
        trapezoid_h = float(hull_segment @ projection_line.tangent)

        # 射影の向きが反転したら打ち切る
        if (idx != start_idx and previous_projection is not None
                and trapezoid_h * previous_projection < 0):
            break

        upper = float(projection_line.distance(p_current))
        lower = float(projection_line.distance(p_next))
        area += (upper + lower) * abs(trapezoid_h) / 2.0

        previous_projection = trapezoid_h
        idx = next_idx

    gap = _steps_to(current_idx, target_idx, direction, len(hull))
    return area, gap


def _score_orientation(
    hull: np.ndarray, theta: float, idx_l: int, idx_r: int, origin: np.ndarray
) -> float:
    lines, vertices, _ = _candidate_rectangle(hull, theta)

    proj_l = _select_projection_edge(
        lines, vertices, hull[idx_l], hull[idx_l], hull[idx_r], origin
    )
    proj_r = _select_projection_edge(
        lines, vertices, hull[idx_r], hull[idx_l], hull[idx_r], origin
    )
    dir_l, dir_r = _iteration_directions(hull, idx_l, idx_r, origin)

    area_r, gap_r = _calculate_or_one_side(
        hull=hull, projection_line=lines[proj_r],
        start_idx=idx_r, target_idx=idx_l, direction=dir_r, max_steps=len(hull),
    )
    area_l, _ = _calculate_or_one_side(
        hull=hull, projection_line=lines[proj_l],
        start_idx=idx_l, target_idx=idx_r, direction=dir_l, max_steps=gap_r,
    )
    return area_l + area_r


def fit_convex_hull_moa(
    points_xyz: np.ndarray,
    angle_step_deg: float = 0.5,
    sensor_origin_xy: tuple[float, float] = (0.0, 0.0),
    z_percentiles: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """凸包 + 最小オクルージョン面積で 3D ボックスを当てはめる.

    Args:
        points_xyz: 1 インスタンスの点群 ``(N, 3)``
        angle_step_deg: 向きの探索刻み [度]
        sensor_origin_xy: 点群と同じ XY 座標系でのセンサー位置。
            **点群が ego 座標なら、カメラ／LiDAR の取り付け位置を渡すこと**
            （(0, 0) はセンサー座標系のときの値）
        z_percentiles: 高さを決めるパーセンタイル。None なら min/max

    Returns:
        DB / API にそのまま載る dict。

    Raises:
        ValueError / RuntimeError: 点が少ない、向きが決まらない等
    """
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3).")

    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("At least 3 valid points are required.")

    origin = np.asarray(sensor_origin_xy, dtype=float)
    hull = _convex_hull_clockwise(points[:, :2])
    if len(hull) < 3:
        raise ValueError("At least 3 non-collinear hull points are required.")

    idx_l, idx_r = _boundary_indices(hull, origin)

    theta_candidates = np.deg2rad(np.arange(0.0, 90.0, angle_step_deg))
    scores = np.full(len(theta_candidates), np.inf)
    for i, theta in enumerate(theta_candidates):
        try:
            scores[i] = _score_orientation(hull, theta, idx_l, idx_r, origin)
        except (ValueError, RuntimeError, FloatingPointError):
            # 個々の候補が破綻しても探索全体は続ける
            continue

    if not np.any(np.isfinite(scores)):
        raise RuntimeError("No valid orientation candidate was found.")

    best_idx = int(np.nanargmin(scores))
    theta_best = float(theta_candidates[best_idx])

    _, corners_xy, params = _candidate_rectangle(hull, theta_best)
    u, v, u_min, u_max, v_min, v_max = params

    extent_u, extent_v = u_max - u_min, v_max - v_min
    center_xy = ((u_min + u_max) / 2.0) * u + ((v_min + v_max) / 2.0) * v

    if z_percentiles is None:
        z_min, z_max = float(np.min(points[:, 2])), float(np.max(points[:, 2]))
    else:
        z_min, z_max = (float(z) for z in np.percentile(points[:, 2], z_percentiles))

    # 長辺を車体前後方向とみなす
    if extent_u >= extent_v:
        length, width, yaw = extent_u, extent_v, theta_best
    else:
        length, width, yaw = extent_v, extent_u, theta_best + np.pi / 2.0
    yaw = float(_wrap_pi(yaw))

    return {
        "center_ego": [
            round(float(center_xy[0]), 3),
            round(float(center_xy[1]), 3),
            round((z_min + z_max) / 2.0, 3),
        ],
        # nuScenes の並び [width, length, height]
        "size_wlh": [
            round(float(width), 3),
            round(float(length), 3),
            round(float(max(z_max - z_min, 1e-6)), 3),
        ],
        "yaw_ego": round(yaw, 4),
        "hull_xy": {"points": np.round(hull, 3).tolist()},
        "corners_xy": {"points": np.round(corners_xy, 3).tolist()},
        "occlusion_area": round(float(scores[best_idx]), 4),
        "theta_search": round(theta_best, 4),
    }
