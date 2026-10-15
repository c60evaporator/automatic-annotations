"""距離フィルタ: 物体の想定サイズから、視線方向に奥へ伸びたノイズを落とす.

二輪車・歩行者・コーンのような小さな物体では、マスクの縁の画素が
背景の深度を拾い、**視線方向に尾を引いたノイズ**（尾状ノイズ）になる。
ROR / DBSCAN は尾が密につながっていると落とせず、L 字フィルタは
対象外（車両用）なので、物体の大きさの事前知識で奥側を切る。

手順（深度・LiDAR に別々に、L 字フィルタの直後に適用する）:
  1. センサーからの BEV 距離の分位点（Quantile）を基準距離 ref とする
  2. 視線と垂直方向の見かけの幅 W を求める
     （方位角の広がり × ref。点を視線に垂直な軸へ射影すると、奥の
     尾ほど横に広がって W が水増しされるため、角度で測る）
  3. W > (length + width) / 2 なら「側面が見えている」とみなし、
     奥行き D = width、そうでなければ「前後面が見えている」として D = length
  4. ref − margin 〜 ref + D + margin の点だけを残す
     （下限にも margin を取る。ref ちょうどで切ると、分位点より手前の
     真の点が常に quantile % ぶん消えるため。手前に飛んだ外れ値だけ落ちる）

サイズは webapp の config（DEFAULT_BOX_SIZES: {label: (length, width, height)}）
から run の box_fitting_params["box_sizes"] として渡ってくる。

NOTE: 推論（boxfitting_pipeline）と再フィルタ（refilter）で**同じ関数・
同じ順序**で呼ぶこと。設定の組み立ても distance_settings に一本化している。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# これ未満の点数なら適用しない（分位点も幅も意味を持たない）
MIN_POINTS = 5
# 見かけの幅を測るときに両端から捨てる方位角の割合 [%]。
# 横に 1〜2 点だけ飛んだ点で幅が決まらないようにする
WIDTH_ANGLE_QUANTILE = 2.0


def _ranges_and_angles(
    points_xy: np.ndarray, origin_xy: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """センサーからの BEV 距離と、点群の中央方向を 0 とした方位角 [rad]."""
    rel = np.asarray(points_xy, dtype=float)[:, :2] - np.asarray(origin_xy, dtype=float)
    ranges = np.hypot(rel[:, 0], rel[:, 1])
    # ±π の折り返しを避けるため、中央方向に回してから角度を取る
    center = np.median(rel, axis=0)
    heading = float(np.arctan2(center[1], center[0]))
    c, s = np.cos(-heading), np.sin(-heading)
    x = rel[:, 0] * c - rel[:, 1] * s
    y = rel[:, 0] * s + rel[:, 1] * c
    return ranges, np.arctan2(y, x)


def perceived_width(
    points_xy: np.ndarray,
    origin_xy: tuple[float, float],
    ref_distance: float,
    *,
    angle_quantile: float = WIDTH_ANGLE_QUANTILE,
) -> float:
    """視線と垂直方向の見かけの幅 [m]（方位角の広がり × 基準距離）."""
    _, angles = _ranges_and_angles(points_xy, origin_xy)
    if angles.size < 2:
        return 0.0
    low, high = np.percentile(angles, [angle_quantile, 100.0 - angle_quantile])
    return float(2.0 * ref_distance * np.tan(max(high - low, 0.0) / 2.0))


def side_visible(width: float, size_lw: tuple[float, float]) -> bool:
    """側面が見えているか（見かけの幅が (length + width) / 2 を超えるか）."""
    length, box_width = float(size_lw[0]), float(size_lw[1])
    return bool(width > (length + box_width) / 2.0)


def distance_filter(
    points: np.ndarray,
    origin_xy: tuple[float, float],
    *,
    size_lw: tuple[float, float],
    quantile: float,
    margin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """距離フィルタ本体.

    Args:
        points: ego 座標の点群 (N, 3)（BEV の xy だけを使う）
        origin_xy: センサー位置（点群と同じ座標系の XY）
        size_lw: 想定サイズ (length, width) [m]
        quantile: 基準距離に使う距離の分位点 [%]
        margin: 余裕 [m]。手前（ref − margin）と奥（ref + 奥行き + margin）の両側に取る

    Returns:
        (残す点のマスク, 診断情報)。点が少なければ全点を残す
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    keep = np.ones(points.shape[0], dtype=bool)
    if points.shape[0] < MIN_POINTS:
        return keep, {"applied": False, "num_removed": 0,
                      "reason": "too_few_points"}

    ranges, _ = _ranges_and_angles(points[:, :2], origin_xy)
    ref = float(np.percentile(ranges, quantile))
    width = perceived_width(points[:, :2], origin_xy, ref)
    side = side_visible(width, size_lw)
    depth_extent = float(size_lw[1] if side else size_lw[0])
    lower = ref - float(margin)
    upper = ref + depth_extent + float(margin)

    keep = (ranges >= lower) & (ranges <= upper)
    return keep, {
        "applied": True,
        "num_removed": int((~keep).sum()),
        "num_points": int(points.shape[0]),
        "ref_distance": round(ref, 3),
        "perceived_width": round(width, 3),
        "side_visible": side,
        "depth_extent": round(depth_extent, 3),
        "lower_distance": round(lower, 3),
        "upper_distance": round(upper, 3),
    }


def box_size_lw(
    box_sizes: dict[str, Any] | None, label: str | None
) -> tuple[float, float] | None:
    """config の DEFAULT_BOX_SIZES から (length, width) を取り出す."""
    size = (box_sizes or {}).get(label or "")
    if not size or len(size) < 2:
        return None
    return float(size[0]), float(size[1])


def distance_settings(
    general: dict[str, Any] | None,
    box_sizes: dict[str, Any] | None,
    label: str | None,
) -> dict[str, Any] | None:
    """run のパラメータから distance_filter() の引数を組み立てる.

    推論と再フィルタで**同じ解釈**をするために 1 箇所にまとめている。

    Args:
        general: box_fitting_params["distance_filter"]（対象ラベル・分位点・余裕）
        box_sizes: box_fitting_params["box_sizes"]

    Returns:
        このラベルに適用しないなら None
    """
    general = general or {}
    if label not in set(general.get("labels") or []):
        return None
    size = box_size_lw(box_sizes, label)
    if size is None:
        # サイズが分からないラベルには適用できない
        return None
    return {
        "size_lw": size,
        "quantile": float(general.get("quantile", 5.0)),
        "margin": float(general.get("margin", 0.3)),
    }
