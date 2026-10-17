"""距離フィルタ: 物体の前後方向に大きく外れた点を落とす.

二輪車・歩行者・コーンのような小さな物体では、マスクの縁の画素が背景の
深度を拾ったり、すぐ後ろの壁などの点がマスクに混ざったりする。
ROR / DBSCAN（外れ値除去）は混ざった点が密につながっていると落とせず、
L 字フィルタは対象外（車両用）なので、ラベルごとに別の方法で落とす。

方式（webapp の config の DISTANCE_FILTER_METHODS: {label: method} で指定）:

  long_side … 想定サイズの長辺で前後の範囲を切る（motorcycle / pedestrian /
              traffic_cone を想定）
      1. センサーからの BEV 距離の分位点（Quantile）を基準距離 ref とする
      2. ref − margin 〜 ref + 長辺 + margin の点だけを残す
      向きは推定しない。どの向きでも見える奥行きは概ね長辺以下なので、
      長辺を上限にすれば安全側に倒れる（以前の「見かけの幅で側面／前後面を
      判定する」方式は、実データのサイズのばらつきで判定が崩れたため廃止）

  dbscan    … BEV で DBSCAN を行い、**点数上位 2 クラスタのうち手前の方**だけを
              残す（bicycle を想定）。壁の 1 m ほど手前を走る自転車のように、
              物体と背景が別の塊に分かれるケースで背景を落とす。
              「手前」はクラスタごとの距離の分位点（Quantile）で比べる。
              ノイズ判定の点（どのクラスタにも入らない点）も落とす

どちらも XY（BEV）だけを使い、Z は使わない。
深度と LiDAR に別々に、L 字フィルタの直後に適用する。

NOTE: 推論（boxfitting_pipeline）と再フィルタ（refilter）で**同じ関数・
同じ順序**で呼ぶこと。設定の組み立ても distance_settings に一本化している。
"""
from __future__ import annotations

from typing import Any

import numpy as np

DISTANCE_METHOD_LONG_SIDE = "long_side"
DISTANCE_METHOD_DBSCAN = "dbscan"
DISTANCE_METHODS = (DISTANCE_METHOD_LONG_SIDE, DISTANCE_METHOD_DBSCAN)

# これ未満の点数なら適用しない（分位点もクラスタも意味を持たない）
MIN_POINTS = 5


def _ranges(points_xy: np.ndarray, origin_xy: tuple[float, float]) -> np.ndarray:
    """センサーからの BEV 距離."""
    rel = np.asarray(points_xy, dtype=float)[:, :2] - np.asarray(origin_xy, dtype=float)
    return np.hypot(rel[:, 0], rel[:, 1])


def _bev_dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """BEV（z を 0 にした点）で DBSCAN を行い、クラスタ番号を返す（-1 はノイズ）.

    外れ値除去の DBSCAN（depth_ops.largest_dbscan_cluster）と同じ実装を使う。
    """
    import open3d as o3d

    flat = np.column_stack([points[:, 0], points[:, 1], np.zeros(points.shape[0])])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(flat)
    return np.asarray(
        cloud.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False)
    )


def _filter_long_side(
    ranges: np.ndarray, *, quantile: float, margin: float, long_side: float
) -> tuple[np.ndarray, dict[str, Any]]:
    ref = float(np.percentile(ranges, quantile))
    lower = ref - float(margin)
    upper = ref + float(long_side) + float(margin)
    keep = (ranges >= lower) & (ranges <= upper)
    return keep, {
        "ref_distance": round(ref, 3),
        "long_side": round(float(long_side), 3),
        "lower_distance": round(lower, 3),
        "upper_distance": round(upper, 3),
    }


def _filter_dbscan(
    points: np.ndarray,
    ranges: np.ndarray,
    *,
    quantile: float,
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = _bev_dbscan_labels(points, eps, min_samples)
    valid = labels >= 0
    if not valid.any():
        # すべてノイズ判定なら絞り込まない（点を失うより情報を残す）
        return np.ones(points.shape[0], dtype=bool), {
            "num_clusters": 0, "reason": "all_noise",
        }
    counts = np.bincount(labels[valid])
    # 点数の多い順に上位 2 クラスタ
    top = [int(c) for c in np.argsort(-counts, kind="stable")[:2]]
    # 手前の判定はクラスタごとの距離の分位点（基準距離と同じ考え方）
    near = {c: float(np.percentile(ranges[labels == c], quantile)) for c in top}
    kept = min(top, key=lambda c: near[c])
    keep = labels == kept
    info: dict[str, Any] = {
        "num_clusters": int(counts.shape[0]),
        "num_noise": int((~valid).sum()),
        "kept_cluster_size": int(counts[kept]),
        "kept_cluster_distance": round(near[kept], 3),
    }
    others = [c for c in top if c != kept]
    if others:
        info["other_cluster_size"] = int(counts[others[0]])
        info["other_cluster_distance"] = round(near[others[0]], 3)
    return keep, info


def distance_filter(
    points: np.ndarray,
    origin_xy: tuple[float, float],
    *,
    method: str,
    quantile: float,
    margin: float = 0.0,
    long_side: float | None = None,
    dbscan_eps: float = 0.0,
    dbscan_min_samples: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """距離フィルタ本体.

    Args:
        points: ego 座標の点群 (N, 3)（BEV の xy だけを使う）
        origin_xy: センサー位置（点群と同じ座標系の XY）
        method: ``long_side`` / ``dbscan``
        quantile: 距離の分位点 [%]（long_side の基準距離、dbscan の「手前」の比較）
        margin: long_side の余裕 [m]（手前と奥の両側に取る）
        long_side: long_side の長辺 [m]（想定サイズの length と width の大きい方）
        dbscan_eps / dbscan_min_samples: dbscan の DBSCAN パラメータ（BEV）

    Returns:
        (残す点のマスク, 診断情報)。点が少なければ全点を残す
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    keep = np.ones(points.shape[0], dtype=bool)
    if points.shape[0] < MIN_POINTS:
        return keep, {"applied": False, "method": method, "num_removed": 0,
                      "reason": "too_few_points"}

    ranges = _ranges(points[:, :2], origin_xy)
    if method == DISTANCE_METHOD_LONG_SIDE:
        if not long_side:
            return keep, {"applied": False, "method": method, "num_removed": 0,
                          "reason": "no_box_size"}
        keep, detail = _filter_long_side(
            ranges, quantile=quantile, margin=margin, long_side=long_side,
        )
    elif method == DISTANCE_METHOD_DBSCAN:
        if dbscan_eps <= 0 or dbscan_min_samples < 1:
            return keep, {"applied": False, "method": method, "num_removed": 0,
                          "reason": "dbscan_disabled"}
        keep, detail = _filter_dbscan(
            points, ranges, quantile=quantile,
            eps=dbscan_eps, min_samples=dbscan_min_samples,
        )
    else:
        raise ValueError(f"未対応の距離フィルタ方式です: {method}")

    return keep, {
        "applied": True,
        "method": method,
        "num_removed": int((~keep).sum()),
        "num_points": int(points.shape[0]),
        **detail,
    }


def box_long_side(box_sizes: dict[str, Any] | None, label: str | None) -> float | None:
    """config の DEFAULT_BOX_SIZES（length, width, height）から長辺を取り出す."""
    size = (box_sizes or {}).get(label or "")
    if not size or len(size) < 2:
        return None
    return float(max(size[0], size[1]))


def _method_for_label(general: dict[str, Any], label: str | None) -> str | None:
    """ラベルの方式。古い run（labels だけを持つ）は long_side とみなす."""
    methods = general.get("methods")
    if methods is not None:
        return methods.get(label or "")
    if label in set(general.get("labels") or []):
        # 見かけの幅で判定する旧方式は廃止したので、長辺で切る方式で代える
        return DISTANCE_METHOD_LONG_SIDE
    return None


def distance_settings(
    general: dict[str, Any] | None,
    box_sizes: dict[str, Any] | None,
    label: str | None,
) -> dict[str, Any] | None:
    """run のパラメータから distance_filter() の引数を組み立てる.

    推論と再フィルタで**同じ解釈**をするために 1 箇所にまとめている。

    Args:
        general: box_fitting_params["distance_filter"]
            （methods・quantile・margin・dbscan_eps・dbscan_min_samples）
        box_sizes: box_fitting_params["box_sizes"]

    Returns:
        このラベルに適用しないなら None
    """
    general = general or {}
    method = _method_for_label(general, label)
    if not method:
        return None
    settings: dict[str, Any] = {
        "method": method,
        "quantile": float(general.get("quantile", 5.0)),
    }
    if method == DISTANCE_METHOD_LONG_SIDE:
        long_side = box_long_side(box_sizes, label)
        if long_side is None:
            # サイズが分からないラベルには適用できない
            return None
        settings.update({
            "margin": float(general.get("margin", 0.3)),
            "long_side": long_side,
        })
    elif method == DISTANCE_METHOD_DBSCAN:
        settings.update({
            "dbscan_eps": float(general.get("dbscan_eps", 0.0)),
            "dbscan_min_samples": int(general.get("dbscan_min_samples", 0)),
        })
    else:
        return None
    return settings
