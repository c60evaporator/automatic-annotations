"""グローバルトラック単位のボックス角度の補正（dbscan_interp）.

フレームごとの向きの推定は外れることがあるが、同じ物体の向きはフレーム間で
大きくは変わらない。グローバル座標の向きを単位円上でクラスタリングし、
最大クラスタを「正しい向き」とみなして、それ以外のフレームを補正する。

対象外のラベル: BOX_ORIENTATION_ESTIMATIONS が None のラベル（向きの推定が
無いと yaw の前後が決まらず、同じ向きの箱が別のクラスタに割れるため）。

時間の順番は見ない（曲がる物体は eps の連鎖で 1 つのクラスタに入る前提）。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.services.box_orientation import (
    ORIENTATION_METHOD_PART_HOE,
    ORIENTATION_STATUS_ESTIMATED,
    apply_orientation,
    wrap_angle,
)

ORIENTATION_CORRECTION_DBSCAN_INTERP = "dbscan_interp"
# 結果の記録先（各行の fit_metrics のキー）
ORIENTATION_CORRECTION_METRICS_KEY = "orientation_correction"
# 向き推定の記録（box_orientation_inference.ORIENTATION_METRICS_KEY と同じ）
_ESTIMATION_METRICS_KEY = "orientation_estimation"

DEFAULT_EPS_DEG = 20.0
DEFAULT_MIN_SAMPLES = 3
DEFAULT_MIN_FRAMES = 4

STATUS_CORRECTED = "corrected"
STATUS_KEPT = "kept"
STATUS_SKIPPED = "skipped"


def correction_settings(box_fitting_params: dict[str, Any] | None) -> dict[str, Any]:
    """run のパラメータから角度補正の設定を取り出す（無ければ補正しない）."""
    orientation = (box_fitting_params or {}).get("orientation") or {}
    params = orientation.get("correction") or {}
    return {
        "methods": {k: v for k, v in (params.get("methods") or {}).items() if v},
        "estimations": {k: v for k, v in (orientation.get("methods") or {}).items() if v},
        "eps_deg": float(params.get("eps_deg", DEFAULT_EPS_DEG)),
        "min_samples": int(params.get("min_samples", DEFAULT_MIN_SAMPLES)),
        "min_frames": int(params.get("min_frames", DEFAULT_MIN_FRAMES)),
    }


def yaw_of_pose(pose: dict[str, Any]) -> float:
    """ego_pose の yaw [rad]（基準 ego 座標 ⇔ グローバル座標の変換に使う）."""
    w, x, y, z = (float(v) for v in pose["rotation"])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def dbscan_eps(eps_deg: float) -> float:
    """角度の閾値 [度] を単位円上の距離（弦の長さ）にする."""
    return 2.0 * math.sin(math.radians(eps_deg) / 2.0)


def cluster_yaws(yaws: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """角度 [rad] を単位円上の点にして DBSCAN する. ノイズは -1."""
    import open3d as o3d

    points = np.column_stack([np.cos(yaws), np.sin(yaws), np.zeros_like(yaws)])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return np.asarray(
        cloud.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False)
    )


def largest_cluster(
    labels: np.ndarray, timestamps: np.ndarray, center_time: float,
) -> int | None:
    """最大クラスタの番号. 同じ大きさなら、平均時刻がトラックの中央に近い方.

    クラスタが無ければ None。
    """
    valid = labels[labels >= 0]
    if valid.size == 0:
        return None
    counts = np.bincount(valid)
    candidates = np.flatnonzero(counts == counts.max())
    return int(min(
        candidates,
        key=lambda c: (abs(float(timestamps[labels == c].mean()) - center_time), int(c)),
    ))


def interpolate_yaw(
    timestamp: float, times: np.ndarray, yaws: np.ndarray,
) -> tuple[float, list[int]]:
    """最大クラスタのフレーム（times は昇順）から、その時刻の向きを線形円周補間する.

    片側にしか無いときは最寄りの値をそのまま使う。

    Returns:
        (向き [rad], 使ったフレームの添字)
    """
    after = int(np.searchsorted(times, timestamp))
    if after == 0:
        return float(yaws[0]), [0]
    if after == times.size:
        return float(yaws[-1]), [times.size - 1]
    before = after - 1
    ratio = (timestamp - times[before]) / (times[after] - times[before])
    delta = wrap_angle(float(yaws[after] - yaws[before]))  # 短い側の弧で補間する
    return wrap_angle(float(yaws[before]) + ratio * delta), [before, after]


def _clusterable(fit: dict[str, Any]) -> bool:
    """クラスタリングに使えるフレームか.

    角度推定直角ずれのフレームと、推定できず前後が決まっていないフレームは使わない。
    """
    estimation = (fit.get("fit_metrics") or {}).get(_ESTIMATION_METRICS_KEY) or {}
    return (estimation.get("status") == ORIENTATION_STATUS_ESTIMATED
            and not estimation.get("perpendicular_flag"))


def correct_track_orientation(
    label: str | None,
    fits: list[dict[str, Any]],
    sample_tokens: list[str],
    poses: list[dict[str, Any] | None],
    settings: dict[str, Any],
) -> int:
    """1 つのグローバルトラックの向きを補正する（fit をその場で書き換える）.

    Args:
        fits: トラックの各フレームのボックス（fit_metrics を持つ）
        sample_tokens: fits と同じ並びの sample
        poses: fits と同じ並びの基準 ego_pose（rotation / timestamp を持つ）

    Returns:
        補正したボックスの数
    """
    method = settings["methods"].get(label or "")
    if method != ORIENTATION_CORRECTION_DBSCAN_INTERP:
        return 0

    def skip(reason: str) -> int:
        for fit in fits:
            fit.setdefault("fit_metrics", {})[ORIENTATION_CORRECTION_METRICS_KEY] = {
                "method": method, "status": STATUS_SKIPPED, "reason": reason,
                "num_frames": len(fits),
            }
        return 0

    if not settings["estimations"].get(label or ""):
        return skip("no_estimation")
    if len(fits) < settings["min_frames"]:
        return skip("too_few_frames")
    if any(pose is None or pose.get("timestamp") is None for pose in poses):
        return skip("no_reference_pose")

    usable = np.array([_clusterable(fit) for fit in fits], dtype=bool)
    if not usable.any():
        return skip("no_targets")

    timestamps = np.array([float(pose["timestamp"]) for pose in poses])
    ego_yaws = np.array([yaw_of_pose(pose) for pose in poses])
    yaws_global = np.array([float(fit["yaw_ego"]) for fit in fits]) + ego_yaws

    labels = np.full(len(fits), -1, dtype=int)
    labels[usable] = cluster_yaws(
        yaws_global[usable], dbscan_eps(settings["eps_deg"]), settings["min_samples"],
    )
    center_time = (float(timestamps.min()) + float(timestamps.max())) / 2.0
    cluster = largest_cluster(labels, timestamps, center_time)
    if cluster is None:
        return skip("no_cluster")

    in_cluster = np.flatnonzero(labels == cluster)
    in_cluster = in_cluster[np.argsort(timestamps[in_cluster])]
    cluster_times = timestamps[in_cluster]
    cluster_yaws_global = yaws_global[in_cluster]
    base = {"method": method, "num_frames": len(fits), "cluster_size": int(in_cluster.size)}

    num_corrected = 0
    for index, fit in enumerate(fits):
        metrics = fit.setdefault("fit_metrics", {})
        if labels[index] == cluster:
            metrics[ORIENTATION_CORRECTION_METRICS_KEY] = {**base, "status": STATUS_KEPT}
            continue
        yaw_global, used = interpolate_yaw(
            float(timestamps[index]), cluster_times, cluster_yaws_global,
        )
        yaw_target = wrap_angle(yaw_global - float(ego_yaws[index]))
        # 回し方は 1 フレーム単位の補正の part_hoe と同じ（45 度の場合分け＋凸包への当て直し）
        applied = apply_orientation(
            fit, ORIENTATION_METHOD_PART_HOE, yaw_target,
            standard_width=None,
            hull_xy=(fit.get("hull_xy") or {}).get("points"),
        )
        metrics[ORIENTATION_CORRECTION_METRICS_KEY] = {
            **base,
            "status": STATUS_CORRECTED,
            # クラスタリングに使わなかったフレームか（直角ずれ・推定失敗）
            "excluded": not bool(usable[index]),
            "yaw_target": yaw_target,
            "interpolated_from": [sample_tokens[int(in_cluster[i])] for i in used],
            **{key: applied.get(key) for key in (
                "applied", "yaw_before", "rotation_deg", "refit", "refit_center_shift")},
        }
        num_corrected += 1
    return num_corrected
