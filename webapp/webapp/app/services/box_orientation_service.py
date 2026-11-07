"""AI モデルによるボックス向き推定の実行と保存（webapp 側）.

保存済みの Box Fitting run に対して推論サーバーの向き推定ジョブを投げ、
結果を各グループの代表行の ``fit_metrics["orientation_estimation"]`` に保存する。
**ボックス（yaw_ego）には反映しない。** 表示（Box Fitting タブの矢印）で
推定の妥当性を確かめるための記録。

推定はシーン全体を 1 ジョブで行う。PartHOE は軽く、画像の読み込みが
大半を占めるため、ポーリングしながら完了まで待つ。
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import read_only_session, session_scope
from app.repositories.depth_boxfitting import DepthBoxFittingRepository
from app.repositories.sensor import SensorRepository

logger = get_logger(__name__)

# 結果の保存先（BoxFitting3D.fit_metrics のキー）
ORIENTATION_METRICS_KEY = "orientation_estimation"
ORIENTATION_STATUS_ESTIMATED = "estimated"
POLL_INTERVAL_SEC = 0.5


def build_orientation_payload(
    scene_token: str, params_id: str, dataroot: str
) -> dict[str, Any]:
    """推論サーバーへ送るリクエストを組み立てる.

    候補は run のうち global_track_id を持つ全行（カメラごと）。
    どのカメラで推定するかは推論サーバーがグループごとに選ぶ。
    """
    settings = get_settings()
    with read_only_session() as session:
        rows = DepthBoxFittingRepository(session).list_orientation_candidates(params_id)
        frames = SensorRepository(session).list_frames_by_scene(
            scene_token, keyframe_only=True
        )
    frame_by_token = {f["token"]: f for f in frames}
    # 点群・ボックスの座標系（sample ごとの基準 ego）。Box Fitting の実行時と同じ
    reference_ego_poses = {
        f["sample_token"]: f["ego_pose"]
        for f in frames
        if f["channel"] == settings.EGO_REFERENCE_CHANNEL and f["ego_pose"]
    }

    candidates = []
    for row in rows:
        frame = frame_by_token.get(row["sample_data_token"])
        if frame is None or row["xmax"] <= row["xmin"] or row["ymax"] <= row["ymin"]:
            continue
        candidates.append({
            "box_fitting_id": row["id"],
            "global_track_id": row["global_track_id"],
            "sample_token": frame["sample_token"],
            "sample_data_token": row["sample_data_token"],
            "label": row["label"],
            "channel": frame["channel"],
            "filename": frame["filename"],
            "bbox": [row["xmin"], row["ymin"], row["xmax"], row["ymax"]],
            # 深度を使わなかったカメラも、深度の点数を数える
            "num_points": int(row["num_points_depth"] or 0)
            + int(row["num_points_lidar"] or 0),
            "is_primary": bool(row["is_primary"]),
            "center_ego": row["center_ego"] if row["is_primary"] else None,
            "calibrated_sensor": frame["calibrated_sensor"],
            "ego_pose": frame["ego_pose"],
            "reference_ego_pose": reference_ego_poses.get(frame["sample_token"]),
        })

    return {
        "dataroot": dataroot,
        "methods": dict(settings.BOX_ORIENTATION_ESTIMATIONS),
        "candidates": candidates,
    }


def save_orientation_results(params_id: str, results: list[dict[str, Any]]) -> int:
    """推定結果を代表行の fit_metrics に保存する（前回の結果は置き換える）."""
    values = {}
    for result in results:
        values[result["box_fitting_id"]] = {
            key: result.get(key)
            for key in ("method", "status", "source_box_fitting_id", "channel",
                        "bbox", "angle_deg", "confidence", "yaw_ego", "error")
        }
    with session_scope() as session:
        DepthBoxFittingRepository(session).replace_fit_metrics_entry(
            params_id, ORIENTATION_METRICS_KEY, values
        )
    return sum(1 for r in results if r.get("status") == ORIENTATION_STATUS_ESTIMATED)


def run_orientation_estimation(
    scene_token: str,
    params_id: str,
    dataroot: str,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """向き推定ジョブを投げ、完了まで待って結果を保存する.

    Returns:
        ``{"status", "num_groups", "num_estimated", "error"}``
    """
    from app.services.inference_client import get_orientation_job, submit_orientation

    payload = build_orientation_payload(scene_token, params_id, dataroot)
    job = submit_orientation(payload)
    results: list[dict[str, Any]] = []
    since = 0
    while True:
        job = get_orientation_job(job["job_id"], since=since)
        new = job.get("partial") or []
        results += [item["data"] for item in new]
        since += len(new)
        if on_progress is not None:
            on_progress(job)
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(POLL_INTERVAL_SEC)

    num_estimated = 0
    if job["status"] == "succeeded":
        num_estimated = save_orientation_results(params_id, results)
    logger.info("orientation estimation %s: %d groups, %d estimated",
                job["status"], len(results), num_estimated)
    return {
        "status": job["status"],
        "num_groups": len(results),
        "num_estimated": num_estimated,
        "error": job.get("error"),
    }
