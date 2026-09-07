"""Depth Estimation & Box Fitting のエンドポイント.

Detection2D / Instance Tracking と同じジョブ方式に揃えてある。

## 処理の流れ

  1. 深度推定    カメラを外側、sample を内側にループ
                 （将来 Pose-Conditioned な複数フレーム推定へ広げるため、
                  同一カメラの連続フレームが並ぶ順序にしておく）
  2. LiDAR 統合  sample ごと。use_lidar に関係なく常に実行する
                 （UI が比較用に生 LiDAR と地面を表示できるようにする）
  3. Box Fitting インスタンスごと

## 進捗

段階で粒度が違うので、total は 3 段階の合計にする。
どの段階かは message で示す。
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core import models as model_registry
from app.core.config import get_settings
from app.core.jobs import Job, JobManager, get_job_manager
from app.core.logging import get_logger
from app.schemas.depth_boxfitting import BoxFittingRequest, JobResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/depth-boxfitting", tags=["Depth Estimation & Box Fitting"])

KIND = "depth_boxfitting"


def _run_boxfitting(req: BoxFittingRequest, job: Job) -> dict:
    """ジョブ本体（ワーカースレッドで実行される）."""
    settings = get_settings()
    output_dir = settings.DERIVED_ROOT / req.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(req.camera_frames) + len(req.lidar_frames) + len(req.instances)
    job.set_progress(0, total, "モデルを準備中...")

    done = 0
    started_all = time.perf_counter()
    depth_results: list[dict[str, Any]] = []
    lidar_results: list[dict[str, Any]] = []
    box_results: list[dict[str, Any]] = []

    mask_params = req.mask_params or {}
    lidar_params = req.lidar_params or {}
    min_lidar_points = int(lidar_params.get("min_points", 10))

    with model_registry.use_gpu("depth_anything") as pipeline:
        # --- 1. 深度推定（カメラ外側）-----------------------------------
        by_channel: dict[str, list[Any]] = defaultdict(list)
        for frame in req.camera_frames:
            by_channel[frame.channel].append(frame)

        for channel, frames in by_channel.items():
            for frame in sorted(frames, key=lambda f: f.timestamp):
                if job.cancel_requested():
                    break
                try:
                    result = pipeline.estimate_depth(
                        frame.model_dump(), output_dir,
                        downscale=req.depth_downscale,
                        stub_delay_sec=req.stub_delay_sec,
                    )
                except Exception as exc:  # noqa: BLE001
                    # 1 フレームの失敗で全体を止めない
                    logger.warning("depth estimation failed for %s: %s",
                                   frame.filename, exc)
                    result = {
                        "sample_data_token": frame.sample_data_token,
                        "depth_path": "", "depth_width": 0, "depth_height": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                depth_results.append(result)
                job.append_partial({"kind": "depth", "data": result})
                done += 1
                job.set_progress(done, message=f"深度推定 {channel}")

        # --- 2. LiDAR の統合と地面除去 ----------------------------------
        for frame in sorted(req.lidar_frames, key=lambda f: f.timestamp):
            if job.cancel_requested():
                break
            try:
                result = pipeline.load_lidar(
                    frame.model_dump(), output_dir,
                    num_sweeps=req.num_lidar_sweeps,
                    stub_delay_sec=req.stub_delay_sec,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("lidar loading failed for %s: %s",
                               frame.sample_token, exc)
                result = {
                    "sample_token": frame.sample_token,
                    "sample_data_token": frame.sample_data_token,
                    "pointcloud_path": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            lidar_results.append(result)
            job.append_partial({"kind": "lidar", "data": result})
            done += 1
            job.set_progress(done, message="LiDAR 統合")

        # --- 3. マスク処理と Box Fitting --------------------------------
        for instance in req.instances:
            if job.cancel_requested():
                break
            payload = instance.model_dump()
            try:
                # クロージング後のマスクは webapp 側で再計算できないため、
                # ここで作ったものをそのまま結果に載せる
                payload["mask_rle"] = pipeline.close_mask(
                    payload["mask_rle"],
                    dilation=int(mask_params.get("dilation", 1)),
                    erosion=int(mask_params.get("erosion", 1)),
                )
                result = pipeline.fit_box(
                    payload,
                    use_lidar=req.use_lidar,
                    min_lidar_points=min_lidar_points,
                    stored_points_max=req.stored_points_max,
                    stub_delay_sec=req.stub_delay_sec,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("box fitting failed for %s: %s",
                               instance.instance_tracking_2d_id, exc)
                result = {
                    "instance_tracking_2d_id": instance.instance_tracking_2d_id,
                    "sample_data_token": instance.sample_data_token,
                    "track_id": instance.track_id, "label": instance.label,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            box_results.append(result)
            job.append_partial({"kind": "box", "data": result})
            done += 1
            job.set_progress(done, message="Box Fitting")

    return {
        "num_depth_frames": len(depth_results),
        "num_lidar_frames": len(lidar_results),
        "num_box_fittings": len(box_results),
        "num_fitted": sum(1 for b in box_results if b.get("status") == "fitted"),
        "inference_time": round(time.perf_counter() - started_all, 3),
        "depth_estimations": depth_results,
        "lidar_pointclouds": lidar_results,
        "box_fittings": box_results,
    }


@router.post("/jobs", response_model=JobResponse, status_code=202)
def create_boxfitting_job(
    req: BoxFittingRequest,
    jobs: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """Depth Estimation & Box Fitting ジョブを登録する."""
    job = jobs.submit(KIND, lambda j: _run_boxfitting(req, j))
    return JobResponse(**job.to_dict(since=0))


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_boxfitting_job(
    job_id: str,
    since: int = Query(0, ge=0, description="受け取り済みの部分結果の件数"),
    jobs: JobManager = Depends(get_job_manager),
) -> JobResponse:
    job = jobs.get(job_id)
    if job is None or job.kind != KIND:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return JobResponse(**job.to_dict(since=since))


@router.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_boxfitting_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager)
) -> JobResponse:
    job = jobs.get(job_id)
    if job is None or job.kind != KIND:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    jobs.cancel(job_id)
    return JobResponse(**job.to_dict())


@router.get("/jobs", response_model=list[JobResponse])
def list_boxfitting_jobs(
    jobs: JobManager = Depends(get_job_manager),
) -> list[JobResponse]:
    return [JobResponse(**j.to_dict()) for j in jobs.list(KIND)]


@router.get("/status")
def status() -> dict[str, str]:
    return {"kind": KIND, "implemented": "stub"}
