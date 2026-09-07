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
    # 画像の実体はデータセットのマウント下にある
    dataroot = settings.DATA_ROOT / req.dataroot

    total = len(req.camera_frames) + len(req.lidar_frames) + len(req.instances)
    job.set_progress(0, total, "モデルを準備中...")

    done = 0
    started_all = time.perf_counter()
    depth_results: list[dict[str, Any]] = []
    lidar_results: list[dict[str, Any]] = []
    box_results: list[dict[str, Any]] = []

    mask_params = req.mask_params or {}
    depth_params = req.depth_params or {}
    # 深度の上限。これ以遠は点群にしない（遠方は誤差が大きく、
    # 入れるとクラスタリングが引きずられる）
    max_depth = depth_params.get("max_depth", settings.DEPTH_MAX_DISTANCE)

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
                        dataroot=dataroot,
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
                    dataroot=dataroot,
                    num_sweeps=req.num_lidar_sweeps,
                    stub_delay_sec=req.stub_delay_sec,
                )
            except NotImplementedError as exc:
                # LiDAR 未実装のまま use_lidar=False で回す場合は正常系。
                # 結果を残さず、UI 側は「ファイル無し」として扱う
                logger.info("lidar skipped for %s: %s", frame.sample_token, exc)
                done += 1
                job.set_progress(done, message="LiDAR スキップ")
                continue
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

        # --- 3. マスク処理とインスタンス点群・Box Fitting ----------------
        # フレーム単位でまとめて渡す。深度マップと空マスクを
        # 1 フレームにつき 1 回だけ読めば済むようにするため
        instances_by_frame: dict[str, list[Any]] = defaultdict(list)
        for instance in req.instances:
            instances_by_frame[instance.sample_data_token].append(instance)

        depth_by_token = {d["sample_data_token"]: d for d in depth_results}
        frame_by_token = {f.sample_data_token: f for f in req.camera_frames}

        for token, frame_instances in instances_by_frame.items():
            if job.cancel_requested():
                break
            depth_result = depth_by_token.get(token)
            frame = frame_by_token.get(token)
            payloads = [i.model_dump() for i in frame_instances]

            if depth_result is None or depth_result.get("error") or frame is None:
                # 深度が無いフレームのインスタンスは、理由を残して飛ばす
                results = [{
                    "instance_tracking_2d_id": i["instance_tracking_2d_id"],
                    "sample_data_token": i["sample_data_token"],
                    "track_id": str(i["track_id"]), "label": i["label"],
                    "status": "no_points",
                } for i in payloads]
            else:
                try:
                    results = pipeline.process_frame_instances(
                        payloads, depth_result, output_dir,
                        frame=frame.model_dump(),
                        mask_params=mask_params,
                        depth_params=req.depth_params or {},
                        use_lidar=req.use_lidar,
                        stored_points_max=req.stored_points_max,
                        max_depth=max_depth,
                        stub_delay_sec=req.stub_delay_sec,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("instance processing failed for %s: %s", token, exc)
                    results = [{
                        "instance_tracking_2d_id": i["instance_tracking_2d_id"],
                        "sample_data_token": i["sample_data_token"],
                        "track_id": str(i["track_id"]), "label": i["label"],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    } for i in payloads]

            for result in results:
                box_results.append(result)
                job.append_partial({"kind": "box", "data": result})
            done += len(payloads)
            job.set_progress(done, message="インスタンス点群")

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
