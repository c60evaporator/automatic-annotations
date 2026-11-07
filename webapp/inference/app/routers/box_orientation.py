"""Box Orientation Estimation のエンドポイント.

保存済みの Box Fitting run に対して、AI モデルでボックスの向きを推定する。
**ボックスには反映せず、推定結果を返すだけ**（webapp が run に保存する）。
他ステップと同じジョブ方式。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from PIL import Image

from app.core import models as model_registry
from app.core.config import get_settings
from app.core.jobs import Job, JobManager, get_job_manager
from app.core.logging import get_logger
from app.schemas.box_orientation import OrientationRequest
from app.schemas.depth_boxfitting import JobResponse
from app.services.box_orientation import (
    ORIENTATION_METHOD_PART_HOE,
    ORIENTATION_STATUS_ESTIMATED,
    ORIENTATION_STATUS_FAILED,
    ORIENTATION_STATUS_NO_BOX,
    ORIENTATION_STATUS_UNSUPPORTED,
    SUPPORTED_ORIENTATION_METHODS,
    camera_relative_to_yaw,
    group_candidates,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/box-orientation", tags=["Box Orientation Estimation"])

KIND = "box_orientation"
# 1 回にモデルへ渡すクロップ数
BATCH_SIZE = 32


def _result_base(group: dict[str, Any]) -> dict[str, Any]:
    primary = group["primary"] or group["source"]
    return {
        "box_fitting_id": primary["box_fitting_id"],
        "global_track_id": primary["global_track_id"],
        "sample_token": primary["sample_token"],
        "label": group["label"],
        "method": group["method"],
    }


def _estimate_part_hoe(
    groups: list[dict[str, Any]], dataroot: Path, job: Job,
    done: int, emit,
) -> int:
    """PartHOE で推定する. 画像はフレームごとに 1 回だけ読む."""
    from app.models_impl.part_hoe import crop_person
    from app.services.lshape_filter import sensor_origin_xy

    # 同じフレームのグループを続けて処理し、画像を使い回す
    groups = sorted(groups, key=lambda g: g["source"]["sample_data_token"])
    with model_registry.use_gpu("part_hoe") as estimator:
        image_cache: dict[str, np.ndarray] = {}
        for start in range(0, len(groups), BATCH_SIZE):
            if job.cancel_requested():
                break
            batch = groups[start:start + BATCH_SIZE]
            crops, valid = [], []
            for group in batch:
                source = group["source"]
                try:
                    path = source["filename"]
                    if path not in image_cache:
                        image_cache.clear()  # 直前のフレームだけ持つ
                        with Image.open(dataroot / path) as img:
                            image_cache[path] = np.asarray(img.convert("RGB"))
                    crops.append(crop_person(image_cache[path], source["bbox"]))
                    valid.append(group)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("crop failed for %s: %s", source["box_fitting_id"], exc)
                    emit({**_result_base(group), "status": ORIENTATION_STATUS_FAILED,
                          "error": f"{type(exc).__name__}: {exc}"})
            if crops:
                probs = estimator.predict(crops)
                angles = estimator.to_degrees(probs)
                for group, prob, angle in zip(valid, probs, angles, strict=True):
                    source, primary = group["source"], group["primary"]
                    camera_xy = sensor_origin_xy(
                        source["calibrated_sensor"], source["ego_pose"],
                        source.get("reference_ego_pose"),
                    )
                    center = primary["center_ego"]
                    emit({
                        **_result_base(group),
                        "status": ORIENTATION_STATUS_ESTIMATED,
                        "source_box_fitting_id": source["box_fitting_id"],
                        "channel": source["channel"],
                        "bbox": list(source["bbox"]),
                        "angle_deg": float(angle),
                        "confidence": float(prob.max()),
                        "yaw_ego": camera_relative_to_yaw(
                            float(angle), camera_xy, (center[0], center[1])
                        ),
                    })
            done += len(batch)
            job.set_progress(done, message="PartHOE")
    return done


def _run_orientation(req: OrientationRequest, job: Job) -> dict:
    settings = get_settings()
    dataroot = settings.DATA_ROOT / req.dataroot
    started = time.perf_counter()

    groups = group_candidates(
        [c.model_dump() for c in req.candidates], req.methods
    )
    job.set_progress(0, len(groups), "モデルを準備中...")
    results: list[dict[str, Any]] = []

    def emit(result: dict[str, Any]) -> None:
        results.append(result)
        job.append_partial({"kind": "orientation", "data": result})

    done = 0
    by_method: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        if group["method"] not in SUPPORTED_ORIENTATION_METHODS:
            emit({**_result_base(group), "status": ORIENTATION_STATUS_UNSUPPORTED})
            done += 1
        elif group["primary"] is None or not group["primary"].get("center_ego"):
            # ボックスが無いと ego の向きに直せない
            emit({**_result_base(group), "status": ORIENTATION_STATUS_NO_BOX})
            done += 1
        else:
            by_method.setdefault(group["method"], []).append(group)
    job.set_progress(done)

    part_hoe_groups = by_method.get(ORIENTATION_METHOD_PART_HOE, [])
    if part_hoe_groups:
        done = _estimate_part_hoe(part_hoe_groups, dataroot, job, done, emit)

    return {
        "num_groups": len(groups),
        "num_estimated": sum(
            1 for r in results if r["status"] == ORIENTATION_STATUS_ESTIMATED
        ),
        "inference_time": round(time.perf_counter() - started, 3),
        "orientations": results,
    }


@router.post("/jobs", response_model=JobResponse, status_code=202)
def create_orientation_job(
    req: OrientationRequest, jobs: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """向き推定ジョブを登録する."""
    job = jobs.submit(KIND, lambda j: _run_orientation(req, j))
    return JobResponse(**job.to_dict(since=0))


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_orientation_job(
    job_id: str,
    since: int = Query(0, ge=0, description="受け取り済みの部分結果の件数"),
    jobs: JobManager = Depends(get_job_manager),
) -> JobResponse:
    job = jobs.get(job_id)
    if job is None or job.kind != KIND:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return JobResponse(**job.to_dict(since=since))


@router.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_orientation_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager)
) -> JobResponse:
    job = jobs.get(job_id)
    if job is None or job.kind != KIND:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    jobs.cancel(job_id)
    return JobResponse(**job.to_dict())
