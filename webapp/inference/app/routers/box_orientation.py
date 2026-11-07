"""Box Orientation Estimation のエンドポイント.

保存済みの Box Fitting run に対して、AI モデル（PartHOE / Orient-Anything V1 Large）で
ボックスの向きを推定する。
**ボックスには反映せず、推定結果を返すだけ**（webapp が run に保存する）。
向き推定をボックスへ反映する前に保存した古い run の確認用。新しい run は
Box Fitting のパス 2 で推定・反映する（services/box_orientation.apply_orientation）。
他ステップと同じジョブ方式。
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import get_settings
from app.core.jobs import Job, JobManager, get_job_manager
from app.core.logging import get_logger
from app.schemas.box_orientation import OrientationRequest
from app.schemas.depth_boxfitting import JobResponse
from app.services.box_orientation import (
    MODEL_FOR_METHOD,
    ORIENTATION_STATUS_ESTIMATED,
    ORIENTATION_STATUS_NO_BOX,
    ORIENTATION_STATUS_UNSUPPORTED,
    SUPPORTED_ORIENTATION_METHODS,
    group_candidates,
)
from app.services.box_orientation_inference import estimate_groups, result_base

logger = get_logger(__name__)

router = APIRouter(prefix="/box-orientation", tags=["Box Orientation Estimation"])

KIND = "box_orientation"


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
    by_model: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        if group["method"] not in SUPPORTED_ORIENTATION_METHODS:
            emit({**result_base(group), "status": ORIENTATION_STATUS_UNSUPPORTED})
            done += 1
        elif group["primary"] is None or not group["primary"].get("center_ego"):
            # ボックスが無いと ego の向きに直せない
            emit({**result_base(group), "status": ORIENTATION_STATUS_NO_BOX})
            done += 1
        else:
            by_model.setdefault(MODEL_FOR_METHOD[group["method"]], []).append(group)
    job.set_progress(done)

    # モデルごとに載せ替える（MAX_RESIDENT_MODELS=1 でも 1 回ずつで済む）
    for model, model_groups in by_model.items():
        if job.cancel_requested():
            break
        def on_batch(n: int, model: str = model) -> None:
            nonlocal done
            done += n
            job.set_progress(done, message=model)

        estimate_groups(model, model_groups, dataroot, emit,
                        cancelled=job.cancel_requested, on_batch=on_batch)

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
