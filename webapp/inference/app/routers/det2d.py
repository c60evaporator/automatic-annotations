"""2D Object Detection のエンドポイント.

## なぜ「1推論1リクエスト」でも「ストリーミング」でもなくジョブ方式か

Streamlit はウィジェット操作のたびにスクリプトを頭から再実行する。

  - 1推論ずつ呼ぶ場合、ループを Streamlit 側で回すことになる。
    ループ中はページが固まり、途中でスライダーを触ると再実行が入って
    ループが最初からやり直しになる。264 回の HTTP 往復も無駄が多い。
  - ストリーミング応答（SSE / chunked）も、受信ループの間スクリプトが
    ブロックされる点は同じで、再実行で接続が切れると推論も道連れになる。

ジョブ方式なら推論はサーバー側で走り続けるので、再実行やページ移動を
しても失われない。UI は job_id を持って進捗を取りに行くだけでよい。

## 進捗の粒度

ループは sample → camera → category group の3重で、
今回の設定だと 11 × 6 × 4 = 264 回になる。
progress は最内周（推論1回）ごとに更新し、
partial（描画に使える結果）は camera 単位で積む。
UI は「サンプル3枚目の CAM_FRONT を処理中」を出しつつ、
終わったカメラから順に結果を描ける。
"""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core import models as model_registry
from app.core.config import get_settings
from app.core.jobs import Job, JobManager, get_job_manager
from app.core.logging import get_logger
from app.schemas.det2d import (
    BBox2D,
    Detection2DFrameResult,
    Detection2DRequest,
    JobResponse,
)
from app.services.postprocess import cross_class_nms

logger = get_logger(__name__)

router = APIRouter(prefix="/detection2d", tags=["2D Object Detection"])

KIND = "detection_2d"


def _run_detection(req: Detection2DRequest, job: Job) -> dict:
    """ジョブ本体（ワーカースレッドで実行される）."""
    settings = get_settings()

    # sample を外側にするため、sample_idx でまとめる
    by_sample: dict[int, list] = defaultdict(list)
    for f in req.frames:
        by_sample[f.sample_idx].append(f)
    sample_indices = sorted(by_sample)

    groups = req.label_groups
    total_inferences = sum(len(by_sample[i]) for i in sample_indices) * max(1, len(groups))
    job.set_progress(0, total_inferences, "モデルを準備中...")

    done = 0
    total_time = 0.0
    num_boxes = 0
    all_frames: list[Detection2DFrameResult] = []

    with model_registry.use_gpu("grounding_dino") as detector:
        for sample_idx in sample_indices:
            if job.cancel_requested():
                break

            for frame in sorted(by_sample[sample_idx], key=lambda f: f.channel or ""):
                if job.cancel_requested():
                    break

                path = settings.DATA_ROOT / req.dataroot / frame.filename
                raw_boxes: list[dict] = []
                error: str | None = None
                frame_time = 0.0

                # カテゴリグループごとに推論する（プロンプトをまとめる単位）。
                # 閾値・同一クラス NMS はグループ単位で完結する
                for group in groups:
                    if job.cancel_requested():
                        break
                    started = time.perf_counter()
                    try:
                        detected = detector.predict(
                            image_path=path,
                            group_name=group.name,
                            labels=group.labels,
                            score_threshold=group.score_threshold,
                            nms_same_class_iou=group.nms_same_class_iou,
                            stub_delay_sec=req.stub_delay_sec,
                        )
                        for b in detected:
                            b.setdefault("group", group.name)
                        raw_boxes.extend(detected)
                    except Exception as exc:  # noqa: BLE001
                        # 1フレームの失敗で全体を止めない。
                        # 画像欠損などは実データでは普通に起こる
                        logger.warning("detection failed for %s (%s): %s",
                                       frame.filename, group.name, exc)
                        error = f"{type(exc).__name__}: {exc}"
                    finally:
                        elapsed = time.perf_counter() - started
                        frame_time += elapsed
                        total_time += elapsed
                        done += 1
                        job.set_progress(
                            done,
                            message=(f"sample {sample_idx} / "
                                     f"{frame.channel or frame.sample_data_token} / "
                                     f"{group.name}"),
                        )

                # クラスをまたぐ NMS は全グループが揃ってから適用する。
                # 同じ物体が別グループで二重に検出されるのを潰す
                kept = cross_class_nms(raw_boxes, req.nms_cross_class_iou)

                result = Detection2DFrameResult(
                    sample_data_token=frame.sample_data_token,
                    sample_token=frame.sample_token,
                    sample_idx=frame.sample_idx,
                    channel=frame.channel,
                    boxes=[BBox2D(**b) for b in kept],
                    inference_time=round(frame_time, 3),
                    error=error,
                )
                num_boxes += len(kept)
                all_frames.append(result)
                # カメラ1枚終わるごとに UI へ流す
                job.append_partial(result.model_dump())

    return {
        "num_frames": len(all_frames),
        "num_boxes": num_boxes,
        "inference_time": round(total_time, 3),
        "frames": [f.model_dump() for f in all_frames],
    }


@router.post("/jobs", response_model=JobResponse, status_code=202)
def create_detection_job(
    req: Detection2DRequest,
    jobs: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """2D 検出ジョブを登録する（即座に job_id を返す）."""
    job = jobs.submit(KIND, lambda j: _run_detection(req, j))
    return JobResponse(**job.to_dict(since=0))


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_detection_job(
    job_id: str,
    since: int = Query(
        0, ge=0,
        description="受け取り済みの部分結果の件数。これ以降の差分だけ返す",
    ),
    jobs: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """ジョブの進捗と、まだ受け取っていない部分結果を返す.

    since を渡すことで、ポーリングのたびに全結果を送り直さずに済む。
    """
    job = jobs.get(job_id)
    if job is None or job.kind != KIND:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return JobResponse(**job.to_dict(since=since))


@router.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_detection_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager)
) -> JobResponse:
    """ジョブのキャンセルを要求する（次の推論の境界で止まる）."""
    job = jobs.get(job_id)
    if job is None or job.kind != KIND:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    jobs.cancel(job_id)
    return JobResponse(**job.to_dict())


@router.get("/jobs", response_model=list[JobResponse])
def list_detection_jobs(
    jobs: JobManager = Depends(get_job_manager),
) -> list[JobResponse]:
    return [JobResponse(**j.to_dict()) for j in jobs.list(KIND)]
