"""Instance Tracking (SAM2) のエンドポイント.

Detection2D と同じジョブ方式（POST で登録 → GET でポーリング）に揃えてある。
UI 側のポーリング処理を共通化できるようにするため。

## 処理の流れ

Detection2D の sample_interval ごとに「プロンプト区間」を作り、
区間 × カメラごとに SAM2 を回す。

    区間1: sample 0 → 4   （sample 0 のボックスをプロンプトに伝播）
    区間2: sample 4 → 8   （sample 4 のボックスをプロンプトに伝播）
    ...

区間の境界（sample 4 など）には、前の区間から伝播したインスタンスと
今回のプロンプトで得たインスタンスの両方が現れる。
これを IoU で貪欲マッチングし、track_id を引き継ぐ。
引き継がないと区間ごとに id が振り直され、シーンを通したトラックにならない。

境界フレームの結果は、**前の区間の伝播結果ではなく今回のプロンプト側**を
採用する。プロンプトは検出器の出力そのもので、伝播より信頼できるため。
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core import models as model_registry
from app.core.config import get_settings
from app.core.jobs import Job, JobManager, get_job_manager
from app.core.logging import get_logger
from app.schemas.instance_tracking import (
    InstanceTrackingRequest,
    JobResponse,
    TrackedInstance,
    TrackingFrameResult,
)
from app.services.tracking_match import assign_track_ids

logger = get_logger(__name__)

router = APIRouter(prefix="/instance-tracking", tags=["Instance Tracking"])

KIND = "instance_tracking"


def _build_segments(
    frames_by_channel: dict[str, list[Any]], sample_interval: int
) -> dict[str, list[list[Any]]]:
    """チャンネルごとに、プロンプト区間へフレームを切り分ける.

    区間は「プロンプトを与える sample のキーフレーム」から
    「次のプロンプト sample のキーフレーム」まで（両端を含む）。
    境界を共有させることで、伝播結果と新規検出を突き合わせられる。

    NOTE: sample_idx の範囲ではなく**タイムスタンプ**で切ること。
    nuScenes では、ある sample の sweep はその sample のキーフレームより
    「手前」にある（キーフレームが sample 内で最も新しい）。
    sample_idx で切ると区間の先頭が sweep になり、
    キーフレームに結び付いたプロンプトが拾えない。
    """
    segments: dict[str, list[list[Any]]] = {}
    for channel, frames in frames_by_channel.items():
        ordered = sorted(frames, key=lambda f: (f.timestamp, f.sample_idx))
        key_frames = [f for f in ordered if f.is_key_frame]
        if not key_frames:
            segments[channel] = []
            continue

        # プロンプトを与えるキーフレーム（interval の倍数 + 末尾）
        anchors = [f for f in key_frames if f.sample_idx % sample_interval == 0]
        if anchors and anchors[-1].sample_idx != key_frames[-1].sample_idx:
            anchors.append(key_frames[-1])
        if not anchors:
            anchors = [key_frames[0]]

        channel_segments: list[list[Any]] = []
        for start, end in zip(anchors, anchors[1:]):
            channel_segments.append([
                f for f in ordered
                if start.timestamp <= f.timestamp <= end.timestamp
            ])
        if not channel_segments:
            # プロンプト sample が1つしかない場合は、その1フレームだけ
            channel_segments.append([anchors[0]])
        segments[channel] = channel_segments
    return segments


def _run_tracking(req: InstanceTrackingRequest, job: Job) -> dict:
    """ジョブ本体（ワーカースレッドで実行される）."""
    settings = get_settings()

    frames_by_channel: dict[str, list[Any]] = defaultdict(list)
    for frame in req.frames:
        frames_by_channel[frame.channel].append(frame)

    prompts_by_token: dict[str, list[Any]] = defaultdict(list)
    for prompt in req.prompts:
        prompts_by_token[prompt.sample_data_token].append(prompt)

    segments = _build_segments(frames_by_channel, max(1, req.sample_interval))
    total = sum(len(v) for v in segments.values())
    job.set_progress(0, total, "モデルを準備中...")

    done = 0
    total_time = 0.0
    next_track_id = 0
    all_frames: list[TrackingFrameResult] = []
    # キーフレームの結果だけを保持する（sweep は伝播にのみ使う）
    results_by_frame: dict[str, TrackingFrameResult] = {}

    with model_registry.use_gpu("sam2") as tracker:
        for channel, channel_segments in segments.items():
            # track_id はカメラを跨がない。
            # 同一物体が別カメラに写っても、2D の情報だけでは同定できないため
            previous_boundary: list[dict[str, Any]] = []

            for seg_index, seg_frames in enumerate(channel_segments):
                if job.cancel_requested():
                    break

                head = seg_frames[0]
                prompts = [p.model_dump() for p in prompts_by_token.get(
                    head.sample_data_token, []
                )]
                started = time.perf_counter()
                error: str | None = None
                per_frame: list[list[dict[str, Any]]] = []

                try:
                    if prompts:
                        per_frame = tracker.propagate(
                            [f.model_dump() for f in seg_frames],
                            prompts,
                            dataroot=settings.DATA_ROOT / req.dataroot,
                            mask_score_threshold=req.mask_score_threshold,
                            stub_delay_sec=req.stub_delay_sec,
                        )
                    else:
                        # プロンプトが無い区間（検出0件）は空で通す
                        per_frame = [[] for _ in seg_frames]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("tracking failed (%s seg %d): %s",
                                   channel, seg_index, exc)
                    error = f"{type(exc).__name__}: {exc}"
                    per_frame = [[] for _ in seg_frames]
                finally:
                    elapsed = time.perf_counter() - started
                    total_time += elapsed
                    done += 1
                    job.set_progress(
                        done,
                        message=f"{channel} / 区間 {seg_index + 1}/{len(channel_segments)}",
                    )

                # 区間の先頭（＝プロンプト sample）で track_id を引き継ぐ
                head_instances = per_frame[0] if per_frame else []
                track_ids, next_track_id = assign_track_ids(
                    previous_boundary,
                    head_instances,
                    next_track_id=next_track_id,
                    iou_threshold=req.iou_threshold,
                    iou_method=req.iou_method,
                    label_match=req.iou_label_match,
                    label_to_category_group=req.label_to_category_group,
                )
                local_to_track = {
                    inst["local_id"]: track_ids[i]
                    for i, inst in enumerate(head_instances)
                }

                for frame, instances in zip(seg_frames, per_frame):
                    resolved: list[TrackedInstance] = []
                    for inst in instances:
                        track_id = local_to_track.get(inst["local_id"])
                        if track_id is None:
                            continue
                        resolved.append(TrackedInstance(
                            track_id=track_id,
                            label=inst["label"],
                            score=inst.get("score"),
                            mask_rle=inst["mask_rle"],
                            mask_area=inst["mask_area"],
                            xmin=inst["xmin"], ymin=inst["ymin"],
                            xmax=inst["xmax"], ymax=inst["ymax"],
                            detection_2d_id=inst.get("detection_2d_id"),
                            is_prompt_frame=bool(inst.get("is_prompt_frame")),
                        ))

                    if not frame.is_key_frame:
                        # sweep は伝播にのみ使い、結果は残さない
                        continue

                    result = TrackingFrameResult(
                        sample_data_token=frame.sample_data_token,
                        sample_token=frame.sample_token,
                        sample_idx=frame.sample_idx,
                        channel=frame.channel,
                        is_key_frame=True,
                        instances=resolved,
                        inference_time=round(elapsed, 3),
                        error=error,
                    )
                    # 区間境界は次の区間のプロンプト側で上書きする
                    results_by_frame[frame.sample_data_token] = result

                # 次の区間へ渡す境界インスタンス（この区間の最後のキーフレーム）
                previous_boundary = []
                for frame, instances in reversed(list(zip(seg_frames, per_frame))):
                    if not frame.is_key_frame:
                        continue
                    for inst in instances:
                        track_id = local_to_track.get(inst["local_id"])
                        if track_id is None:
                            continue
                        previous_boundary.append({**inst, "track_id": track_id})
                    break

                # 完了した区間ぶんを UI へ流す
                for frame in seg_frames:
                    result = results_by_frame.get(frame.sample_data_token)
                    if result is not None and frame.is_key_frame:
                        job.append_partial(result.model_dump())

    all_frames = sorted(
        results_by_frame.values(), key=lambda r: (r.channel, r.sample_idx)
    )
    num_instances = sum(len(f.instances) for f in all_frames)
    track_ids_all = {
        (f.channel, i.track_id) for f in all_frames for i in f.instances
    }

    return {
        "num_frames": len(all_frames),
        "num_instances": num_instances,
        "num_tracks": len(track_ids_all),
        "inference_time": round(total_time, 3),
        "frames": [f.model_dump() for f in all_frames],
    }


@router.post("/jobs", response_model=JobResponse, status_code=202)
def create_tracking_job(
    req: InstanceTrackingRequest,
    jobs: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """Instance Tracking ジョブを登録する（即座に job_id を返す）."""
    job = jobs.submit(KIND, lambda j: _run_tracking(req, j))
    return JobResponse(**job.to_dict(since=0))


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_tracking_job(
    job_id: str,
    since: int = Query(0, ge=0, description="受け取り済みの部分結果の件数"),
    jobs: JobManager = Depends(get_job_manager),
) -> JobResponse:
    job = jobs.get(job_id)
    if job is None or job.kind != KIND:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return JobResponse(**job.to_dict(since=since))


@router.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_tracking_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager)
) -> JobResponse:
    job = jobs.get(job_id)
    if job is None or job.kind != KIND:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    jobs.cancel(job_id)
    return JobResponse(**job.to_dict())


@router.get("/jobs", response_model=list[JobResponse])
def list_tracking_jobs(
    jobs: JobManager = Depends(get_job_manager),
) -> list[JobResponse]:
    return [JobResponse(**j.to_dict()) for j in jobs.list(KIND)]


@router.get("/status")
def status() -> dict[str, str]:
    return {"kind": KIND, "implemented": "stub"}
