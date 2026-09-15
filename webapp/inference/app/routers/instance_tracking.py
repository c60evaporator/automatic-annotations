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
    ORIGIN_BACKWARD,
    ORIGIN_FORWARD,
    ORIGIN_PROMPT,
    ORIGIN_PROPAGATED,
    TRACK_ID_INHERITANCE_FB,
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

    # 関数へ切り出した区間ループと進捗・採番を共有する
    state: dict[str, Any] = {"done": 0, "total_time": 0.0, "next_track_id": 0}
    all_frames: list[TrackingFrameResult] = []
    # キーフレームの結果だけを保持する（sweep は伝播にのみ使う）。
    # キーは (sample_data_token, origin)。区間境界の sample には
    # propagated（前区間からの伝播）と prompt（今回の推論）の両方が入る
    results_by_frame: dict[tuple[str, str], TrackingFrameResult] = {}

    with model_registry.use_gpu("sam2") as tracker:
        if req.track_id_inheritance == TRACK_ID_INHERITANCE_FB:
            _run_forward_backward(
                req, job, tracker, segments, prompts_by_token,
                results_by_frame, state,
            )
        else:
            _run_continuous(
                req, job, tracker, segments, prompts_by_token,
                results_by_frame, state,
            )

    total_time = state["total_time"]

    all_frames = sorted(
        results_by_frame.values(),
        key=lambda r: (r.channel, r.sample_idx, r.origin),
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


def _run_continuous(
    req: InstanceTrackingRequest,
    job: Job,
    tracker: Any,
    segments: dict[str, list[list[Any]]],
    prompts_by_token: dict[str, list[Any]],
    results_by_frame: dict[tuple[str, str], TrackingFrameResult],
    state: dict[str, Any],
) -> None:
    """continuous_id: Forward のみ。区間境界で次のプロンプトと照合する.

    従来の挙動をそのまま維持している（回帰を避けるため中身は変更していない）。
    """
    # 呼び出し元のローカル変数は見えないので、ここで取り直す
    settings = get_settings()

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
                state["total_time"] += elapsed
                state["done"] += 1
                job.set_progress(
                    state["done"],
                    message=f"{channel} / 区間 {seg_index + 1}/{len(channel_segments)}",
                )

            # 区間の先頭（＝プロンプト sample）で track_id を引き継ぐ
            head_instances = per_frame[0] if per_frame else []
            track_ids, state["next_track_id"] = assign_track_ids(
                previous_boundary,
                head_instances,
                next_track_id=state["next_track_id"],
                iou_threshold=req.iou_threshold,
                iou_method=req.iou_method,
                label_match=req.iou_label_match,
                label_to_category_group=req.label_to_category_group,
            )
            local_to_track = {
                inst["local_id"]: track_ids[i]
                for i, inst in enumerate(head_instances)
            }

            for frame_position, (frame, instances) in enumerate(
                zip(seg_frames, per_frame)
            ):
                # 区間の先頭はプロンプト由来、それ以降は伝播由来。
                # 次の区間の先頭が同じ sample を prompt として上書きせず、
                # 別レコードとして併存する
                origin = ORIGIN_PROMPT if frame_position == 0 else ORIGIN_PROPAGATED
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
                    origin=origin,
                    sample_data_token=frame.sample_data_token,
                    sample_token=frame.sample_token,
                    sample_idx=frame.sample_idx,
                    channel=frame.channel,
                    is_key_frame=True,
                    instances=resolved,
                    inference_time=round(elapsed, 3),
                    error=error,
                )
                results_by_frame[(frame.sample_data_token, origin)] = result

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
            for frame_position, frame in enumerate(seg_frames):
                if not frame.is_key_frame:
                    continue
                origin = ORIGIN_PROMPT if frame_position == 0 else ORIGIN_PROPAGATED
                result = results_by_frame.get((frame.sample_data_token, origin))
                if result is not None:
                    job.append_partial(result.model_dump())



def _to_tracked(
    inst: dict[str, Any], track_id: str, *, is_selected: bool
) -> TrackedInstance:
    """伝播結果の dict を API の形へ直す."""
    return TrackedInstance(
        track_id=track_id,
        label=inst["label"],
        score=inst.get("score"),
        mask_rle=inst["mask_rle"],
        mask_area=inst["mask_area"],
        xmin=inst["xmin"], ymin=inst["ymin"],
        xmax=inst["xmax"], ymax=inst["ymax"],
        detection_2d_id=inst.get("detection_2d_id"),
        is_prompt_frame=bool(inst.get("is_prompt_frame")),
        is_selected=is_selected,
    )


def _tracks_by_local_id(
    per_frame: list[list[dict[str, Any]]], frame_offset: int = 0
) -> tuple[list[int], list[dict[int, dict[str, Any]]]]:
    """伝播結果を local_id ごとの ``{frame index: instance}`` へ組み替える.

    fb_matching は「トラックのリスト」を受け取るため、
    フレーム単位の結果を転置しておく。

    Returns:
        (local_id のリスト, それに対応するトラックのリスト)
    """
    tracks: dict[int, dict[int, dict[str, Any]]] = {}
    for frame_index, instances in enumerate(per_frame):
        for inst in instances:
            tracks.setdefault(inst["local_id"], {})[frame_index + frame_offset] = inst
    local_ids = sorted(tracks)
    return local_ids, [tracks[i] for i in local_ids]


def _run_forward_backward(
    req: InstanceTrackingRequest,
    job: Job,
    tracker: Any,
    segments: dict[str, list[list[Any]]],
    prompts_by_token: dict[str, list[Any]],
    results_by_frame: dict[tuple[str, str], TrackingFrameResult],
    state: dict[str, Any],
) -> None:
    """forward_backward_matching: 両方向を走らせ、区間内で照合する.

    区間 s の backward プロンプトと区間 s+1 の forward プロンプトは
    **同じアンカーの同じ Detection2D ボックス**なので、境界での照合は不要。
    照合するのは区間内の F×B だけで、その結果を連鎖させて track_id を配る。

    両方向のマスクを ``origin='forward'/'backward'`` で残し、
    採用した方に ``is_selected`` を立てる（比較表示のため）。
    """
    # 呼び出し元のローカル変数は見えないので、ここで取り直す
    settings = get_settings()

    from app.services.fb_matching import (
        assign_chain_track_ids,
        match_forward_backward,
        select_direction,
        temporal_iou_matrix,
    )

    for channel, channel_segments in segments.items():
        if not channel_segments:
            continue
        # track_id はカメラを跨がない
        anchors: list[int] = []
        prompt_counts: dict[int, int] = {}
        segment_matches: dict[tuple[int, int], dict[int, int]] = {}
        # 区間ごとの伝播結果を保持する（照合後にまとめて結果へ変換する）
        segment_data: list[dict[str, Any]] = []

        for seg_index, seg_frames in enumerate(channel_segments):
            if job.cancel_requested():
                break

            head, tail = seg_frames[0], seg_frames[-1]
            fwd_prompts = [p.model_dump() for p in prompts_by_token.get(
                head.sample_data_token, []
            )]
            bwd_prompts = [p.model_dump() for p in prompts_by_token.get(
                tail.sample_data_token, []
            )]

            started = time.perf_counter()
            error: str | None = None
            per_frame_f: list[list[dict[str, Any]]] = [[] for _ in seg_frames]
            per_frame_b: list[list[dict[str, Any]]] = [[] for _ in seg_frames]

            try:
                frame_dicts = [f.model_dump() for f in seg_frames]
                common = dict(
                    dataroot=settings.DATA_ROOT / req.dataroot,
                    mask_score_threshold=req.mask_score_threshold,
                    stub_delay_sec=req.stub_delay_sec,
                )
                if fwd_prompts:
                    per_frame_f = tracker.propagate(
                        frame_dicts, fwd_prompts, direction="forward", **common
                    )
                if bwd_prompts:
                    per_frame_b = tracker.propagate(
                        frame_dicts, bwd_prompts, direction="backward", **common
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("tracking failed (%s seg %d): %s",
                               channel, seg_index, exc)
                error = f"{type(exc).__name__}: {exc}"
            finally:
                elapsed = time.perf_counter() - started
                state["total_time"] += elapsed
                state["done"] += 1
                job.set_progress(
                    state["done"],
                    message=f"{channel} / 区間 {seg_index + 1}/{len(channel_segments)}",
                )

            # アンカーの並びとプロンプト数を記録する。
            # アンカーは「区間の先頭 index」を通し番号として使う
            if not anchors:
                anchors.append(seg_index)
                prompt_counts[seg_index] = len(fwd_prompts)
            anchors.append(seg_index + 1)
            prompt_counts[seg_index + 1] = len(bwd_prompts)

            # F×B の照合。local_id の並び順が prompt の index に対応する
            f_ids, f_tracks = _tracks_by_local_id(per_frame_f)
            b_ids, b_tracks = _tracks_by_local_id(per_frame_b)
            score = temporal_iou_matrix(
                f_tracks, b_tracks,
                iou_method=req.iou_method,
                label_match=req.iou_label_match,
                label_to_category_group=req.label_to_category_group,
            )
            matched = match_forward_backward(score, req.iou_threshold)
            # 行・列の index を prompt の index（= local_id）へ戻す
            segment_matches[(seg_index, seg_index + 1)] = {
                f_ids[i]: b_ids[j] for i, j in matched.items()
            }

            segment_data.append({
                "seg_index": seg_index,
                "frames": seg_frames,
                "forward": per_frame_f,
                "backward": per_frame_b,
                "elapsed": elapsed,
                "error": error,
            })

        # 連鎖割り当て。カメラごとに採番を続ける
        assigned, state["next_track_id"] = assign_chain_track_ids(
            anchors, prompt_counts, segment_matches,
            start_track_id=state["next_track_id"],
        )

        # 結果へ変換する
        for data in segment_data:
            seg_index = data["seg_index"]
            seg_frames = data["frames"]
            last = len(seg_frames) - 1
            # 区間は境界フレームを共有している。両方の区間が同じ
            # (sample_data_token, origin) へ書くと後から書いた方で上書きされ、
            # is_selected の判定も壊れる。末尾フレームは次の区間に任せる
            # （最終区間だけは自分で持つ）
            is_last_segment = seg_index == len(segment_data) - 1

            directions = (
                ("forward", data["forward"], ORIGIN_FORWARD, seg_index),
                ("backward", data["backward"], ORIGIN_BACKWARD, seg_index + 1),
            )

            for frame_index, frame in enumerate(seg_frames):
                if not frame.is_key_frame:
                    # sweep は伝播にのみ使い、結果は残さない
                    continue
                if frame_index == last and not is_last_segment:
                    # 境界フレームは次の区間が担当する
                    continue

                # そのフレームで、方向ごとに track_id を解決しておく
                resolved_by_direction: dict[str, dict[str, dict[str, Any]]] = {}
                for direction, per_frame, _origin, anchor in directions:
                    resolved: dict[str, dict[str, Any]] = {}
                    for inst in per_frame[frame_index]:
                        track_id = assigned.get((anchor, inst["local_id"]))
                        if track_id is not None:
                            resolved[str(track_id)] = inst
                    resolved_by_direction[direction] = resolved

                # 距離で決めるのは「両方向にあるトラック」だけ。
                # 片方向しか無いトラック（プロンプトが一方のアンカーにしか
                # 無い場合など）は、その方向を必ず採用する。
                # フレーム単位で決めると、そのトラックが全フレームで
                # 表示から消える
                preferred = select_direction(frame_index, 0, last)

                for direction, per_frame, origin, _anchor in directions:
                    resolved = resolved_by_direction[direction]
                    if not resolved:
                        continue
                    other = "backward" if direction == "forward" else "forward"
                    counterpart = resolved_by_direction[other]

                    instances = [
                        _to_tracked(
                            inst, track_id,
                            is_selected=(
                                direction == preferred
                                if track_id in counterpart else True
                            ),
                        )
                        for track_id, inst in resolved.items()
                    ]

                    result = TrackingFrameResult(
                        origin=origin,
                        sample_data_token=frame.sample_data_token,
                        sample_token=frame.sample_token,
                        sample_idx=frame.sample_idx,
                        channel=frame.channel,
                        is_key_frame=True,
                        instances=instances,
                        inference_time=round(data["elapsed"], 3),
                        error=data["error"],
                    )
                    results_by_frame[(frame.sample_data_token, origin)] = result
                    job.append_partial(result.model_dump())
