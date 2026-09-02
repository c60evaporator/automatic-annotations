"""Instance Tracking run の保存と読み出し（webapp 側の書き込み主体）.

Detection2D と同じ形に揃えてある:
  - ジョブ完了時に 1 トランザクションでまとめて保存
  - 追加（履歴保持）＋ 上限プルーニング
  - キャンセル・失敗も status 付きで残す
"""
from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import read_only_session, session_scope
from app.models.ann_intermediate import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_SUCCEEDED,
)
from app.repositories.detection2d import Detection2DRepository
from app.repositories.instance_tracking import InstanceTrackingRepository
from app.repositories.sensor import SensorRepository
from app.services.sweep_service import select_tracking_frames

logger = get_logger(__name__)

_JOB_STATUS_MAP = {
    "succeeded": RUN_STATUS_SUCCEEDED,
    "failed": RUN_STATUS_FAILED,
    "cancelled": RUN_STATUS_CANCELLED,
}


# ── リクエストの組み立て ──────────────────────────────────────────────────────

def build_tracking_payload(
    dataset_id: str,
    scene_token: str,
    dataroot: str,
    *,
    detection_run_id: str,
    num_sweeps: int,
    iou_threshold: float,
    iou_method: str,
    iou_label_match: str,
    mask_score_threshold: float = 0.5,
    stub_delay_sec: float | None = None,
) -> dict[str, Any]:
    """推論サーバーへ送るリクエストを組み立てる.

    プロンプトは選択した Detection2D run のボックスをそのまま使う。
    その run は自身の sample_interval で実行されているので、
    ボックスが存在するキーフレーム＝プロンプトを与える sample になる。
    """
    settings = get_settings()

    with read_only_session() as session:
        det_repo = Detection2DRepository(session)
        det_run = det_repo.get_run(detection_run_id)
        if det_run is None:
            raise ValueError(f"Detection2D run が見つかりません: {detection_run_id}")
        boxes_by_frame = det_repo.list_boxes_by_run(detection_run_id)

        # sweep を含む全フレームを取り、そこから使うものを選ぶ
        all_frames = SensorRepository(session).list_frames_by_scene(
            scene_token, keyframe_only=False
        )

    frames = select_tracking_frames(all_frames, num_sweeps)

    prompts = [
        {
            "sample_data_token": sample_data_token,
            "xmin": box["xmin"], "ymin": box["ymin"],
            "xmax": box["xmax"], "ymax": box["ymax"],
            "label": box["label"],
            "score": box.get("score"),
            "detection_2d_id": box.get("id"),
        }
        for sample_data_token, boxes in boxes_by_frame.items()
        for box in boxes
    ]

    return {
        "dataroot": dataroot,
        "frames": [
            {
                "sample_data_token": f["token"],
                "sample_token": f["sample_token"],
                "filename": f["filename"],
                "channel": f["channel"],
                "sample_idx": f.get("sample_idx", 0),
                "is_key_frame": f["is_key_frame"],
                "timestamp": f["timestamp"],
                "width": f["width"],
                "height": f["height"],
            }
            for f in frames
        ],
        "prompts": prompts,
        "sample_interval": det_run["sample_interval"],
        "num_sweeps": num_sweeps,
        "iou_threshold": iou_threshold,
        "iou_method": iou_method,
        "iou_label_match": iou_label_match,
        # ラベル体系は webapp 側の設定なので、解決済みのものを渡す
        "label_to_category_group": dict(settings.LABEL_TO_CATEGORY_GROUP),
        "mask_score_threshold": mask_score_threshold,
        "stub_delay_sec": stub_delay_sec,
    }


# ── 保存 ──────────────────────────────────────────────────────────────────────

def save_tracking_run(
    dataset_id: str,
    scene_token: str,
    *,
    job: dict[str, Any],
    instances_by_frame: dict[str, list[dict[str, Any]]],
    detection_run_id: str,
    sample_interval: int,
    num_sweeps: int,
    iou_threshold: float,
    iou_method: str,
    iou_label_match: str,
    mask_score_threshold: float = 0.5,
    model_name: str = "",
) -> str:
    """完了したジョブの結果を 1 run として保存する."""
    settings = get_settings()
    status = _JOB_STATUS_MAP.get(job.get("status", ""), RUN_STATUS_FAILED)

    num_tracks = len({
        (token, inst["track_id"])
        for token, insts in instances_by_frame.items()
        for inst in insts
    })

    with session_scope() as session:
        repo = InstanceTrackingRepository(session)
        params_id = repo.create_run(
            dataset_id, scene_token,
            detection_2d_params_id=detection_run_id,
            model_name=model_name or settings.TRACKING_MODEL_NAME,
            sample_interval=sample_interval,
            num_sweeps=num_sweeps,
            mask_score_threshold=mask_score_threshold,
            new_track_iou_threshold=iou_threshold,
            iou_method=iou_method,
            iou_label_match=iou_label_match,
            status=status,
        )
        saved = repo.save_instances(params_id, dataset_id, instances_by_frame)
        repo.finish_run(
            params_id,
            status=status,
            num_inferences=int(job.get("processed", 0)),
            num_tracks=(job.get("result") or {}).get("num_tracks", num_tracks),
            inference_time=(job.get("result") or {}).get("inference_time"),
        )
        pruned = repo.prune_runs(
            dataset_id, scene_token, keep=settings.TRACKING_MAX_RUNS_PER_SCENE
        )

    logger.info(
        "saved tracking run %s: %d instances, status=%s, pruned=%d",
        params_id, saved, status, len(pruned),
    )
    return params_id


# ── 読み出し ──────────────────────────────────────────────────────────────────

def resolve_display_run(dataset_id: str, scene_token: str) -> tuple[str | None, str]:
    with read_only_session() as session:
        return InstanceTrackingRepository(session).resolve_display_run(
            dataset_id, scene_token
        )


def load_run_instances(
    params_id: str, *, include_mask: bool = True
) -> dict[str, list[dict[str, Any]]]:
    with read_only_session() as session:
        return InstanceTrackingRepository(session).list_instances_by_run(
            params_id, include_mask=include_mask
        )


def list_runs(dataset_id: str, scene_token: str) -> list[dict[str, Any]]:
    with read_only_session() as session:
        return InstanceTrackingRepository(session).list_runs(dataset_id, scene_token)


def get_run(params_id: str) -> dict[str, Any] | None:
    with read_only_session() as session:
        return InstanceTrackingRepository(session).get_run(params_id)


def list_track_ids(params_id: str) -> list[dict[str, Any]]:
    with read_only_session() as session:
        return InstanceTrackingRepository(session).list_track_ids(params_id)


def list_prompt_runs(dataset_id: str, scene_token: str) -> list[dict[str, Any]]:
    """Box Prompt に選べる Detection2D run の一覧."""
    with read_only_session() as session:
        return InstanceTrackingRepository(session).list_detection_runs_for_prompt(
            dataset_id, scene_token
        )


def delete_run(params_id: str) -> None:
    with session_scope() as session:
        InstanceTrackingRepository(session).delete_run(params_id)
