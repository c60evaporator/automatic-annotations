"""2D 検出 run の保存と読み出し（webapp 側の書き込み主体）.

推論サーバーは DB を持たないため、結果の永続化は webapp が行う。
保存はジョブ完了時に 1 トランザクションでまとめて実行する
（partial ごとに書くと、キャンセル時の後始末と SQLite の書き込みロックが面倒）。
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
from app.services.label_service import label_groups
from app.services.manual_merge import merge_manual_boxes_by_frame

logger = get_logger(__name__)

# ジョブの status → DB の run status
_JOB_STATUS_MAP = {
    "succeeded": RUN_STATUS_SUCCEEDED,
    "failed": RUN_STATUS_FAILED,
    "cancelled": RUN_STATUS_CANCELLED,
}


def save_detection_run(
    dataset_id: str,
    scene_token: str,
    *,
    job: dict[str, Any],
    boxes_by_frame: dict[str, list[dict[str, Any]]],
    sample_interval: int,
    score_threshold: dict[str, float],
    nms_same_class_ious: dict[str, float],
    nms_cross_class_iou: float,
    model_name: str,
    inherit_from_params_id: str | None = None,
) -> str:
    """完了したジョブの結果を 1 run として保存する.

    Args:
        job: 推論サーバーから受け取ったジョブ情報（status/processed 等）
        boxes_by_frame: {sample_data_token: [box, ...]}
        inherit_from_params_id: 手修正を引き継ぐ元の run。
            表示に使っている run を渡す。None なら引き継がない。

    Returns:
        作成した run の id。
    """
    settings = get_settings()
    status = _JOB_STATUS_MAP.get(job.get("status", ""), RUN_STATUS_FAILED)

    # config のラベル体系は「実行時点のスナップショット」として丸ごと保存する。
    # 後から config を変えても過去の run の解釈が変わらないようにするため
    with session_scope() as session:
        repo = Detection2DRepository(session)

        # 手修正の引き継ぎ。参照する run が消えていても落ちないようにする
        merged = boxes_by_frame
        if inherit_from_params_id:
            manual = repo.list_boxes_by_run(
                inherit_from_params_id, manual_only=True
            )
            if manual:
                merged = merge_manual_boxes_by_frame(
                    boxes_by_frame, manual, settings.DET2D_MANUAL_REPLACE_IOU
                )
                logger.info(
                    "inherited %d manual boxes from run %s",
                    sum(len(v) for v in manual.values()), inherit_from_params_id,
                )

        params_id = repo.create_run(
            dataset_id, scene_token,
            model_name=model_name,
            sample_interval=sample_interval,
            nusc_category_to_label=dict(settings.NUSC_CATEGORY_TO_LABEL),
            label_to_nusc_category=dict(settings.LABEL_TO_NUSC_CATEGORY),
            label_to_category_group=dict(settings.LABEL_TO_CATEGORY_GROUP),
            score_threshold=dict(score_threshold),
            nms_same_class_ious=dict(nms_same_class_ious),
            # 保存時は「グループ→値」の形に揃える。
            # 実行時は全グループ共通の1値だが、列は dict なので展開しておく
            nms_cross_class_ious={g: nms_cross_class_iou for g in label_groups()},
            status=status,
        )
        saved = repo.save_boxes(params_id, dataset_id, merged)
        repo.finish_run(
            params_id,
            status=status,
            num_inferences=int(job.get("processed", 0)),
            inference_time=(job.get("result") or {}).get("inference_time"),
        )

        # 上限を超えた古い run を削除（参照されているものは残る）
        pruned = repo.prune_runs(
            dataset_id, scene_token, keep=settings.DET2D_MAX_RUNS_PER_SCENE
        )

    logger.info(
        "saved detection run %s: %d boxes, status=%s, pruned=%d",
        params_id, saved, status, len(pruned),
    )
    return params_id


def resolve_display_run(dataset_id: str, scene_token: str) -> tuple[str | None, str]:
    """初期表示に使う run を決める（優先順は Repository 側に記述）."""
    with read_only_session() as session:
        return Detection2DRepository(session).resolve_display_run(
            dataset_id, scene_token
        )


def load_run_boxes(params_id: str) -> dict[str, list[dict[str, Any]]]:
    with read_only_session() as session:
        return Detection2DRepository(session).list_boxes_by_run(params_id)


def list_runs(dataset_id: str, scene_token: str) -> list[dict[str, Any]]:
    with read_only_session() as session:
        return Detection2DRepository(session).list_runs(dataset_id, scene_token)


def get_run(params_id: str) -> dict[str, Any] | None:
    with read_only_session() as session:
        return Detection2DRepository(session).get_run(params_id)


def delete_run(params_id: str) -> None:
    with session_scope() as session:
        Detection2DRepository(session).delete_run(params_id)
