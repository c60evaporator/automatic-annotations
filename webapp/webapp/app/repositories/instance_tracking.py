"""Instance Tracking の実行単位（run）と結果のクエリ.

run = InstanceTracking2DParams の 1 行。結果の InstanceTracking2D 行が紐づく。
保持ポリシーは Detection2D と同じ「追加（履歴保持）＋ 上限プルーニング」。

上書きにしない理由も同じで、DepthEstimationParams が
instance_tracking_2d_params_id を ON DELETE CASCADE で参照しているため、
run を消すと 3D ボックスまで連鎖削除される。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, desc, func, insert, select, update
from sqlalchemy.orm import Session

from app.models.ann_intermediate import (
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    DepthEstimationParams,
    Detection2DParams,
    InstanceTracking2D,
    InstanceTracking2DParams,
)

CHUNK_SIZE = 1_000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InstanceTrackingRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ── run の作成・更新 ──────────────────────────────────────────────────

    def create_run(
        self,
        dataset_id: str,
        scene_token: str,
        *,
        detection_2d_params_id: str,
        model_name: str,
        sample_interval: int,
        num_sweeps: int,
        mask_score_threshold: float,
        new_track_iou_threshold: float,
        iou_method: str,
        iou_label_match: str,
        max_lost_frames: int = 0,
        status: str = RUN_STATUS_RUNNING,
    ) -> str:
        """run を作成して id を返す.

        sample_interval は入力にした Detection2D run から引き継ぐ。
        トラッキングは検出のプロンプト間隔でしか区切れないため、
        独立に持たせると食い違う。
        """
        params_id = str(uuid.uuid4())
        self.session.execute(insert(InstanceTracking2DParams.__table__), [{
            "id": params_id,
            "dataset_id": dataset_id,
            "scene_token": scene_token,
            "detection_2d_params_id": detection_2d_params_id,
            "model_name": model_name,
            "sample_interval": sample_interval,
            "num_sweeps": num_sweeps,
            "mask_score_threshold": mask_score_threshold,
            "new_track_iou_threshold": new_track_iou_threshold,
            "iou_method": iou_method,
            "iou_label_match": iou_label_match,
            "max_lost_frames": max_lost_frames,
            "status": status,
            "num_inferences": 0,
            "num_tracks": 0,
            "started_at": _utcnow(),
        }])
        return params_id

    def finish_run(
        self,
        params_id: str,
        *,
        status: str,
        num_inferences: int = 0,
        num_tracks: int = 0,
        inference_time: float | None = None,
    ) -> None:
        self.session.execute(
            update(InstanceTracking2DParams.__table__)
            .where(InstanceTracking2DParams.__table__.c.id == params_id)
            .values(
                status=status,
                num_inferences=num_inferences,
                num_tracks=num_tracks,
                inference_time=inference_time,
                ended_at=_utcnow(),
            )
        )

    # ── 結果の保存 ────────────────────────────────────────────────────────

    def save_instances(
        self,
        params_id: str,
        dataset_id: str,
        instances_by_frame: dict[str, list[dict[str, Any]]],
    ) -> int:
        """トラッキング結果をまとめて保存する.

        Args:
            instances_by_frame: {sample_data_token: [instance, ...]}

        NOTE: mask_rle は JSON 列。非圧縮 RLE は 1 マスクで数千要素になるため、
        チャンクを Detection2D より小さめにしてメモリのピークを抑える。
        """
        rows: list[dict[str, Any]] = []
        for sample_data_token, instances in instances_by_frame.items():
            for inst in instances:
                rows.append({
                    "id": str(uuid.uuid4()),
                    "dataset_id": dataset_id,
                    "sample_data_token": sample_data_token,
                    "instance_tracking_2d_params_id": params_id,
                    "detection_2d_id": inst.get("detection_2d_id"),
                    "track_id": str(inst["track_id"]),
                    "label": inst["label"],
                    "mask_rle": inst["mask_rle"],
                    "mask_area": int(inst.get("mask_area", 0)),
                    "xmin": int(inst["xmin"]), "ymin": int(inst["ymin"]),
                    "xmax": int(inst["xmax"]), "ymax": int(inst["ymax"]),
                    "score": inst.get("score"),
                    "instance_token": None,
                })

        if not rows:
            return 0

        stmt = insert(InstanceTracking2D.__table__)
        for i in range(0, len(rows), CHUNK_SIZE):
            self.session.execute(stmt, rows[i:i + CHUNK_SIZE])
        return len(rows)

    # ── run の参照 ────────────────────────────────────────────────────────

    def get_run(self, params_id: str) -> dict[str, Any] | None:
        row = self.session.scalars(
            select(InstanceTracking2DParams)
            .where(InstanceTracking2DParams.id == params_id)
        ).first()
        return _run_to_dict(row) if row else None

    def list_runs(
        self, dataset_id: str, scene_token: str, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        """シーンの run 一覧を新しい順で返す（UI の run セレクタ用）.

        後段（Depth Estimation）から参照されている数も返す。
        参照されている run を削除すると 3D ボックスまで消えるため、
        UI 側で警告できるようにしておく。
        """
        nbr_instances = (
            select(func.count())
            .select_from(InstanceTracking2D)
            .where(
                InstanceTracking2D.instance_tracking_2d_params_id
                == InstanceTracking2DParams.id
            )
            .scalar_subquery()
        )
        nbr_depth = (
            select(func.count())
            .select_from(DepthEstimationParams)
            .where(
                DepthEstimationParams.instance_tracking_2d_params_id
                == InstanceTracking2DParams.id
            )
            .scalar_subquery()
        )
        stmt = (
            select(
                InstanceTracking2DParams,
                nbr_instances.label("nbr_instances"),
                nbr_depth.label("nbr_depth_runs"),
            )
            .where(
                InstanceTracking2DParams.dataset_id == dataset_id,
                InstanceTracking2DParams.scene_token == scene_token,
            )
            .order_by(desc(InstanceTracking2DParams.started_at))
        )
        if status is not None:
            stmt = stmt.where(InstanceTracking2DParams.status == status)

        out: list[dict[str, Any]] = []
        for row, nbr, refs in self.session.execute(stmt):
            data = _run_to_dict(row)
            data["nbr_instances"] = nbr
            data["nbr_depth_runs"] = refs
            out.append(data)
        return out

    def resolve_display_run(
        self, dataset_id: str, scene_token: str
    ) -> tuple[str | None, str]:
        """初期表示に使う run を決める.

        優先順は Detection2D と同じ考え方:
          1. 最新の成功した Depth Estimation が参照しているトラッキング run
          2. 最新の成功したトラッキング run
          3. なし
        """
        referenced = self.session.scalar(
            select(DepthEstimationParams.instance_tracking_2d_params_id)
            .where(
                DepthEstimationParams.scene_token == scene_token,
                DepthEstimationParams.status == RUN_STATUS_SUCCEEDED,
            )
            .order_by(desc(DepthEstimationParams.started_at))
            .limit(1)
        )
        if referenced is not None:
            exists = self.session.scalar(
                select(InstanceTracking2DParams.id)
                .where(InstanceTracking2DParams.id == referenced)
            )
            if exists is not None:
                return referenced, "depth"

        latest = self.session.scalar(
            select(InstanceTracking2DParams.id)
            .where(
                InstanceTracking2DParams.dataset_id == dataset_id,
                InstanceTracking2DParams.scene_token == scene_token,
                InstanceTracking2DParams.status == RUN_STATUS_SUCCEEDED,
            )
            .order_by(desc(InstanceTracking2DParams.started_at))
            .limit(1)
        )
        if latest is not None:
            return latest, "latest"

        return None, "none"

    def list_detection_runs_for_prompt(
        self, dataset_id: str, scene_token: str
    ) -> list[dict[str, Any]]:
        """プロンプトに使える Detection2D run の一覧（成功したもののみ）.

        UI の Box Prompt 選択に使う。ボックス数も返して、
        「検出0件の run を選んでしまう」ことを避けられるようにする。
        """
        from app.models.ann_intermediate import Detection2D

        nbr_boxes = (
            select(func.count())
            .select_from(Detection2D)
            .where(Detection2D.detection_2d_params_id == Detection2DParams.id)
            .scalar_subquery()
        )
        stmt = (
            select(Detection2DParams, nbr_boxes.label("nbr_boxes"))
            .where(
                Detection2DParams.dataset_id == dataset_id,
                Detection2DParams.scene_token == scene_token,
                Detection2DParams.status == RUN_STATUS_SUCCEEDED,
            )
            .order_by(desc(Detection2DParams.started_at))
        )
        out: list[dict[str, Any]] = []
        for row, nbr in self.session.execute(stmt):
            out.append({
                "id": row.id,
                "sample_interval": row.sample_interval,
                "model_name": row.model_name,
                "score_threshold": row.score_threshold,
                "started_at": row.started_at,
                "nbr_boxes": nbr,
            })
        return out

    # ── 結果の参照 ────────────────────────────────────────────────────────

    def list_instances_by_run(
        self, params_id: str, *, include_mask: bool = True
    ) -> dict[str, list[dict[str, Any]]]:
        """run の結果を {sample_data_token: [instance, ...]} で返す.

        Args:
            include_mask: False ならマスクを読まない。
                外接矩形だけで足りる表示のとき、JSON のデコードを丸ごと省ける
                （マスクは 1 件で数千要素になるため効く）
        """
        columns = [
            InstanceTracking2D.id,
            InstanceTracking2D.sample_data_token,
            InstanceTracking2D.track_id,
            InstanceTracking2D.label,
            InstanceTracking2D.score,
            InstanceTracking2D.mask_area,
            InstanceTracking2D.xmin, InstanceTracking2D.ymin,
            InstanceTracking2D.xmax, InstanceTracking2D.ymax,
            InstanceTracking2D.detection_2d_id,
            InstanceTracking2D.manually_modified,
        ]
        if include_mask:
            columns.append(InstanceTracking2D.mask_rle)

        stmt = select(*columns).where(
            InstanceTracking2D.instance_tracking_2d_params_id == params_id
        )

        result: dict[str, list[dict[str, Any]]] = {}
        for r in self.session.execute(stmt).mappings():
            item = {
                "id": r["id"],
                "track_id": r["track_id"],
                "label": r["label"],
                "score": r["score"],
                "mask_area": r["mask_area"],
                "xmin": r["xmin"], "ymin": r["ymin"],
                "xmax": r["xmax"], "ymax": r["ymax"],
                "detection_2d_id": r["detection_2d_id"],
                "manually_modified": r["manually_modified"],
            }
            if include_mask:
                item["mask_rle"] = r["mask_rle"]
            result.setdefault(r["sample_data_token"], []).append(item)
        return result

    def list_track_ids(self, params_id: str) -> list[dict[str, Any]]:
        """run に含まれる track の一覧（凡例・色分け用）.

        track_id ごとのラベルと出現フレーム数を返す。
        """
        stmt = (
            select(
                InstanceTracking2D.track_id,
                InstanceTracking2D.label,
                func.count().label("nbr_frames"),
            )
            .where(InstanceTracking2D.instance_tracking_2d_params_id == params_id)
            .group_by(InstanceTracking2D.track_id, InstanceTracking2D.label)
            .order_by(InstanceTracking2D.track_id)
        )
        return [dict(r) for r in self.session.execute(stmt).mappings()]

    # ── 削除・プルーニング ────────────────────────────────────────────────

    def delete_run(self, params_id: str) -> None:
        self.session.execute(
            delete(InstanceTracking2DParams.__table__)
            .where(InstanceTracking2DParams.__table__.c.id == params_id)
        )

    def prune_runs(
        self, dataset_id: str, scene_token: str, *, keep: int
    ) -> list[str]:
        """古い run を削除して保持数を上限以内に収める.

        Depth Estimation から参照されている run と実行中の run は残す。
        """
        runs = self.list_runs(dataset_id, scene_token)
        deletable = [
            r for r in runs
            if r["nbr_depth_runs"] == 0 and r["status"] != RUN_STATUS_RUNNING
        ]
        protected_ids = {r["id"] for r in runs[:keep]}
        targets = [r["id"] for r in deletable if r["id"] not in protected_ids]
        for params_id in targets:
            self.delete_run(params_id)
        return targets


def _run_to_dict(row: InstanceTracking2DParams) -> dict[str, Any]:
    return {
        "id": row.id,
        "dataset_id": row.dataset_id,
        "scene_token": row.scene_token,
        "detection_2d_params_id": row.detection_2d_params_id,
        "model_name": row.model_name,
        "sample_interval": row.sample_interval,
        "num_sweeps": row.num_sweeps,
        "mask_score_threshold": row.mask_score_threshold,
        "new_track_iou_threshold": row.new_track_iou_threshold,
        "iou_method": row.iou_method,
        "iou_label_match": row.iou_label_match,
        "max_lost_frames": row.max_lost_frames,
        "status": row.status,
        "num_inferences": row.num_inferences,
        "num_tracks": row.num_tracks,
        "inference_time": row.inference_time,
        "started_at": row.started_at,
        "ended_at": row.ended_at,
    }
