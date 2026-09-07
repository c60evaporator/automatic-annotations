"""深度推定＋Box Fitting の実行単位（run）と結果のクエリ.

run = DepthEstimationParams の 1 行。1 つの run が 3 種類の成果物を持つ:

  DepthEstimation  … カメラフレームごと（深度マップのパス）
  LidarPointcloud  … sample ごと（統合 LiDAR のパスと地面マスク）
  BoxFitting3D     … インスタンスごと（点群・3D ボックス・来歴）

最終成果物の 3D ボックス自体は SampleAnnotation（source='auto'）に入り、
BoxFitting3D からは sample_annotation_token で辿れる。

NOTE: run の削除は DB 行しか消さない。深度マップと LiDAR の .npz は
DERIVED_ROOT に残るため、サービス層でディレクトリごと削除すること。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, desc, func, insert, select, update
from sqlalchemy.orm import Session

from app.models.ann_intermediate import (
    BOXFIT_STATUS_FITTED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    BoxFitting3D,
    DepthEstimation,
    DepthEstimationParams,
    InstanceTracking2DParams,
    LidarPointcloud,
)
from app.models.annotation import SampleAnnotation

# 点群 JSON を含む行は重いので、挿入チャンクは小さめにする
CHUNK_SIZE = 500


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DepthBoxFittingRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ── run の作成・更新 ──────────────────────────────────────────────────

    def create_run(
        self,
        dataset_id: str,
        scene_token: str,
        *,
        instance_tracking_2d_params_id: str,
        model_name: str,
        use_lidar: bool,
        num_lidar_sweeps: int,
        mask_params: dict[str, Any],
        depth_params: dict[str, Any],
        lidar_params: dict[str, Any],
        box_fitting_params: dict[str, Any] | None = None,
        sample_interval: int = 1,
        status: str = RUN_STATUS_RUNNING,
    ) -> str:
        """run を作成して id を返す.

        sample_interval は通常 1（全 sample を処理する）。
        プロンプト間隔はトラッキング run 側を辿れば分かる。
        """
        params_id = str(uuid.uuid4())
        self.session.execute(insert(DepthEstimationParams.__table__), [{
            "id": params_id,
            "dataset_id": dataset_id,
            "scene_token": scene_token,
            "instance_tracking_2d_params_id": instance_tracking_2d_params_id,
            "model_name": model_name,
            "sample_interval": sample_interval,
            "use_lidar": use_lidar,
            "num_lidar_sweeps": num_lidar_sweeps,
            "mask_params": mask_params,
            "depth_params": depth_params,
            "lidar_params": lidar_params,
            "box_fitting_params": box_fitting_params or {},
            "status": status,
            "num_inferences": 0,
            "num_boxes": 0,
            "started_at": _utcnow(),
        }])
        return params_id

    def finish_run(
        self,
        params_id: str,
        *,
        status: str,
        num_inferences: int = 0,
        num_boxes: int = 0,
        inference_time: float | None = None,
    ) -> None:
        self.session.execute(
            update(DepthEstimationParams.__table__)
            .where(DepthEstimationParams.__table__.c.id == params_id)
            .values(
                status=status,
                num_inferences=num_inferences,
                num_boxes=num_boxes,
                inference_time=inference_time,
                ended_at=_utcnow(),
            )
        )

    # ── 結果の保存 ────────────────────────────────────────────────────────

    def _bulk_insert(self, table: Any, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        stmt = insert(table)
        for i in range(0, len(rows), CHUNK_SIZE):
            self.session.execute(stmt, rows[i:i + CHUNK_SIZE])
        return len(rows)

    def save_depth_estimations(
        self, params_id: str, dataset_id: str, items: list[dict[str, Any]]
    ) -> int:
        """深度推定の結果（カメラフレームごと）を保存する."""
        return self._bulk_insert(DepthEstimation.__table__, [{
            "id": str(uuid.uuid4()),
            "dataset_id": dataset_id,
            "sample_data_token": item["sample_data_token"],
            "depth_estimation_params_id": params_id,
            "depth_path": item["depth_path"],
            "depth_width": item.get("depth_width"),
            "depth_height": item.get("depth_height"),
            "scale": item.get("scale"),
            "shift": item.get("shift"),
            "min_depth": item.get("min_depth"),
            "max_depth": item.get("max_depth"),
            "num_points": int(item.get("num_points", 0)),
        } for item in items])

    def save_lidar_pointclouds(
        self, params_id: str, dataset_id: str, items: list[dict[str, Any]]
    ) -> int:
        """LiDAR 統合点群（sample ごと）を保存する."""
        return self._bulk_insert(LidarPointcloud.__table__, [{
            "id": str(uuid.uuid4()),
            "dataset_id": dataset_id,
            "sample_token": item["sample_token"],
            "sample_data_token": item["sample_data_token"],
            "depth_estimation_params_id": params_id,
            "pointcloud_path": item["pointcloud_path"],
            "coordinate_frame": item.get("coordinate_frame", "ego"),
            "num_sweeps": int(item.get("num_sweeps", 1)),
            "num_points": int(item.get("num_points", 0)),
            "num_ground_points": int(item.get("num_ground_points", 0)),
        } for item in items])

    def save_box_fittings(
        self, params_id: str, dataset_id: str, items: list[dict[str, Any]]
    ) -> int:
        """Box Fitting の結果（インスタンスごと）を保存する.

        ボックスを作れなかったインスタンスも status 付きで保存する。
        行が無いと「なぜ 3D ボックスが無いのか」を UI で説明できない。
        """
        return self._bulk_insert(BoxFitting3D.__table__, [{
            "id": str(uuid.uuid4()),
            "dataset_id": dataset_id,
            "depth_estimation_params_id": params_id,
            "instance_tracking_2d_id": item["instance_tracking_2d_id"],
            "sample_data_token": item["sample_data_token"],
            "sample_annotation_token": None,
            "track_id": str(item["track_id"]),
            "label": item["label"],
            "mask_rle_closed": item.get("mask_rle_closed"),
            "points_depth_ego": item.get("points_depth_ego"),
            "points_lidar_ego": item.get("points_lidar_ego"),
            "num_points_depth": int(item.get("num_points_depth", 0)),
            "num_points_lidar": int(item.get("num_points_lidar", 0)),
            "depth_align_scale": item.get("depth_align_scale"),
            "depth_align_shift": item.get("depth_align_shift"),
            "center_ego": item.get("center_ego"),
            "size_wlh": item.get("size_wlh"),
            "yaw_ego": item.get("yaw_ego"),
            "fitting_score": item.get("fitting_score"),
            "status": item.get("status", BOXFIT_STATUS_FITTED),
        } for item in items])

    def link_annotation(self, box_fitting_id: str, annotation_token: str) -> None:
        """生成した SampleAnnotation を Box Fitting 行に結び付ける."""
        self.session.execute(
            update(BoxFitting3D.__table__)
            .where(BoxFitting3D.__table__.c.id == box_fitting_id)
            .values(sample_annotation_token=annotation_token)
        )

    # ── run の参照 ────────────────────────────────────────────────────────

    def get_run(self, params_id: str) -> dict[str, Any] | None:
        row = self.session.scalars(
            select(DepthEstimationParams)
            .where(DepthEstimationParams.id == params_id)
        ).first()
        return _run_to_dict(row) if row else None

    def list_runs(
        self, dataset_id: str, scene_token: str, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        """シーンの run 一覧を新しい順で返す（UI の run セレクタ用）."""
        nbr_boxes = (
            select(func.count()).select_from(BoxFitting3D)
            .where(BoxFitting3D.depth_estimation_params_id == DepthEstimationParams.id)
            .scalar_subquery()
        )
        nbr_fitted = (
            select(func.count()).select_from(BoxFitting3D)
            .where(
                BoxFitting3D.depth_estimation_params_id == DepthEstimationParams.id,
                BoxFitting3D.status == BOXFIT_STATUS_FITTED,
            )
            .scalar_subquery()
        )
        # この run が生成した 3D アノテーション。削除すると一緒に消えるため
        # UI で件数を見せて確認を取れるようにする
        nbr_annotations = (
            select(func.count()).select_from(SampleAnnotation)
            .where(
                SampleAnnotation.depth_estimation_params_id
                == DepthEstimationParams.id
            )
            .scalar_subquery()
        )
        stmt = (
            select(
                DepthEstimationParams,
                nbr_boxes.label("nbr_box_fittings"),
                nbr_fitted.label("nbr_fitted"),
                nbr_annotations.label("nbr_annotations"),
            )
            .where(
                DepthEstimationParams.dataset_id == dataset_id,
                DepthEstimationParams.scene_token == scene_token,
            )
            .order_by(desc(DepthEstimationParams.started_at))
        )
        if status is not None:
            stmt = stmt.where(DepthEstimationParams.status == status)

        out: list[dict[str, Any]] = []
        for row, nbr, fitted, anns in self.session.execute(stmt):
            data = _run_to_dict(row)
            data["nbr_box_fittings"] = nbr
            data["nbr_fitted"] = fitted
            data["nbr_annotations"] = anns
            out.append(data)
        return out

    def resolve_display_run(
        self, dataset_id: str, scene_token: str
    ) -> tuple[str | None, str]:
        """初期表示に使う run を決める.

        Box Fitting は最終ステップなので、後段から参照される run は無い。
        「最新の成功した run」だけを見る。
        """
        latest = self.session.scalar(
            select(DepthEstimationParams.id)
            .where(
                DepthEstimationParams.dataset_id == dataset_id,
                DepthEstimationParams.scene_token == scene_token,
                DepthEstimationParams.status == RUN_STATUS_SUCCEEDED,
            )
            .order_by(desc(DepthEstimationParams.started_at))
            .limit(1)
        )
        return (latest, "latest") if latest else (None, "none")

    def list_tracking_runs_for_input(
        self, dataset_id: str, scene_token: str
    ) -> list[dict[str, Any]]:
        """入力に選べる Instance Tracking run の一覧（成功したもののみ）.

        UI の Instance Tracking セレクタに使う。
        """
        from app.models.ann_intermediate import InstanceTracking2D

        nbr_instances = (
            select(func.count()).select_from(InstanceTracking2D)
            .where(
                InstanceTracking2D.instance_tracking_2d_params_id
                == InstanceTracking2DParams.id
            )
            .scalar_subquery()
        )
        stmt = (
            select(InstanceTracking2DParams, nbr_instances.label("nbr_instances"))
            .where(
                InstanceTracking2DParams.dataset_id == dataset_id,
                InstanceTracking2DParams.scene_token == scene_token,
                InstanceTracking2DParams.status == RUN_STATUS_SUCCEEDED,
            )
            .order_by(desc(InstanceTracking2DParams.started_at))
        )
        return [{
            "id": row.id,
            "model_name": row.model_name,
            "sample_interval": row.sample_interval,
            "num_sweeps": row.num_sweeps,
            "num_tracks": row.num_tracks,
            "started_at": row.started_at,
            "nbr_instances": nbr,
        } for row, nbr in self.session.execute(stmt)]

    # ── 結果の参照 ────────────────────────────────────────────────────────

    def list_depth_estimations_by_run(
        self, params_id: str
    ) -> dict[str, dict[str, Any]]:
        """{sample_data_token: 深度推定の情報} を返す."""
        stmt = select(
            DepthEstimation.sample_data_token, DepthEstimation.depth_path,
            DepthEstimation.depth_width, DepthEstimation.depth_height,
            DepthEstimation.scale, DepthEstimation.shift,
            DepthEstimation.min_depth, DepthEstimation.max_depth,
            DepthEstimation.num_points,
        ).where(DepthEstimation.depth_estimation_params_id == params_id)
        return {
            r["sample_data_token"]: dict(r)
            for r in self.session.execute(stmt).mappings()
        }

    def list_lidar_pointclouds_by_run(
        self, params_id: str
    ) -> dict[str, dict[str, Any]]:
        """{sample_token: LiDAR 統合点群の情報} を返す."""
        stmt = select(
            LidarPointcloud.sample_token, LidarPointcloud.sample_data_token,
            LidarPointcloud.pointcloud_path, LidarPointcloud.coordinate_frame,
            LidarPointcloud.num_sweeps, LidarPointcloud.num_points,
            LidarPointcloud.num_ground_points,
        ).where(LidarPointcloud.depth_estimation_params_id == params_id)
        return {
            r["sample_token"]: dict(r)
            for r in self.session.execute(stmt).mappings()
        }

    def list_box_fittings_by_run(
        self,
        params_id: str,
        *,
        sample_data_tokens: list[str] | None = None,
        include_points: bool = False,
        include_mask: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """{sample_data_token: [Box Fitting 結果, ...]} を返す.

        Args:
            include_points: 点群 JSON を読むか。
                1 run で数十 MB になるため、点群ビュー以外では False にする
            include_mask: クロージング後マスクを読むか

        点群とマスクを既定で読まないのは、一覧表示や 3D ボックスの
        重ね描きだけなら不要で、読むと JSON パースが支配的になるため。
        """
        columns = [
            BoxFitting3D.id, BoxFitting3D.sample_data_token,
            BoxFitting3D.instance_tracking_2d_id,
            BoxFitting3D.sample_annotation_token,
            BoxFitting3D.track_id, BoxFitting3D.label, BoxFitting3D.status,
            BoxFitting3D.num_points_depth, BoxFitting3D.num_points_lidar,
            BoxFitting3D.depth_align_scale, BoxFitting3D.depth_align_shift,
            BoxFitting3D.center_ego, BoxFitting3D.size_wlh, BoxFitting3D.yaw_ego,
            BoxFitting3D.fitting_score, BoxFitting3D.manually_modified,
        ]
        if include_points:
            columns += [BoxFitting3D.points_depth_ego, BoxFitting3D.points_lidar_ego]
        if include_mask:
            columns.append(BoxFitting3D.mask_rle_closed)

        stmt = select(*columns).where(
            BoxFitting3D.depth_estimation_params_id == params_id
        )
        if sample_data_tokens is not None:
            stmt = stmt.where(
                BoxFitting3D.sample_data_token.in_(sample_data_tokens)
            )

        result: dict[str, list[dict[str, Any]]] = {}
        for r in self.session.execute(stmt).mappings():
            item = dict(r)
            result.setdefault(item["sample_data_token"], []).append(item)
        return result

    def list_unlinked_fittings(self, params_id: str) -> list[dict[str, Any]]:
        """まだ SampleAnnotation を作っていない成功インスタンスを返す."""
        stmt = select(
            BoxFitting3D.id, BoxFitting3D.sample_data_token,
            BoxFitting3D.track_id, BoxFitting3D.label,
            BoxFitting3D.center_ego, BoxFitting3D.size_wlh,
            BoxFitting3D.yaw_ego, BoxFitting3D.fitting_score,
            BoxFitting3D.num_points_depth, BoxFitting3D.num_points_lidar,
        ).where(
            BoxFitting3D.depth_estimation_params_id == params_id,
            BoxFitting3D.status == BOXFIT_STATUS_FITTED,
            BoxFitting3D.sample_annotation_token.is_(None),
        )
        return [dict(r) for r in self.session.execute(stmt).mappings()]

    # ── 削除・プルーニング ────────────────────────────────────────────────

    def delete_run(self, params_id: str) -> None:
        """run を削除する（DB 行のみ）.

        DepthEstimation / LidarPointcloud / BoxFitting3D と、
        この run が生成した SampleAnnotation は CASCADE で消える。
        DERIVED_ROOT のファイルはサービス層で削除すること。
        """
        self.session.execute(
            delete(DepthEstimationParams.__table__)
            .where(DepthEstimationParams.__table__.c.id == params_id)
        )

    def prune_runs(
        self, dataset_id: str, scene_token: str, *, keep: int
    ) -> list[str]:
        """古い run を削除して保持数を上限以内に収める.

        Box Fitting は最終ステップなので参照による保護は不要だが、
        実行中の run は残す。
        """
        runs = self.list_runs(dataset_id, scene_token)
        protected = {r["id"] for r in runs[:keep]}
        targets = [
            r["id"] for r in runs
            if r["id"] not in protected and r["status"] != RUN_STATUS_RUNNING
        ]
        for params_id in targets:
            self.delete_run(params_id)
        return targets


def _run_to_dict(row: DepthEstimationParams) -> dict[str, Any]:
    return {
        "id": row.id,
        "dataset_id": row.dataset_id,
        "scene_token": row.scene_token,
        "instance_tracking_2d_params_id": row.instance_tracking_2d_params_id,
        "model_name": row.model_name,
        "sample_interval": row.sample_interval,
        "use_lidar": row.use_lidar,
        "num_lidar_sweeps": row.num_lidar_sweeps,
        "mask_params": row.mask_params,
        "depth_params": row.depth_params,
        "lidar_params": row.lidar_params,
        "box_fitting_params": row.box_fitting_params,
        "status": row.status,
        "num_inferences": row.num_inferences,
        "num_boxes": row.num_boxes,
        "inference_time": row.inference_time,
        "started_at": row.started_at,
        "ended_at": row.ended_at,
    }
