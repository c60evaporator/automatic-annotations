"""nuScenes 形式のエクスポートに必要な行を取り出す.

devkit が読み込むテーブル一式を、指定したシーンぶんだけ抜き出す。
参照が閉じていないと devkit のトークン解決が失敗するため、
「シーン → sample → sample_data → ego_pose / calibrated_sensor → sensor」と
参照を辿って必要な行だけを集める。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.annotation import (
    SOURCE_AUTO,
    SOURCE_IMPORTED,
    Attribute,
    Category,
    Instance,
    SampleAnnotation,
    Visibility,
)
from app.models.dataset import Dataset
from app.models.scene import Log, Sample, Scene
from app.models.sensor import CalibratedSensor, EgoPose, SampleData, Sensor

# エクスポート対象のアノテーション種別
ANNOTATION_SOURCE_AUTO = "auto"
ANNOTATION_SOURCE_GT = "gt"
ANNOTATION_SOURCE_BOTH = "both"
ANNOTATION_SOURCES = (
    ANNOTATION_SOURCE_AUTO,
    ANNOTATION_SOURCE_GT,
    ANNOTATION_SOURCE_BOTH,
)

_SOURCE_FILTER = {
    ANNOTATION_SOURCE_AUTO: (SOURCE_AUTO,),
    ANNOTATION_SOURCE_GT: (SOURCE_IMPORTED,),
    ANNOTATION_SOURCE_BOTH: (SOURCE_AUTO, SOURCE_IMPORTED),
}


class NuScenesExportRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ── 基本情報 ──────────────────────────────────────────────────────────

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self.session.scalars(
            select(Dataset).where(Dataset.id == dataset_id)
        ).first()
        if row is None:
            return None
        return {
            "id": row.id, "name": row.name, "version": row.version,
            "dataroot": row.dataroot, "dataset_type": row.dataset_type,
        }

    def list_scenes(
        self, dataset_id: str, scene_tokens: list[str] | None = None
    ) -> list[dict[str, Any]]:
        stmt = select(Scene).where(Scene.dataset_id == dataset_id)
        if scene_tokens:
            stmt = stmt.where(Scene.token.in_(scene_tokens))
        return [{
            "token": r.token, "log_token": r.log_token, "name": r.name,
            "description": r.description, "nbr_samples": r.nbr_samples,
            "first_sample_token": r.first_sample_token,
            "last_sample_token": r.last_sample_token,
        } for r in self.session.scalars(stmt.order_by(Scene.name))]

    def list_logs(self, log_tokens: list[str]) -> list[dict[str, Any]]:
        if not log_tokens:
            return []
        stmt = select(Log).where(Log.token.in_(log_tokens))
        return [{
            "token": r.token, "logfile": r.logfile, "vehicle": r.vehicle,
            "date_captured": r.date_captured, "location": r.location,
        } for r in self.session.scalars(stmt)]

    # ── サンプルとセンサーデータ ──────────────────────────────────────────

    def list_samples(self, scene_tokens: list[str]) -> list[dict[str, Any]]:
        if not scene_tokens:
            return []
        stmt = (
            select(Sample)
            .where(Sample.scene_token.in_(scene_tokens))
            .order_by(Sample.scene_token, Sample.timestamp)
        )
        return [{
            "token": r.token, "timestamp": r.timestamp,
            "prev": r.prev, "next": r.next, "scene_token": r.scene_token,
        } for r in self.session.scalars(stmt)]

    def list_sample_data(self, sample_tokens: list[str]) -> list[dict[str, Any]]:
        """sweep を含む全 sample_data を返す（キーフレームだけでは不足）."""
        if not sample_tokens:
            return []
        stmt = (
            select(SampleData)
            .where(SampleData.sample_token.in_(sample_tokens))
            .order_by(SampleData.timestamp)
        )
        return [{
            "token": r.token, "sample_token": r.sample_token,
            "ego_pose_token": r.ego_pose_token,
            "calibrated_sensor_token": r.calibrated_sensor_token,
            "filename": r.filename, "fileformat": r.fileformat,
            "width": r.width, "height": r.height, "timestamp": r.timestamp,
            "is_key_frame": r.is_key_frame, "prev": r.prev, "next": r.next,
        } for r in self.session.scalars(stmt)]

    def list_ego_poses(self, tokens: list[str]) -> list[dict[str, Any]]:
        if not tokens:
            return []
        stmt = select(EgoPose).where(EgoPose.token.in_(tokens))
        return [{
            "token": r.token, "timestamp": r.timestamp,
            "translation": r.translation, "rotation": r.rotation,
        } for r in self.session.scalars(stmt)]

    def list_calibrated_sensors(self, tokens: list[str]) -> list[dict[str, Any]]:
        if not tokens:
            return []
        stmt = select(CalibratedSensor).where(CalibratedSensor.token.in_(tokens))
        return [{
            "token": r.token, "sensor_token": r.sensor_token,
            "translation": r.translation, "rotation": r.rotation,
            "camera_intrinsic": r.camera_intrinsic or [],
        } for r in self.session.scalars(stmt)]

    def list_sensors(self, tokens: list[str]) -> list[dict[str, Any]]:
        if not tokens:
            return []
        stmt = select(Sensor).where(Sensor.token.in_(tokens))
        return [{
            "token": r.token, "channel": r.channel, "modality": r.modality,
        } for r in self.session.scalars(stmt)]

    # ── 分類系（データセット全体をそのまま出す）─────────────────────────

    def list_categories(self, dataset_id: str) -> list[dict[str, Any]]:
        stmt = select(Category).where(Category.dataset_id == dataset_id)
        return [{
            "token": r.token, "name": r.name, "description": r.description or "",
        } for r in self.session.scalars(stmt)]

    def list_attributes(self, dataset_id: str) -> list[dict[str, Any]]:
        stmt = select(Attribute).where(Attribute.dataset_id == dataset_id)
        return [{
            "token": r.token, "name": r.name, "description": r.description or "",
        } for r in self.session.scalars(stmt)]

    def list_visibilities(self, dataset_id: str) -> list[dict[str, Any]]:
        stmt = select(Visibility).where(Visibility.dataset_id == dataset_id)
        return [{
            "token": r.token, "level": r.level, "description": r.description or "",
        } for r in self.session.scalars(stmt)]

    # ── アノテーション ────────────────────────────────────────────────────

    def list_annotations(
        self,
        sample_tokens: list[str],
        *,
        source: str = ANNOTATION_SOURCE_AUTO,
        depth_estimation_params_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """対象サンプルのアノテーションを返す.

        Args:
            source: 'auto' | 'gt' | 'both'
            depth_estimation_params_id: 指定すると、その run が生成した
                自動アノテーションだけに絞る（run ごとに出し分けるため）
        """
        if not sample_tokens:
            return []
        sources = _SOURCE_FILTER.get(source, (SOURCE_AUTO,))
        stmt = select(SampleAnnotation).where(
            SampleAnnotation.sample_token.in_(sample_tokens),
            SampleAnnotation.source.in_(sources),
        )
        if depth_estimation_params_id is not None:
            # GT を含める指定なら、GT は run に紐づかないので残す
            condition = (
                SampleAnnotation.depth_estimation_params_id
                == depth_estimation_params_id
            )
            if SOURCE_IMPORTED in sources:
                condition = condition | (
                    SampleAnnotation.source == SOURCE_IMPORTED
                )
            stmt = stmt.where(condition)

        return [{
            "token": r.token, "sample_token": r.sample_token,
            "instance_token": r.instance_token,
            "translation": r.translation, "rotation": r.rotation,
            "size": r.size, "prev": r.prev, "next": r.next,
            "num_lidar_pts": r.num_lidar_pts, "num_radar_pts": r.num_radar_pts,
            "visibility_token": r.visibility_token,
            "source": r.source,
        } for r in self.session.scalars(stmt)]

    def list_instances(self, instance_tokens: list[str]) -> list[dict[str, Any]]:
        if not instance_tokens:
            return []
        stmt = select(Instance).where(Instance.token.in_(instance_tokens))
        return [{
            "token": r.token, "category_token": r.category_token,
            "nbr_annotations": r.nbr_annotations,
            "first_annotation_token": r.first_annotation_token,
            "last_annotation_token": r.last_annotation_token,
            "source": r.source,
        } for r in self.session.scalars(stmt)]
