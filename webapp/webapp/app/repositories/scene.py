"""シーン単位のクエリ."""
from __future__ import annotations

from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models.map import MapMeta
from app.models.scene import Log, Sample, Scene
from app.models.sensor import CalibratedSensor, EgoPose, SampleData, Sensor

# waypoint を取る基準チャンネル。
# nuScenes は sample_data ごとに ego_pose を持つため、どのセンサーの
# タイムスタンプで軌跡を描くかを決める必要がある。LiDAR がキーフレームの
# 基準になっているので既定にする。
DEFAULT_POSE_CHANNEL = "LIDAR_TOP"


class SceneRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_by_dataset(self, dataset_id: str) -> list[dict[str, Any]]:
        """シーン一覧を、ログ情報と開始・終了時刻付きで返す.

        start_time / end_time は sample.timestamp（UNIX マイクロ秒）の
        最小・最大。表示用の整形は UI 側で行う。
        """
        stmt = (
            select(
                Scene.token,
                Scene.name,
                Scene.description,
                Scene.nbr_samples,
                Scene.log_token,
                Scene.first_sample_token,
                Scene.last_sample_token,
                Log.location,
                Log.vehicle,
                Log.date_captured,
                func.min(Sample.timestamp).label("start_time"),
                func.max(Sample.timestamp).label("end_time"),
            )
            .join(Log, Log.token == Scene.log_token)
            .outerjoin(Sample, Sample.scene_token == Scene.token)
            .where(Scene.dataset_id == dataset_id)
            .group_by(Scene.token)
            .order_by(Scene.name)
        )
        return [dict(r) for r in self.session.execute(stmt).mappings()]

    def get(self, dataset_id: str, scene_token: str) -> dict[str, Any] | None:
        stmt = (
            select(
                Scene.token,
                Scene.name,
                Scene.description,
                Scene.nbr_samples,
                Scene.log_token,
                Scene.first_sample_token,
                Scene.last_sample_token,
                Log.location,
                Log.vehicle,
                Log.date_captured,
                func.min(Sample.timestamp).label("start_time"),
                func.max(Sample.timestamp).label("end_time"),
            )
            .join(Log, Log.token == Scene.log_token)
            .outerjoin(Sample, Sample.scene_token == Scene.token)
            .where(Scene.dataset_id == dataset_id, Scene.token == scene_token)
            .group_by(Scene.token)
        )
        row = self.session.execute(stmt).mappings().first()
        return dict(row) if row else None

    def _waypoint_stmt(self, scene_token: str, channel: str) -> Select:
        return (
            select(
                Sample.token.label("sample_token"),
                Sample.timestamp,
                EgoPose.translation,
                EgoPose.rotation,
            )
            .join(SampleData, SampleData.sample_token == Sample.token)
            .join(CalibratedSensor,
                  CalibratedSensor.token == SampleData.calibrated_sensor_token)
            .join(Sensor, Sensor.token == CalibratedSensor.sensor_token)
            .join(EgoPose, EgoPose.token == SampleData.ego_pose_token)
            .where(
                Sample.scene_token == scene_token,
                SampleData.is_key_frame.is_(True),
                Sensor.channel == channel,
            )
            .order_by(Sample.timestamp)
        )

    def list_waypoints(
        self, scene_token: str, channel: str = DEFAULT_POSE_CHANNEL
    ) -> list[dict[str, Any]]:
        """シーン内の各キーフレームの自車位置を時系列で返す.

        translation は JSON の [x, y, z]。UI 側で扱いやすいよう
        x / y / z に展開して返す。
        """
        rows = self.session.execute(self._waypoint_stmt(scene_token, channel)).mappings()
        waypoints: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            t = r["translation"]
            waypoints.append({
                "sample_idx": i,
                "sample_token": r["sample_token"],
                "timestamp": r["timestamp"],
                "x": t[0],
                "y": t[1],
                "z": t[2],
                "rotation": r["rotation"],
            })
        return waypoints

    def get_map(self, dataset_id: str, scene_token: str) -> dict[str, Any] | None:
        """シーンに対応する basemap 情報を返す.

        旧実装では map_token ごとの canvas_edge をコードに直書きしていたが、
        インポート時に Map Expansion から DB に取り込んでいるので不要になった。
        紐付けは scene → log.location → map_meta.location で辿る。
        """
        stmt = (
            select(
                MapMeta.token,
                MapMeta.location,
                MapMeta.version,
                MapMeta.canvas_edge,
                MapMeta.basemap_path,
            )
            .join(Log, Log.location == MapMeta.location)
            .join(Scene, Scene.log_token == Log.token)
            .where(
                Scene.token == scene_token,
                MapMeta.dataset_id == dataset_id,
            )
            .limit(1)
        )
        row = self.session.execute(stmt).mappings().first()
        return dict(row) if row else None

    def count_samples(self, scene_token: str) -> int:
        return self.session.scalar(
            select(func.count()).select_from(Sample)
            .where(Sample.scene_token == scene_token)
        ) or 0
