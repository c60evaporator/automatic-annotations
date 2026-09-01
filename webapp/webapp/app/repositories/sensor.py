"""センサーフレーム単位のクエリ.

sample_data / calibrated_sensor / ego_pose は常に3点セットで使う
（画像を読む → 内部・外部パラメータで投影する → 自車位置でグローバル座標へ）。
毎回3回クエリを投げるのは無駄なので、1回の JOIN でまとめて返す。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models.scene import Sample
from app.models.sensor import CalibratedSensor, EgoPose, SampleData, Sensor


def _row_to_frame(r: Any) -> dict[str, Any]:
    """JOIN 結果の1行を、用途ごとにまとまった dict へ整形する.

    投影計算では calibrated_sensor と ego_pose をセットで扱うため、
    フラットに潰さずサブ dict のまま返す。
    """
    return {
        "token": r["sd_token"],
        "sample_token": r["sample_token"],
        "filename": r["filename"],
        "fileformat": r["fileformat"],
        "timestamp": r["sd_timestamp"],
        "is_key_frame": r["is_key_frame"],
        "width": r["width"],
        "height": r["height"],
        "prev": r["sd_prev"],
        "next": r["sd_next"],
        "channel": r["channel"],
        "modality": r["modality"],
        "calibrated_sensor": {
            "token": r["cs_token"],
            "sensor_token": r["sensor_token"],
            "translation": r["cs_translation"],
            "rotation": r["cs_rotation"],
            "camera_intrinsic": r["camera_intrinsic"],
        },
        "ego_pose": {
            "token": r["ep_token"],
            "timestamp": r["ep_timestamp"],
            "translation": r["ep_translation"],
            "rotation": r["ep_rotation"],
        },
    }


class SensorRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _frame_stmt(self) -> Select:
        """sample_data + calibrated_sensor + sensor + ego_pose の JOIN."""
        return (
            select(
                SampleData.token.label("sd_token"),
                SampleData.sample_token,
                SampleData.filename,
                SampleData.fileformat,
                SampleData.timestamp.label("sd_timestamp"),
                SampleData.is_key_frame,
                SampleData.width,
                SampleData.height,
                SampleData.prev.label("sd_prev"),
                SampleData.next.label("sd_next"),
                Sensor.channel,
                Sensor.modality,
                CalibratedSensor.token.label("cs_token"),
                CalibratedSensor.sensor_token,
                CalibratedSensor.translation.label("cs_translation"),
                CalibratedSensor.rotation.label("cs_rotation"),
                CalibratedSensor.camera_intrinsic,
                EgoPose.token.label("ep_token"),
                EgoPose.timestamp.label("ep_timestamp"),
                EgoPose.translation.label("ep_translation"),
                EgoPose.rotation.label("ep_rotation"),
            )
            .join(CalibratedSensor,
                  CalibratedSensor.token == SampleData.calibrated_sensor_token)
            .join(Sensor, Sensor.token == CalibratedSensor.sensor_token)
            .join(EgoPose, EgoPose.token == SampleData.ego_pose_token)
        )

    def _apply_filters(
        self, stmt: Select, keyframe_only: bool, sensor_token: str | None
    ) -> Select:
        if keyframe_only:
            stmt = stmt.where(SampleData.is_key_frame.is_(True))
        if sensor_token is not None:
            # calibrated_sensors 側で絞る。sensors への JOIN 結果ではなく
            # FK 列を直接比較したほうがインデックスが効く
            stmt = stmt.where(CalibratedSensor.sensor_token == sensor_token)
        return stmt

    def list_frames_by_sample(
        self,
        sample_token: str,
        *,
        keyframe_only: bool = True,
        sensor_token: str | None = None,
    ) -> list[dict[str, Any]]:
        """1 sample のセンサーフレームを返す.

        Args:
            keyframe_only: True なら is_key_frame のものだけ
            sensor_token: 指定するとそのセンサー（channel 単位）のみ
        """
        stmt = self._frame_stmt().where(SampleData.sample_token == sample_token)
        stmt = self._apply_filters(stmt, keyframe_only, sensor_token)
        stmt = stmt.order_by(Sensor.channel, SampleData.timestamp)
        return [_row_to_frame(r) for r in self.session.execute(stmt).mappings()]

    def list_frames_by_scene(
        self,
        scene_token: str,
        *,
        keyframe_only: bool = True,
        sensor_token: str | None = None,
    ) -> list[dict[str, Any]]:
        """シーン内の全センサーフレームを sample の時刻順でまとめて返す.

        sample ごとに list_frames_by_sample を呼ぶとクエリ数が
        sample 数だけ増える。バッチ処理（推論の入力作成など）では
        こちらを使って1回のクエリで取り切る。

        戻り値には sample_idx（シーン内でのキーフレーム連番）を付与する。
        """
        stmt = (
            self._frame_stmt()
            .add_columns(Sample.timestamp.label("sample_timestamp"))
            .join(Sample, Sample.token == SampleData.sample_token)
            .where(Sample.scene_token == scene_token)
        )
        stmt = self._apply_filters(stmt, keyframe_only, sensor_token)
        stmt = stmt.order_by(Sample.timestamp, Sensor.channel, SampleData.timestamp)

        frames: list[dict[str, Any]] = []
        # sample_token ごとの連番を振る（同じ sample に複数チャンネルが並ぶ）
        idx_of: dict[str, int] = {}
        for r in self.session.execute(stmt).mappings():
            frame = _row_to_frame(r)
            st = frame["sample_token"]
            if st not in idx_of:
                idx_of[st] = len(idx_of)
            frame["sample_idx"] = idx_of[st]
            frame["sample_timestamp"] = r["sample_timestamp"]
            frames.append(frame)
        return frames

    def list_sensors(
        self, dataset_id: str, *, modality: str | None = None
    ) -> list[dict[str, Any]]:
        """データセット内のセンサー一覧（UI のチャンネル選択用）.

        Args:
            modality: 'camera' / 'lidar' / 'radar' で絞る（None なら全件）。
                      2D 検出はカメラのみ対象なので 'camera' を指定する。
        """
        stmt = select(Sensor.token, Sensor.channel, Sensor.modality).where(
            Sensor.dataset_id == dataset_id
        )
        if modality is not None:
            stmt = stmt.where(Sensor.modality == modality)
        stmt = stmt.order_by(Sensor.modality, Sensor.channel)
        return [dict(r) for r in self.session.execute(stmt).mappings()]
