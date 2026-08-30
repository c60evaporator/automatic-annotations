"""センサー定義・キャリブレーション・自車位置姿勢・センサーフレームデータ"""
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.ann_intermediate import (
        DepthEstimation,
        Detection2D,
        InstanceTracking2D,
    )
    from app.models.dataset import Dataset
    from app.models.scene import Sample


class Sensor(Base):
    """センサー定義（カメラ・LiDAR・RADARの種別）"""
    __tablename__ = "sensors"
    __table_args__ = (
        # channel 指定でのセンサー引き当てをデータセット内で絞り込む
        Index("ix_sensors_dataset_channel", "dataset_id", "channel"),
    )
    # Columns
    token:      Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    channel:    Mapped[str] = mapped_column(String, nullable=False)  # 'CAM_FRONT', 'LIDAR_TOP' etc.
    modality:   Mapped[str] = mapped_column(String, nullable=False)  # 'camera', 'lidar', 'radar'
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    calibrated_sensors: Mapped[list["CalibratedSensor"]] = relationship(
        back_populates="sensor",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CalibratedSensor(Base):
    """キャリブレーション済みセンサー（車体座標系での位置・姿勢・内部パラメータ）"""
    __tablename__ = "calibrated_sensors"
    # Columns
    token:        Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:   Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # RESTRICT FK の参照チェックを index scan にするため index=True
    sensor_token: Mapped[str] = mapped_column(
        ForeignKey("sensors.token", ondelete="RESTRICT"), nullable=False, index=True
    )
    # 外部パラメータ（車体座標系での位置・姿勢）
    translation: Mapped[list] = mapped_column(JSON, nullable=False)  # [x, y, z]
    rotation:    Mapped[list] = mapped_column(JSON, nullable=False)  # [w, x, y, z]
    # 内部パラメータ（カメラのみ。LiDAR/RADARはnull）
    camera_intrinsic: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 3x3 matrix
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    sensor:      Mapped["Sensor"]           = relationship(back_populates="calibrated_sensors")
    sample_data: Mapped[list["SampleData"]] = relationship(
        back_populates="calibrated_sensor",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class EgoPose(Base):
    """自車位置姿勢（各タイムスタンプでのグローバル座標）"""
    __tablename__ = "ego_poses"
    # Columns
    token:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp:   Mapped[int] = mapped_column(BigInteger, nullable=False)
    # グローバル座標系での位置・姿勢
    translation: Mapped[list] = mapped_column(JSON, nullable=False)  # [x, y, z]
    rotation:    Mapped[list] = mapped_column(JSON, nullable=False)  # [w, x, y, z]
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    sample_data: Mapped[list["SampleData"]] = relationship(
        back_populates="ego_pose",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SampleData(Base):
    """センサー1フレーム分のデータ参照（ファイルパス・タイムスタンプ）"""
    __tablename__ = "sample_data"
    __table_args__ = (
        # (sample_token, is_key_frame) 複合インデックス：キーフレーム絞り込みクエリを高速化
        Index("ix_sample_data_sample_key_frame", "sample_token", "is_key_frame"),
    )
    # Columns
    token:                   Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:              Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample_token:            Mapped[str] = mapped_column(
        ForeignKey("samples.token", ondelete="CASCADE"), nullable=False
    )
    # RESTRICT FK の参照チェック（親削除時のトリガ）を index scan にするため index=True
    calibrated_sensor_token: Mapped[str] = mapped_column(
        ForeignKey("calibrated_sensors.token", ondelete="RESTRICT"), nullable=False, index=True
    )
    ego_pose_token:          Mapped[str] = mapped_column(
        ForeignKey("ego_poses.token", ondelete="RESTRICT"), nullable=False, index=True
    )
    # データファイル参照
    filename:     Mapped[str]  = mapped_column(String, nullable=False)
    fileformat:   Mapped[str]  = mapped_column(String, nullable=False)  # 'jpg', 'pcd', 'bin', 'npz'
    timestamp:    Mapped[int]  = mapped_column(BigInteger, nullable=False)
    is_key_frame: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # カメラのみ（LiDARはnull）
    width:  Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 隣接フレーム参照：参照先が消えても行は残す
    # SET NULL トリガ（行削除ごとの WHERE prev/next=$1）を index scan にするため index=True
    prev: Mapped[str | None] = mapped_column(
        ForeignKey("sample_data.token", ondelete="SET NULL"), nullable=True, index=True
    )
    next: Mapped[str | None] = mapped_column(
        ForeignKey("sample_data.token", ondelete="SET NULL"), nullable=True, index=True
    )
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    sample:            Mapped["Sample"]           = relationship(back_populates="sample_data")
    calibrated_sensor: Mapped["CalibratedSensor"] = relationship(back_populates="sample_data")
    ego_pose:          Mapped["EgoPose"]          = relationship(back_populates="sample_data")
    # 自動アノテーション中間出力（Step 1〜3）
    detection_2ds: Mapped[list["Detection2D"]] = relationship(
        back_populates="sample_data",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    instance_tracking_2ds: Mapped[list["InstanceTracking2D"]] = relationship(
        back_populates="sample_data",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    depth_estimations: Mapped[list["DepthEstimation"]] = relationship(
        back_populates="sample_data",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
