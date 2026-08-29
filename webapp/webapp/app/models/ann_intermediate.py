"""アノテーションの中間出力保存用テーブル"""
from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.models.scene import SampleData
from app.db.base import Base

class Detection2DParams(Base):
    """2D物体検出で使用したパラメータ"""
    __tablename__ = "detection_2d_params"
    # Columns
    id:       Mapped[str] = mapped_column(String, primary_key=True)
    sample_interval:   Mapped[int] = mapped_column(Integer, nullable=False)  # 推論を実施するキーフレーム間隔
    nusc_category_to_label:   Mapped[dict] = mapped_column(JSON, nullable=False)  # 元nuScenesのcategory_nameと推論で使用するラベル名のマッピング
    label_to_nusc_category:   Mapped[dict] = mapped_column(JSON, nullable=False)  # 推論で使用するラベル名を最終的なnuScenesのcategory_name（ラベルプロンプトをまとめて推論する単位）に変換するマッピング
    label_to_category_group:   Mapped[dict] = mapped_column(JSON, nullable=False)  # 推論で使用するラベル名をカテゴリグループに変換するマッピング
    score_threshold:   Mapped[list] = mapped_column(JSON, nullable=False)  # カテゴリごとに閾値を変えるのでJSON
    nms_same_class_ious:   Mapped[list] = mapped_column(JSON, nullable=False)  # カテゴリごとに閾値を変えるのでJSON
    nms_cross_class_ious:   Mapped[list] = mapped_column(JSON, nullable=False)  # カテゴリごとに閾値を変えるのでJSON
    num_inferences:   Mapped[int] = mapped_column(Integer, nullable=False)  # 推論を実施した回数
    inference_time:   Mapped[float] = mapped_column(Float, nullable=True) # 推論にかかった時間（オーバーヘッドを除くためended_at - started_atより短い）
    started_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Relationships
    detection_2ds: Mapped["Detection2D"] = relationship(back_populates="detection_2d_params")


class Detection2D(Base):
    """2D物体検出の結果（GroundingDINO）"""
    __tablename__ = "detection_2ds"
    # Columns
    id:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:     Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample_data_token:   Mapped[str] = mapped_column(
        ForeignKey("sample_data.token", ondelete="CASCADE"), nullable=False, index=True
    )
    detection_2d_params_id:   Mapped[str] = mapped_column(
        ForeignKey("detection_2d_params.id", ondelete="CASCADE"), nullable=False, index=True
    )
    xmin:       Mapped[int] = mapped_column(Integer, nullable=False)
    ymin:       Mapped[int] = mapped_column(Integer, nullable=False)
    xmax:       Mapped[int] = mapped_column(Integer, nullable=False)
    ymax:       Mapped[int] = mapped_column(Integer, nullable=False)
    label:      Mapped[str] = mapped_column(String, nullable=False)
    score:      Mapped[float] = mapped_column(Float, nullable=True)
    created_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Manual modification flag
    manually_modified:   Mapped[bool] = mapped_column(nullable=False, default=False)
    updated_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    # Relationships
    sample_data: Mapped["SampleData"] = relationship(back_populates="detection_2ds")
    detection_2d_params: Mapped["Detection2DParams"] = relationship(back_populates="detection_2ds")


class InstanceTracking2D(Base):
    """2Dインスタンスセグメンテーショントラッキングの結果（SAM2）"""
    __tablename__ = "instance_tracking_2ds"
    # Columns
    id:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:     Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample_data_token:   Mapped[str] = mapped_column(
        ForeignKey("sample_data.token", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Relationships
    sample_data: Mapped["SampleData"] = relationship(back_populates="instance_tracking_2ds")


class DepthEstimation(Base):
    """深度推定の結果"""
    __tablename__ = "depth_estimations"
    # Columns
    id:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:     Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample_data_token:   Mapped[str] = mapped_column(
        ForeignKey("sample_data.token", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Relationships
    sample_data: Mapped["SampleData"] = relationship(back_populates="depth_estimations")
