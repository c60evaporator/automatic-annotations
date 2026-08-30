"""走行ログ・シーン・サンプル（キーフレーム）"""
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.annotation import SampleAnnotation
    from app.models.dataset import Dataset
    from app.models.sensor import SampleData


class Log(Base):
    """走行ログ（場所・車両・日付等のメタ情報）"""
    __tablename__ = "logs"
    __table_args__ = (
        # location フィルタをデータセット内で絞り込むための複合インデックス
        Index("ix_logs_dataset_location", "dataset_id", "location"),
    )
    # Columns
    token:         Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:    Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_token:  Mapped[str | None] = mapped_column(String, nullable=True)
    logfile:       Mapped[str] = mapped_column(String, nullable=False)
    vehicle:       Mapped[str] = mapped_column(String, nullable=False)
    date_captured: Mapped[str] = mapped_column(String, nullable=False)
    location:      Mapped[str] = mapped_column(String, nullable=False)  # 'boston-seaport' etc.
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    scenes: Mapped[list["Scene"]] = relationship(
        back_populates="log",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Scene(Base):
    """シーン（約20秒の走行シーケンス）"""
    __tablename__ = "scenes"
    __table_args__ = (
        # log フィルタをデータセット内で絞り込むための複合インデックス
        Index("ix_scenes_dataset_log_token", "dataset_id", "log_token"),
    )
    # Columns
    token:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # RESTRICT FK の参照チェックと log フィルタ検索のため index=True
    log_token:   Mapped[str] = mapped_column(
        ForeignKey("logs.token", ondelete="RESTRICT"), nullable=False, index=True
    )
    name:        Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    nbr_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    # samples への FK は張らない（scene → sample → scene の循環参照になるため）
    first_sample_token: Mapped[str] = mapped_column(String, nullable=False)
    last_sample_token:  Mapped[str] = mapped_column(String, nullable=False)
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    log:     Mapped["Log"]          = relationship(back_populates="scenes")
    samples: Mapped[list["Sample"]] = relationship(
        back_populates="scene",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Sample(Base):
    """サンプル（キーフレーム、約0.5秒間隔）"""
    __tablename__ = "samples"
    __table_args__ = (
        # scene 内サンプル取得をデータセット内で絞り込むための複合インデックス
        Index("ix_samples_dataset_scene_token", "dataset_id", "scene_token"),
        # タイムライン順の取得を index scan にする
        Index("ix_samples_scene_timestamp", "scene_token", "timestamp"),
    )
    # Columns
    token:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scene_token: Mapped[str] = mapped_column(
        ForeignKey("scenes.token", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp:   Mapped[int] = mapped_column(BigInteger, nullable=False)  # UNIX usec
    # SET NULL トリガ（行削除ごとの WHERE prev/next=$1）を index scan にするため index=True
    prev: Mapped[str | None] = mapped_column(
        ForeignKey("samples.token", ondelete="SET NULL"), nullable=True, index=True
    )
    next: Mapped[str | None] = mapped_column(
        ForeignKey("samples.token", ondelete="SET NULL"), nullable=True, index=True
    )
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    scene:       Mapped["Scene"]            = relationship(back_populates="samples")
    sample_data: Mapped[list["SampleData"]] = relationship(
        back_populates="sample",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    annotations: Mapped[list["SampleAnnotation"]] = relationship(
        back_populates="sample",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
