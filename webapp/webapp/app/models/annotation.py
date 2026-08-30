"""カテゴリ・インスタンス・3Dバウンディングボックスアノテーション"""
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Table,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.ann_intermediate import DepthEstimationParams
    from app.models.dataset import Dataset
    from app.models.scene import Sample


# アノテーション／インスタンスの生成元。
#   'imported' : nuScenes JSON からの初回インポート（GT）
#   'auto'     : 自動アノテーションパイプラインによる生成
#   'manual'   : UI 上でユーザーが手動追加
SOURCE_IMPORTED = "imported"
SOURCE_AUTO = "auto"
SOURCE_MANUAL = "manual"


class Category(Base):
    """物体カテゴリ（car, pedestrian等）"""
    __tablename__ = "categories"
    __table_args__ = (
        # category_name からの引き当て（自動アノテーション結果の登録時に多用）
        Index("ix_categories_dataset_name", "dataset_id", "name", unique=True),
    )
    # Columns
    token:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name:        Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    instances: Mapped[list["Instance"]] = relationship(
        back_populates="category",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Attribute(Base):
    """アノテーション属性（vehicle.moving等）"""
    __tablename__ = "attributes"
    # Columns
    token:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name:        Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    # Relationships
    dataset: Mapped["Dataset"] = relationship()


class Visibility(Base):
    """可視性レベル（0-40%, 40-60%, 60-80%, 80-100%）"""
    __tablename__ = "visibilities"
    # Columns
    token:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    level:       Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    # Relationships
    dataset: Mapped["Dataset"] = relationship()


class Instance(Base):
    """物体インスタンス（シーン内で同一物体を追跡する単位）"""
    __tablename__ = "instances"
    # Columns
    token:           Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:      Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # RESTRICT FK の参照チェックを index scan にするため index=True
    category_token:  Mapped[str] = mapped_column(
        ForeignKey("categories.token", ondelete="RESTRICT"), nullable=False, index=True
    )
    nbr_annotations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 先頭・末尾アノテーションへの参照
    # instances → sample_annotations → instances の循環 FK は SQLite と相性が悪いため
    # FK 制約は張らず、値の整合はアプリ側（annotation_service）で担保する
    first_annotation_token: Mapped[str | None] = mapped_column(String, nullable=True)
    last_annotation_token:  Mapped[str | None] = mapped_column(String, nullable=True)
    # 生成元: 'imported' | 'auto' | 'manual'
    source: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text(f"'{SOURCE_IMPORTED}'")
    )
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    category:    Mapped["Category"]               = relationship(back_populates="instances")
    annotations: Mapped[list["SampleAnnotation"]] = relationship(
        back_populates="instance",
        foreign_keys="SampleAnnotation.instance_token",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# SampleAnnotation ↔ Attribute の多対多中間テーブル
# 両端の FK が dataset_id を持つため、この表自体には dataset_id を持たせない
annotation_attribute = Table(
    "annotation_attributes",
    Base.metadata,
    Column(
        "annotation_token",
        ForeignKey("sample_annotations.token", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "attribute_token",
        ForeignKey("attributes.token", ondelete="RESTRICT"),
        primary_key=True,
        index=True,
    ),
)


class SampleAnnotation(Base):
    """1フレームの3Dバウンディングボックスアノテーション"""
    __tablename__ = "sample_annotations"
    __table_args__ = (
        # sample 単位のアノテーション取得をデータセット内で絞り込むための複合インデックス
        Index("ix_sample_annotations_dataset_sample_token", "dataset_id", "sample_token"),
        # GT と自動生成結果を並べて比較する UI 用
        Index("ix_sample_annotations_sample_source", "sample_token", "source"),
    )
    # Columns
    token:          Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:     Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample_token:   Mapped[str] = mapped_column(
        ForeignKey("samples.token", ondelete="CASCADE"), nullable=False, index=True
    )
    instance_token: Mapped[str] = mapped_column(
        ForeignKey("instances.token", ondelete="CASCADE"), nullable=False, index=True
    )
    # 3Dバウンディングボックス（グローバル座標）
    translation: Mapped[list] = mapped_column(JSON, nullable=False)  # [x, y, z] 中心座標
    rotation:    Mapped[list] = mapped_column(JSON, nullable=False)  # [w, x, y, z] クォータニオン
    size:        Mapped[list] = mapped_column(JSON, nullable=False)  # [width, length, height]
    # トラッキング（前後フレームの同インスタンスアノテーションへの参照）
    # SET NULL トリガを index scan にするため index=True
    prev: Mapped[str | None] = mapped_column(
        ForeignKey("sample_annotations.token", ondelete="SET NULL"), nullable=True, index=True
    )
    next: Mapped[str | None] = mapped_column(
        ForeignKey("sample_annotations.token", ondelete="SET NULL"), nullable=True, index=True
    )
    # アノテーション品質
    num_lidar_pts:    Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_radar_pts:    Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    visibility_token: Mapped[str | None] = mapped_column(
        ForeignKey("visibilities.token", ondelete="SET NULL"), nullable=True, index=True
    )
    # 生成元: 'imported' | 'auto' | 'manual'
    source: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text(f"'{SOURCE_IMPORTED}'")
    )
    # source='auto' の場合、どの Box Fitting 実行で生成されたかを辿るための参照。
    # 実行単位でのやり直し（DELETE → 再推論）を可能にする
    depth_estimation_params_id: Mapped[str | None] = mapped_column(
        ForeignKey("depth_estimation_params.id", ondelete="CASCADE"), nullable=True, index=True
    )
    # 自動生成ボックスの当てはまり具合（source='auto' のみ。UI でのフィルタ用）
    score: Mapped[float | None] = mapped_column(nullable=True)
    # 生成後に UI で手修正されたか
    manually_modified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("0")
    )
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    sample:     Mapped["Sample"]   = relationship(back_populates="annotations")
    instance:   Mapped["Instance"] = relationship(
        back_populates="annotations",
        foreign_keys=[instance_token],
    )
    visibility: Mapped["Visibility | None"] = relationship()
    attributes: Mapped[list["Attribute"]]   = relationship(secondary=annotation_attribute)
    depth_estimation_params: Mapped["DepthEstimationParams | None"] = relationship(
        back_populates="annotations"
    )
