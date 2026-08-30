from app.models.scene import Sample
from sqlalchemy import String, Integer, ForeignKey, Index, JSON, Table, Column, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.db.base import Base


class Category(Base):
    """物体カテゴリ（car, pedestrian等）"""
    __tablename__ = "categories"
    # Columns
    token:       Mapped[str]      = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str]      = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name:        Mapped[str]      = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    # Relationships
    instances: Mapped[list["Instance"]] = relationship(
        back_populates="category",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Attribute(Base):
    """アノテーション属性（vehicle.moving等）"""
    __tablename__ = "attributes"
    # Columns
    token:       Mapped[str]      = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str]      = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name:        Mapped[str]      = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)


class Instance(Base):
    """物体インスタンス（シーンをまたぐ同一物体の追跡単位）"""
    __tablename__ = "instances"
    # Columns
    token:           Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:      Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category_token:  Mapped[str] = mapped_column(
        ForeignKey("categories.token", ondelete="RESTRICT"), nullable=False
    )
    nbr_annotations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 先頭・末尾アノテーションへの参照（SET NULL：アノテーション削除時も行は残す）
    first_annotation_token: Mapped[str | None] = mapped_column(
        String,  # SQLiteと相性が悪いため循環FKは張らない
        nullable=True,
    )
    last_annotation_token: Mapped[str | None] = mapped_column(
        String,  # SQLiteと相性が悪いため循環FKは張らない
        nullable=True,
    )
    # 初回インポート=false / ユーザ追加=true
    is_user_created: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default='false'
    )
    # Relationships
    category:    Mapped["Category"]               = relationship(back_populates="instances")
    annotations: Mapped[list["SampleAnnotation"]] = relationship(
        back_populates="instance",
        foreign_keys="SampleAnnotation.instance_token",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Visibility(Base):
    """可視性レベル（0-40%, 40-60%, 60-80%, 80-100%）"""
    __tablename__ = "visibilities"
    # Columns
    token:       Mapped[str]      = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str]      = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    level:       Mapped[str]      = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)


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
    ),
)


class SampleAnnotation(Base):
    """1フレームの3Dバウンディングボックスアノテーション"""
    __tablename__ = "sample_annotations"
    __table_args__ = (
        # sample 単位のアノテーション取得をデータセット内で絞り込むための複合インデックス
        Index("ix_sample_annotations_dataset_sample_token", "dataset_id", "sample_token"),
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
    prev: Mapped[str | None] = mapped_column(
        ForeignKey("sample_annotations.token", ondelete="SET NULL"), nullable=True
    )
    next: Mapped[str | None] = mapped_column(
        ForeignKey("sample_annotations.token", ondelete="SET NULL"), nullable=True
    )
    # アノテーション品質
    num_lidar_pts:    Mapped[int]      = mapped_column(Integer, nullable=False, default=0)
    num_radar_pts:    Mapped[int]      = mapped_column(Integer, nullable=False, default=0)
    visibility_token: Mapped[str | None] = mapped_column(
        ForeignKey("visibilities.token", ondelete="SET NULL"), nullable=True
    )
    # 初回インポート=false / ユーザ追加=true
    is_user_created: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default='false'
    )
    # Relationships
    sample:     Mapped["Sample"]          = relationship(back_populates="annotations")
    instance:   Mapped["Instance"]        = relationship(
        back_populates="annotations",
        foreign_keys=[instance_token],
    )
    visibility: Mapped["Visibility | None"] = relationship()
    attributes: Mapped[list["Attribute"]]   = relationship(
        secondary=annotation_attribute,
    )
