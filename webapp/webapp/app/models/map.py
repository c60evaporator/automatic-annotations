"""マップメタ情報"""
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.dataset import Dataset


class MapMeta(Base):
    """マップメタ情報（ロケーション・バージョン・サイズ）"""
    __tablename__ = "map_meta"
    __table_args__ = (
        # location 絞り込みをデータセット内で行うための複合インデックス
        Index("ix_map_meta_dataset_location", "dataset_id", "location"),
    )
    # Columns
    token:       Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id:  Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    location:    Mapped[str] = mapped_column(String, nullable=False)  # 'boston-seaport' etc.
    version:     Mapped[str] = mapped_column(String, nullable=False)  # '1.3' etc.
    canvas_edge: Mapped[list] = mapped_column(JSON, nullable=False)   # [width_m, height_m]
    # basemap 画像パス（dataroot からの相対パス。waypoint_viewer の背景に使用）
    basemap_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
