from sqlalchemy import Index
from sqlalchemy import String, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

class MapMeta(Base):
    """マップメタ情報（ロケーション・バージョン・サイズ）"""
    __tablename__ = "map_meta"
    # location絞り込みを行うためのインデックス
    __table_args__ = (
        Index("ix_map_meta_location", "location"),
    )
    token:      Mapped[str]   = mapped_column(String, primary_key=True)
    dataset_id:     Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    location:   Mapped[str]   = mapped_column(String, nullable=False)  # 'boston-seaport' etc.
    version:    Mapped[str]   = mapped_column(String, nullable=False)  # '1.3' etc.
    canvas_edge: Mapped[list] = mapped_column(JSON, nullable=False)    # [width_m, height_m]
    # basemap image path (optional, for visualization reference)
    basemap_path: Mapped[str | None] = mapped_column(String, nullable=True)
