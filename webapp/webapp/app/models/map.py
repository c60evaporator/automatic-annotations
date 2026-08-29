from sqlalchemy import Index
from sqlalchemy import String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

class MapMeta(Base):
    """マップメタ情報（ロケーション・バージョン・サイズ）"""
    __tablename__ = "map_meta"
    __table_args__ = (
        # location 絞り込みを map_set 内で行うための複合インデックス
        Index("ix_map_meta_map_set_location", "map_set_id", "location"),
    )
    token:      Mapped[str]   = mapped_column(String, primary_key=True)
    # 元マップデータでの token（名前空間化前）
    source_token: Mapped[str | None] = mapped_column(String, nullable=True)
    location:   Mapped[str]   = mapped_column(String, nullable=False)  # 'boston-seaport' etc.
    version:    Mapped[str]   = mapped_column(String, nullable=False)  # '1.3' etc.
    canvas_edge: Mapped[list] = mapped_column(JSON, nullable=False)    # [width_m, height_m]
    # basemap image path (optional, for visualization reference)
    basemap_path: Mapped[str | None] = mapped_column(String, nullable=True)
