"""データセット（全テーブルのルート）"""
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Dataset(Base):
    """データセット単位.

    データ系の全テーブルは dataset_id でこのテーブルに紐づき、
    データセットごとに独立して管理・削除される。

    制約: 1 データセットにつき読み込めるメタデータ・マップデータは 1 つのみ。
    （token がグローバルに一意でないため、trainval と mini を同一データセットに
      読み込むと primary key 衝突が発生する）
    """
    __tablename__ = "datasets"

    # Columns
    id:   Mapped[str] = mapped_column(String, primary_key=True)  # UUID文字列
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    # 'nuscenes' | 'carla' 等。インポート形式の判別に使用
    dataset_type: Mapped[str] = mapped_column(String, nullable=False)
    version:      Mapped[str | None] = mapped_column(String, nullable=True)  # 'v1.0-trainval' 等
    dataroot:     Mapped[str] = mapped_column(String, nullable=False)  # settings.DATA_ROOT からの相対パス
    description:  Mapped[str | None] = mapped_column(String, nullable=True)
    created_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
