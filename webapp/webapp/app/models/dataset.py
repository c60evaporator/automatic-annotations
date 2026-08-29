from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Dataset(Base):
    """データセット（データ系の全テーブルは dataset_id でこのテーブルに紐づき、データセットごとに別個に管理される）"""
    __tablename__ = "datasets"

    id:   Mapped[str] = mapped_column(String, primary_key=True)  # UUID文字列
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    # 'nuscenes' | 'carla' 等。インポート形式の判別に使用
    dataset_type: Mapped[str] = mapped_column(String, nullable=False)
    version:      Mapped[str | None] = mapped_column(String, nullable=True)  # 'v1.0-trainval' 等のフォルダ名
    dataroot:     Mapped[str] = mapped_column(String, nullable=False)  # ファイルルートパス（settings.DATA_ROOTからの相対パス）
    description:  Mapped[str | None] = mapped_column(String, nullable=True)
    created_at:   Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
