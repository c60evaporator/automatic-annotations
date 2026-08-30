"""データセット単位のクエリ（DB アクセスの抽象化）.

Repository 層は ORM インスタンスではなく dict を返す。
Streamlit の @st.cache_data は戻り値を保持するため、ORM オブジェクトを
返すと Session が閉じた後に DetachedInstanceError になる。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.annotation import SampleAnnotation
from app.models.dataset import Dataset
from app.models.scene import Sample, Scene


class DatasetRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list(self) -> list[dict[str, Any]]:
        """データセット一覧を、シーン数・サンプル数・アノテーション数付きで返す.

        相関サブクエリで件数を出しているのは、複数の LEFT JOIN + GROUP BY を
        重ねると行が掛け算されて件数が壊れるため。
        """
        n_scenes = (
            select(func.count())
            .select_from(Scene)
            .where(Scene.dataset_id == Dataset.id)
            .scalar_subquery()
        )
        n_samples = (
            select(func.count())
            .select_from(Sample)
            .where(Sample.dataset_id == Dataset.id)
            .scalar_subquery()
        )
        n_annotations = (
            select(func.count())
            .select_from(SampleAnnotation)
            .where(SampleAnnotation.dataset_id == Dataset.id)
            .scalar_subquery()
        )
        stmt = (
            select(
                Dataset.id,
                Dataset.name,
                Dataset.dataset_type,
                Dataset.version,
                Dataset.dataroot,
                Dataset.description,
                Dataset.created_at,
                n_scenes.label("nbr_scenes"),
                n_samples.label("nbr_samples"),
                n_annotations.label("nbr_annotations"),
            )
            .order_by(Dataset.created_at.desc())
        )
        return [dict(r) for r in self.session.execute(stmt).mappings()]

    def get(self, dataset_id: str) -> dict[str, Any] | None:
        stmt = select(
            Dataset.id,
            Dataset.name,
            Dataset.dataset_type,
            Dataset.version,
            Dataset.dataroot,
            Dataset.description,
            Dataset.created_at,
        ).where(Dataset.id == dataset_id)
        row = self.session.execute(stmt).mappings().first()
        return dict(row) if row else None

    def exists(self, dataset_id: str) -> bool:
        return self.session.scalar(
            select(Dataset.id).where(Dataset.id == dataset_id)
        ) is not None

    def delete(self, dataset_id: str) -> None:
        """データセットを削除する.

        全テーブルが dataset_id で ON DELETE CASCADE しているため、
        この1行の削除で配下のデータもまとめて消える。
        """
        self.session.execute(delete(Dataset).where(Dataset.id == dataset_id))
