"""アノテーション単位のクエリ."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models.annotation import (
    Attribute,
    Category,
    Instance,
    SampleAnnotation,
    Visibility,
    annotation_attribute,
)
from app.models.scene import Sample


def _row_to_annotation(r: Any) -> dict[str, Any]:
    return {
        "token": r["ann_token"],
        "sample_token": r["sample_token"],
        "translation": r["translation"],
        "rotation": r["rotation"],
        "size": r["size"],
        "num_lidar_pts": r["num_lidar_pts"],
        "num_radar_pts": r["num_radar_pts"],
        "visibility_token": r["visibility_token"],
        "visibility_level": r["visibility_level"],
        "source": r["ann_source"],
        "score": r["score"],
        "manually_modified": r["manually_modified"],
        "depth_estimation_params_id": r["depth_estimation_params_id"],
        "prev": r["ann_prev"],
        "next": r["ann_next"],
        "instance": {
            "token": r["instance_token"],
            "category_token": r["category_token"],
            "category_name": r["category_name"],
            "nbr_annotations": r["nbr_annotations"],
            "source": r["inst_source"],
        },
        "attribute_names": [],
    }


class AnnotationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _annotation_stmt(self) -> Select:
        """sample_annotation + instance + category + visibility の JOIN.

        attribute は多対多なので、ここで JOIN すると行が掛け算されて
        アノテーションが重複する。別クエリで取って後から束ねる。
        """
        return (
            select(
                SampleAnnotation.token.label("ann_token"),
                SampleAnnotation.sample_token,
                SampleAnnotation.translation,
                SampleAnnotation.rotation,
                SampleAnnotation.size,
                SampleAnnotation.num_lidar_pts,
                SampleAnnotation.num_radar_pts,
                SampleAnnotation.visibility_token,
                SampleAnnotation.source.label("ann_source"),
                SampleAnnotation.score,
                SampleAnnotation.manually_modified,
                SampleAnnotation.depth_estimation_params_id,
                SampleAnnotation.prev.label("ann_prev"),
                SampleAnnotation.next.label("ann_next"),
                Instance.token.label("instance_token"),
                Instance.category_token,
                Instance.nbr_annotations,
                Instance.source.label("inst_source"),
                Category.name.label("category_name"),
                Visibility.level.label("visibility_level"),
            )
            .join(Instance, Instance.token == SampleAnnotation.instance_token)
            .join(Category, Category.token == Instance.category_token)
            .outerjoin(Visibility, Visibility.token == SampleAnnotation.visibility_token)
        )

    def _attach_attributes_by_sample(
        self, annotations: list[dict[str, Any]], sample_token: str
    ) -> None:
        stmt = (
            select(annotation_attribute.c.annotation_token, Attribute.name)
            .join(Attribute, Attribute.token == annotation_attribute.c.attribute_token)
            .join(SampleAnnotation,
                  SampleAnnotation.token == annotation_attribute.c.annotation_token)
            .where(SampleAnnotation.sample_token == sample_token)
        )
        self._merge_attributes(annotations, stmt)

    def _attach_attributes_by_scene(
        self, annotations: list[dict[str, Any]], scene_token: str
    ) -> None:
        # IN 句に大量の token を並べる代わりに samples まで JOIN で辿る。
        # シーン内アノテーションが数千件あってもバインド変数が増えない
        stmt = (
            select(annotation_attribute.c.annotation_token, Attribute.name)
            .join(Attribute, Attribute.token == annotation_attribute.c.attribute_token)
            .join(SampleAnnotation,
                  SampleAnnotation.token == annotation_attribute.c.annotation_token)
            .join(Sample, Sample.token == SampleAnnotation.sample_token)
            .where(Sample.scene_token == scene_token)
        )
        self._merge_attributes(annotations, stmt)

    def _merge_attributes(self, annotations: list[dict[str, Any]], stmt: Select) -> None:
        by_token: dict[str, list[str]] = defaultdict(list)
        for token, name in self.session.execute(stmt):
            by_token[token].append(name)
        for ann in annotations:
            ann["attribute_names"] = by_token.get(ann["token"], [])

    def list_by_sample(
        self,
        sample_token: str,
        *,
        source: str | None = None,
        include_attributes: bool = True,
    ) -> list[dict[str, Any]]:
        """1 sample のアノテーションを instance 情報付きで返す.

        Args:
            source: 'imported' / 'auto' / 'manual' で絞る（None なら全件）
        """
        stmt = self._annotation_stmt().where(
            SampleAnnotation.sample_token == sample_token
        )
        if source is not None:
            stmt = stmt.where(SampleAnnotation.source == source)
        stmt = stmt.order_by(Category.name, SampleAnnotation.token)

        annotations = [_row_to_annotation(r) for r in self.session.execute(stmt).mappings()]
        if include_attributes and annotations:
            self._attach_attributes_by_sample(annotations, sample_token)
        return annotations

    def list_by_scene(
        self,
        scene_token: str,
        *,
        source: str | None = None,
        include_attributes: bool = True,
    ) -> list[dict[str, Any]]:
        """シーン内の全アノテーションを sample の時刻順でまとめて返す.

        sample ごとに list_by_sample を呼ぶとクエリ数が sample 数だけ増える。
        トラッキング表示や一括評価ではこちらを使う。

        戻り値には sample_idx（シーン内での sample 連番）を付与する。
        """
        stmt = (
            self._annotation_stmt()
            .add_columns(Sample.timestamp.label("sample_timestamp"))
            .join(Sample, Sample.token == SampleAnnotation.sample_token)
            .where(Sample.scene_token == scene_token)
        )
        if source is not None:
            stmt = stmt.where(SampleAnnotation.source == source)
        stmt = stmt.order_by(Sample.timestamp, Category.name, SampleAnnotation.token)

        annotations: list[dict[str, Any]] = []
        idx_of: dict[str, int] = {}
        for r in self.session.execute(stmt).mappings():
            ann = _row_to_annotation(r)
            st = ann["sample_token"]
            if st not in idx_of:
                idx_of[st] = len(idx_of)
            ann["sample_idx"] = idx_of[st]
            ann["sample_timestamp"] = r["sample_timestamp"]
            annotations.append(ann)

        if include_attributes and annotations:
            self._attach_attributes_by_scene(annotations, scene_token)
        return annotations

    def list_categories(
        self, dataset_id: str, *, include_counts: bool = False
    ) -> list[dict[str, Any]]:
        """データセット内のカテゴリ一覧を返す.

        Detection2D ページのラベルマッピング設定
        （nusc_category_to_label など）を組み立てるのに使う。

        Args:
            include_counts: True なら instance 数と annotation 数を付ける。
                件数は相関サブクエリで数えるため、カテゴリ数ぶんの集計が走る。
                nuScenes のカテゴリは 20〜30 程度なので実用上は問題ないが、
                単に選択肢を並べたいだけなら False のままにする。
        """
        columns = [Category.token, Category.name, Category.description]

        if include_counts:
            nbr_instances = (
                select(func.count())
                .select_from(Instance)
                .where(Instance.category_token == Category.token)
                .scalar_subquery()
            )
            nbr_annotations = (
                select(func.count())
                .select_from(SampleAnnotation)
                .join(Instance, Instance.token == SampleAnnotation.instance_token)
                .where(Instance.category_token == Category.token)
                .scalar_subquery()
            )
            columns += [
                nbr_instances.label("nbr_instances"),
                nbr_annotations.label("nbr_annotations"),
            ]

        stmt = (
            select(*columns)
            .where(Category.dataset_id == dataset_id)
            .order_by(Category.name)
        )
        return [dict(r) for r in self.session.execute(stmt).mappings()]

    def list_instances_by_scene(
        self, scene_token: str, *, source: str | None = None
    ) -> list[dict[str, Any]]:
        """シーンに登場する instance の一覧（トラック単位の表示用）.

        同じ instance がシーン内の複数 sample に現れるため DISTINCT する。
        """
        stmt = (
            select(
                Instance.token,
                Instance.category_token,
                Category.name.label("category_name"),
                Instance.nbr_annotations,
                Instance.first_annotation_token,
                Instance.last_annotation_token,
                Instance.source,
            )
            .join(SampleAnnotation, SampleAnnotation.instance_token == Instance.token)
            .join(Sample, Sample.token == SampleAnnotation.sample_token)
            .join(Category, Category.token == Instance.category_token)
            .where(Sample.scene_token == scene_token)
        )
        if source is not None:
            stmt = stmt.where(Instance.source == source)
        stmt = stmt.distinct().order_by(Category.name, Instance.token)
        return [dict(r) for r in self.session.execute(stmt).mappings()]
