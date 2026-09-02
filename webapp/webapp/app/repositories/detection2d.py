"""2D 物体検出の実行単位（run）と結果のクエリ.

run = Detection2DParams の 1 行。結果の Detection2D 行はこれに紐づく。

保持ポリシーは「追加（履歴保持）＋ 上限プルーニング」。
上書きにしない理由は、InstanceTracking2DParams.detection_2d_params_id が
ON DELETE CASCADE で参照しているため、run を消すと
それを入力にしたトラッキング結果と 3D ボックスまで連鎖削除されるから。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, desc, func, insert, select, update
from sqlalchemy.orm import Session

from app.models.ann_intermediate import (
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    Detection2D,
    Detection2DParams,
    InstanceTracking2DParams,
)

# 一度に INSERT する行数
CHUNK_SIZE = 2_000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Detection2DRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ── run の作成・更新 ──────────────────────────────────────────────────

    def create_run(
        self,
        dataset_id: str,
        scene_token: str,
        *,
        model_name: str,
        sample_interval: int,
        nusc_category_to_label: dict[str, str],
        label_to_nusc_category: dict[str, str],
        label_to_category_group: dict[str, str],
        score_threshold: dict[str, float],
        nms_same_class_ious: dict[str, float],
        nms_cross_class_ious: dict[str, float],
        status: str = RUN_STATUS_RUNNING,
    ) -> str:
        """run を作成して id を返す.

        ラベル体系のマッピングは**実行時点の config をそのまま保存する**。
        後から config を変えても、過去の run の解釈が変わらないようにするため。
        """
        params_id = str(uuid.uuid4())
        self.session.execute(insert(Detection2DParams.__table__), [{
            "id": params_id,
            "dataset_id": dataset_id,
            "scene_token": scene_token,
            "model_name": model_name,
            "sample_interval": sample_interval,
            "nusc_category_to_label": nusc_category_to_label,
            "label_to_nusc_category": label_to_nusc_category,
            "label_to_category_group": label_to_category_group,
            "score_threshold": score_threshold,
            "nms_same_class_ious": nms_same_class_ious,
            "nms_cross_class_ious": nms_cross_class_ious,
            "status": status,
            "num_inferences": 0,
            "started_at": _utcnow(),
        }])
        return params_id

    def finish_run(
        self,
        params_id: str,
        *,
        status: str,
        num_inferences: int = 0,
        inference_time: float | None = None,
    ) -> None:
        """run を終了状態にする."""
        self.session.execute(
            update(Detection2DParams.__table__)
            .where(Detection2DParams.__table__.c.id == params_id)
            .values(
                status=status,
                num_inferences=num_inferences,
                inference_time=inference_time,
                ended_at=_utcnow(),
            )
        )

    # ── 結果の保存 ────────────────────────────────────────────────────────

    def save_boxes(
        self,
        params_id: str,
        dataset_id: str,
        boxes_by_frame: dict[str, list[dict[str, Any]]],
    ) -> int:
        """検出結果をまとめて保存する.

        Args:
            boxes_by_frame: {sample_data_token: [box, ...]}
                box は xmin/ymin/xmax/ymax/label/score と、
                任意で manually_modified を持つ

        Returns:
            保存した行数。

        NOTE: ORM ではなく Core の Table に対して executemany する。
        1 run で数千行になるため、ORM オブジェクト生成のコストを避ける。
        """
        rows: list[dict[str, Any]] = []
        for sample_data_token, boxes in boxes_by_frame.items():
            for box in boxes:
                rows.append({
                    "id": str(uuid.uuid4()),
                    "dataset_id": dataset_id,
                    "sample_data_token": sample_data_token,
                    "detection_2d_params_id": params_id,
                    "xmin": int(box["xmin"]), "ymin": int(box["ymin"]),
                    "xmax": int(box["xmax"]), "ymax": int(box["ymax"]),
                    "label": box["label"],
                    "score": box.get("score"),
                    "manually_modified": bool(box.get("manually_modified", False)),
                })

        if not rows:
            return 0

        stmt = insert(Detection2D.__table__)
        for i in range(0, len(rows), CHUNK_SIZE):
            self.session.execute(stmt, rows[i:i + CHUNK_SIZE])
        return len(rows)

    # ── run の参照 ────────────────────────────────────────────────────────

    def get_run(self, params_id: str) -> dict[str, Any] | None:
        stmt = select(Detection2DParams).where(Detection2DParams.id == params_id)
        row = self.session.scalars(stmt).first()
        return _run_to_dict(row) if row else None

    def list_runs(
        self, dataset_id: str, scene_token: str, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        """シーンの run 一覧を新しい順で返す（UI の run セレクタ用）.

        各 run が持つ検出数と、後段から参照されているかも返す。
        参照されている run を削除するとトラッキング結果まで消えるため、
        UI 側で警告できるようにしておく。
        """
        nbr_boxes = (
            select(func.count())
            .select_from(Detection2D)
            .where(Detection2D.detection_2d_params_id == Detection2DParams.id)
            .scalar_subquery()
        )
        nbr_tracking = (
            select(func.count())
            .select_from(InstanceTracking2DParams)
            .where(
                InstanceTracking2DParams.detection_2d_params_id
                == Detection2DParams.id
            )
            .scalar_subquery()
        )
        stmt = (
            select(
                Detection2DParams,
                nbr_boxes.label("nbr_boxes"),
                nbr_tracking.label("nbr_tracking_runs"),
            )
            .where(
                Detection2DParams.dataset_id == dataset_id,
                Detection2DParams.scene_token == scene_token,
            )
            .order_by(desc(Detection2DParams.started_at))
        )
        if status is not None:
            stmt = stmt.where(Detection2DParams.status == status)

        out: list[dict[str, Any]] = []
        for row, nbr, refs in self.session.execute(stmt):
            data = _run_to_dict(row)
            data["nbr_boxes"] = nbr
            data["nbr_tracking_runs"] = refs
            out.append(data)
        return out

    def resolve_display_run(
        self, dataset_id: str, scene_token: str
    ) -> tuple[str | None, str]:
        """初期表示に使う run を決める.

        優先順:
          1. 最新の成功した Instance Tracking が参照している Detection2D run
          2. 最新の成功した Detection2D run
          3. なし

        Returns:
            (params_id, 理由を表す文字列)。該当なしなら (None, "none")。
        """
        # ① トラッキングの参照先。
        #    トラッキングを回した後にパラメータ違いの検出を試した場合でも、
        #    「いま後段が使っている入力」を見せるほうが辻褄が合う
        referenced = self.session.scalar(
            select(InstanceTracking2DParams.detection_2d_params_id)
            .where(
                InstanceTracking2DParams.scene_token == scene_token,
                InstanceTracking2DParams.status == RUN_STATUS_SUCCEEDED,
            )
            .order_by(desc(InstanceTracking2DParams.started_at))
            .limit(1)
        )
        if referenced is not None:
            # 参照先が削除済みでないことを確認する
            exists = self.session.scalar(
                select(Detection2DParams.id).where(
                    Detection2DParams.id == referenced
                )
            )
            if exists is not None:
                return referenced, "tracking"

        # ② 最新の成功した検出 run
        latest = self.session.scalar(
            select(Detection2DParams.id)
            .where(
                Detection2DParams.dataset_id == dataset_id,
                Detection2DParams.scene_token == scene_token,
                Detection2DParams.status == RUN_STATUS_SUCCEEDED,
            )
            .order_by(desc(Detection2DParams.started_at))
            .limit(1)
        )
        if latest is not None:
            return latest, "latest"

        return None, "none"

    # ── 結果の参照 ────────────────────────────────────────────────────────

    def list_boxes_by_run(
        self, params_id: str, *, manual_only: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """run の検出結果を {sample_data_token: [box, ...]} で返す.

        Args:
            manual_only: True なら手修正されたボックスだけ返す
                （再実行時の引き継ぎに使う）
        """
        stmt = select(
            Detection2D.id,
            Detection2D.sample_data_token,
            Detection2D.xmin, Detection2D.ymin,
            Detection2D.xmax, Detection2D.ymax,
            Detection2D.label, Detection2D.score,
            Detection2D.manually_modified,
        ).where(Detection2D.detection_2d_params_id == params_id)
        if manual_only:
            stmt = stmt.where(Detection2D.manually_modified.is_(True))

        result: dict[str, list[dict[str, Any]]] = {}
        for r in self.session.execute(stmt).mappings():
            result.setdefault(r["sample_data_token"], []).append({
                "id": r["id"],
                "xmin": r["xmin"], "ymin": r["ymin"],
                "xmax": r["xmax"], "ymax": r["ymax"],
                "label": r["label"], "score": r["score"],
                "manually_modified": r["manually_modified"],
            })
        return result

    # ── 削除・プルーニング ────────────────────────────────────────────────

    def delete_run(self, params_id: str) -> None:
        """run を削除する.

        Detection2D は CASCADE で消える。
        この run を参照するトラッキング run も CASCADE で消えるため、
        UI では削除前に nbr_tracking_runs を見せて確認を取ること。
        """
        self.session.execute(
            delete(Detection2DParams.__table__)
            .where(Detection2DParams.__table__.c.id == params_id)
        )

    def prune_runs(
        self, dataset_id: str, scene_token: str, *, keep: int
    ) -> list[str]:
        """古い run を削除して保持数を上限以内に収める.

        後段（Instance Tracking）から参照されている run と、
        実行中の run は削除対象から除外する。
        参照されている run を消すと、トラッキング結果と 3D ボックスまで
        CASCADE で消えてしまうため。

        Returns:
            削除した run の id。
        """
        runs = self.list_runs(dataset_id, scene_token)
        deletable = [
            r for r in runs
            if r["nbr_tracking_runs"] == 0 and r["status"] != RUN_STATUS_RUNNING
        ]
        # 新しい順に keep 件を残し、それより古い削除可能な run を消す
        protected_ids = {r["id"] for r in runs[:keep]}
        targets = [r["id"] for r in deletable if r["id"] not in protected_ids]
        for params_id in targets:
            self.delete_run(params_id)
        return targets


def _run_to_dict(row: Detection2DParams) -> dict[str, Any]:
    return {
        "id": row.id,
        "dataset_id": row.dataset_id,
        "scene_token": row.scene_token,
        "model_name": row.model_name,
        "sample_interval": row.sample_interval,
        "nusc_category_to_label": row.nusc_category_to_label,
        "label_to_nusc_category": row.label_to_nusc_category,
        "label_to_category_group": row.label_to_category_group,
        "score_threshold": row.score_threshold,
        "nms_same_class_ious": row.nms_same_class_ious,
        "nms_cross_class_ious": row.nms_cross_class_ious,
        "status": row.status,
        "num_inferences": row.num_inferences,
        "inference_time": row.inference_time,
        "started_at": row.started_at,
        "ended_at": row.ended_at,
    }
