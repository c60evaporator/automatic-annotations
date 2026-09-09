"""GT アノテーション（global 座標の 3D ボックス）を BEV 用に変換する.

Box Fitting の結果は ego 座標なので、比較するには GT も ego へ揃える。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.services.geometry.pointcloud import transform_global_to_ego
from app.services.geometry.transform import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_yaw,
)
from app.services.label_service import nusc_category_to_label

logger = get_logger(__name__)


def gt_boxes_in_ego(
    annotations: list[dict[str, Any]], ego_pose: dict[str, Any]
) -> list[dict[str, Any]]:
    """GT ボックスを ego 座標の BEV ボックスへ変換する.

    Args:
        annotations: AnnotationRepository.list_by_sample() の結果
        ego_pose: その sample の EgoPose（ego → global）

    Returns:
        ``{center_ego, size_wlh, yaw_ego, label, category_name, token}`` のリスト。

    ラベルは検出ラベル空間へ寄せる（GT と推定で色を揃えるため）。
    """
    if not annotations or not ego_pose:
        return []

    ego_quaternion = normalize_quaternion(ego_pose["rotation"])
    centers = np.asarray(
        [a["translation"] for a in annotations], dtype=np.float64
    )
    centers_ego = transform_global_to_ego(
        centers, ego_pose["translation"], ego_pose["rotation"]
    )

    boxes: list[dict[str, Any]] = []
    for annotation, center_ego in zip(annotations, centers_ego, strict=True):
        # global の回転を ego 座標での回転へ直してから yaw を取り出す。
        # global の yaw をそのまま使うと、自車の向きぶんずれる
        rotation_ego = quaternion_multiply(
            quaternion_conjugate(ego_quaternion), annotation["rotation"]
        )
        category = (annotation.get("instance") or {}).get("category_name", "")
        boxes.append({
            "token": annotation.get("token"),
            "center_ego": [float(v) for v in center_ego],
            "size_wlh": [float(v) for v in annotation["size"]],
            "yaw_ego": float(quaternion_to_yaw(rotation_ego)),
            "label": nusc_category_to_label(category) or category,
            "category_name": category,
        })
    return boxes
