"""GT アノテーション（global 座標の 3D ボックス）を BEV 用に変換する.

Box Fitting の結果は ego 座標なので、比較するには GT も ego へ揃える。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.services.geometry.pointcloud import (
    transform_ego_to_global,
    transform_global_to_ego,
)
from app.services.geometry.transform import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_yaw,
    yaw_to_quaternion,
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


def rebase_boxes_to_ego(
    boxes: list[dict[str, Any]],
    from_ego_pose: dict[str, Any] | None,
    to_ego_pose: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """ある ego 座標のボックスを、別時刻の ego 座標へ移す.

    Box Fitting のボックスは基準センサーの ego 座標にある。カメラへ投影するときは
    そのカメラの ego_pose に揃えないと、時刻差ぶん（自車の移動量）ずれる。
    どちらかの姿勢が無ければそのまま返す。元の dict は変更しない。
    """
    if not boxes or not from_ego_pose or not to_ego_pose:
        return list(boxes)

    centers = np.asarray([b["center_ego"] for b in boxes], dtype=np.float64)
    centers_to = transform_global_to_ego(
        transform_ego_to_global(
            centers, from_ego_pose["translation"], from_ego_pose["rotation"]
        ),
        to_ego_pose["translation"], to_ego_pose["rotation"],
    )
    # from → global → to の回転を yaw に掛けて取り出す
    relative = quaternion_multiply(
        quaternion_conjugate(normalize_quaternion(to_ego_pose["rotation"])),
        normalize_quaternion(from_ego_pose["rotation"]),
    )
    rebased: list[dict[str, Any]] = []
    for box, center in zip(boxes, centers_to, strict=True):
        rotation = quaternion_multiply(
            relative, yaw_to_quaternion(box.get("yaw_ego") or 0.0)
        )
        rebased.append({
            **box,
            "center_ego": [float(v) for v in center],
            "yaw_ego": float(quaternion_to_yaw(rotation)),
        })
    return rebased
