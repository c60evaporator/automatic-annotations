"""SampleAnnotation（3D、global 座標）をカメラ画像上の 2D BBox へ投影する.

DB の各値の意味:
  SampleAnnotation.translation / rotation / size … global 座標のボックス
      size は nuScenes の並びで ``[width, length, height]``
  frame["ego_pose"]                              … ego → global
  frame["calibrated_sensor"]                     … camera → ego
"""
from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.services.geometry.detection import convert_global_box_to_2d_box
from app.services.label_service import nusc_category_to_label

logger = get_logger(__name__)

# 小さすぎる投影結果は捨てる（画面端をかすめただけのボックス）
MIN_BOX_SIZE_PX = 4.0


def project_annotations_to_frame(
    annotations: list[dict[str, Any]],
    frame: dict[str, Any],
    *,
    min_size: float = MIN_BOX_SIZE_PX,
) -> list[dict[str, Any]]:
    """1フレーム（1カメラ）に写るアノテーションを 2D BBox にして返す.

    Args:
        annotations: AnnotationRepository.list_by_sample() の結果
        frame: SensorRepository.list_frames_by_sample() の1要素

    Returns:
        det2d_viewer が扱えるボックス dict のリスト。
        label は検出ラベル空間へ寄せる（GT と推論結果で色を揃えるため）。
    """
    calib = frame.get("calibrated_sensor") or {}
    ego = frame.get("ego_pose") or {}
    intrinsic = calib.get("camera_intrinsic")
    width, height = frame.get("width"), frame.get("height")

    if intrinsic is None or not width or not height:
        # LiDAR など、カメラでない sample_data には投影できない
        return []

    boxes: list[dict[str, Any]] = []
    for ann in annotations:
        try:
            xyxy = convert_global_box_to_2d_box(
                center_global=ann["translation"],
                size_wlh=ann["size"],
                rotation_global=ann["rotation"],
                ego_translation=ego["translation"],
                ego_quaternion=ego["rotation"],
                camera_translation=calib["translation"],
                camera_quaternion=calib["rotation"],
                camera_intrinsic=intrinsic,
                image_width=width,
                image_height=height,
                min_size=min_size,
            )
        except (KeyError, ValueError) as exc:
            logger.warning("failed to project annotation %s: %s",
                           ann.get("token"), exc)
            continue

        if xyxy is None:
            # カメラ背後・画角外・小さすぎる
            continue

        category = (ann.get("instance") or {}).get("category_name", "")
        # 検出ラベルへ寄せることで、凡例の色が GT と推論で一致する。
        # 対応が無いカテゴリは元の名前のまま出す（色は md5 で決まる）
        label = nusc_category_to_label(category) or category

        xmin, ymin, xmax, ymax = xyxy
        boxes.append({
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "label": label,
            "category_name": category,
            "score": None,
            "source": ann.get("source"),
            "annotation_token": ann.get("token"),
            "instance_token": (ann.get("instance") or {}).get("token"),
        })
    return boxes
