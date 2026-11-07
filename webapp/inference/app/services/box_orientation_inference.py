"""向き推定モデルの実行（切り出し → 推論 → ego の yaw への変換）.

向き推定ジョブ（routers/box_orientation.py）と Box Fitting のパス 2
（routers/depth_boxfitting.py）の両方から使う。グループの形は
``services.box_orientation.group_candidates`` が返すものと同じ。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core import models as model_registry
from app.core.logging import get_logger
from app.services.box_orientation import (
    MODEL_PART_HOE,
    ORIENTATION_STATUS_ESTIMATED,
    ORIENTATION_STATUS_FAILED,
    camera_relative_to_yaw,
    orient_anything_to_view_angle,
)

logger = get_logger(__name__)

# 1 回にまとめて切り出すクロップ数（モデルへの入力は各モデルの batch_size で分ける）
BATCH_SIZE = 32


def result_base(group: dict[str, Any]) -> dict[str, Any]:
    primary = group["primary"] or group["source"]
    return {
        "box_fitting_id": primary["box_fitting_id"],
        "global_track_id": primary["global_track_id"],
        "sample_token": primary["sample_token"],
        "label": group["label"],
        "method": group["method"],
    }


def _crop(model: str, image: np.ndarray, source: dict[str, Any]) -> np.ndarray:
    """モデルごとの切り出し."""
    if model == MODEL_PART_HOE:
        from app.models_impl.part_hoe import crop_person

        return crop_person(image, source["bbox"])
    from app.models_impl.orient_anything_v1 import crop_object_on_white
    from common.mask_rle import decode_rle, resize_mask_nearest

    mask = None
    if source.get("mask_rle"):
        mask = decode_rle(source["mask_rle"])
        if mask.shape != image.shape[:2]:
            mask = resize_mask_nearest(mask, *image.shape[:2])
    return crop_object_on_white(image, source["bbox"], mask)


def _predict(model: str, estimator: Any, crops: list[np.ndarray]) -> list[dict[str, Any]]:
    """モデルごとの推論. angle_deg はモデルの出力、view_deg は視線基準の角度."""
    if model == MODEL_PART_HOE:
        probs = estimator.predict(crops)
        return [
            {"angle_deg": float(angle), "view_deg": float(angle),
             "confidence": float(prob.max())}
            for prob, angle in zip(probs, estimator.to_degrees(probs), strict=True)
        ]
    out = estimator.predict(crops)
    return [
        {"angle_deg": float(az), "view_deg": orient_anything_to_view_angle(float(az)),
         "confidence": float(conf)}
        for az, conf in zip(out["azimuth_deg"], out["confidence"], strict=True)
    ]


def estimate_groups(
    model: str,
    groups: list[dict[str, Any]],
    dataroot: Path,
    emit: Callable[[dict[str, Any]], None],
    *,
    cancelled: Callable[[], bool] = lambda: False,
    on_batch: Callable[[int], None] = lambda n: None,
) -> None:
    """1 つのモデルでまとめて推定する. 画像はフレームごとに 1 回だけ読む.

    グループの source（推定に使うカメラ）は filename / bbox / mask_rle /
    calibrated_sensor / ego_pose / reference_ego_pose を、primary（ボックスを持つ行）は
    center_ego を持つこと。結果はグループごとに ``emit`` へ渡す。
    """
    from app.services.lshape_filter import sensor_origin_xy

    # 同じフレームのグループを続けて処理し、画像を使い回す
    groups = sorted(groups, key=lambda g: g["source"]["sample_data_token"])
    with model_registry.use_gpu(model) as estimator:
        image_cache: dict[str, np.ndarray] = {}
        for start in range(0, len(groups), BATCH_SIZE):
            if cancelled():
                break
            batch = groups[start:start + BATCH_SIZE]
            crops, valid = [], []
            for group in batch:
                source = group["source"]
                try:
                    path = source["filename"]
                    if path not in image_cache:
                        image_cache.clear()  # 直前のフレームだけ持つ
                        with Image.open(dataroot / path) as img:
                            image_cache[path] = np.asarray(img.convert("RGB"))
                    crops.append(_crop(model, image_cache[path], source))
                    valid.append(group)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("crop failed for %s: %s", source["box_fitting_id"], exc)
                    emit({**result_base(group), "status": ORIENTATION_STATUS_FAILED,
                          "error": f"{type(exc).__name__}: {exc}"})
            if crops:
                for group, pred in zip(valid, _predict(model, estimator, crops), strict=True):
                    source, primary = group["source"], group["primary"]
                    camera_xy = sensor_origin_xy(
                        source["calibrated_sensor"], source["ego_pose"],
                        source.get("reference_ego_pose"),
                    )
                    center = primary["center_ego"]
                    view_deg = pred.pop("view_deg")
                    emit({
                        **result_base(group),
                        **pred,
                        "status": ORIENTATION_STATUS_ESTIMATED,
                        "source_box_fitting_id": source["box_fitting_id"],
                        "channel": source["channel"],
                        "bbox": list(source["bbox"]),
                        "yaw_ego": camera_relative_to_yaw(
                            view_deg, camera_xy, (center[0], center[1])
                        ),
                    })
            on_batch(len(batch))


# ── Box Fitting のパス 2 からの呼び出し ──────────────────────────────────

# 結果の記録先（代表行の fit_metrics のキー。webapp の ORIENTATION_METRICS_KEY と同じ）
ORIENTATION_METRICS_KEY = "orientation_estimation"


def orientation_settings(box_fitting_params: dict[str, Any] | None) -> dict[str, Any]:
    """run のパラメータから向き推定の設定を取り出す（無ければ推定しない）."""
    from app.services.box_orientation import DEFAULT_SWITCH_LENGTH_RATIO

    params = (box_fitting_params or {}).get("orientation") or {}
    return {
        "methods": {k: v for k, v in (params.get("methods") or {}).items() if v},
        "switch_length_ratio": float(
            params.get("switch_length_ratio", DEFAULT_SWITCH_LENGTH_RATIO)
        ),
        # (length, width, height)
        "box_sizes": (box_fitting_params or {}).get("box_sizes") or {},
    }


def orient_fitted_groups(
    fitted_groups: list[dict[str, Any]],
    *,
    box_fitting_params: dict[str, Any] | None,
    frame_by_token: dict[str, dict[str, Any]],
    mask_by_instance: dict[str, dict[str, Any]],
    reference_ego_poses: dict[str, dict[str, Any]] | None,
    dataroot: Path,
    cancelled: Callable[[], bool] = lambda: False,
    on_model: Callable[[str], None] = lambda model: None,
) -> dict[str, int]:
    """当てはめたボックスの向きを AI モデルで推定し、代表行へ反映する.

    Args:
        fitted_groups: merge_and_fit が渡すグループ（primary はボックスを持つ行、
            members は同じ global_track_id × sample の全カメラの行）
        frame_by_token: sample_data_token → カメラフレーム（filename /
            channel / calibrated_sensor / ego_pose）
        mask_by_instance: instance_tracking_2d_id → トラッキングのマスク（RLE）

    推定に使うカメラは、点数（深度 + LiDAR の実点数。深度を使わなかった
    カメラも深度の点数を数える）が最大のもの。結果は代表行の
    ``fit_metrics["orientation_estimation"]`` に残す。
    """
    from common.mask_rle import rle_bbox

    from app.services.box_orientation import (
        METHODS_NEEDING_MASK,
        MODEL_FOR_METHOD,
        ORIENTATION_STATUS_UNSUPPORTED,
        SUPPORTED_ORIENTATION_METHODS,
        apply_orientation,
    )

    settings = orientation_settings(box_fitting_params)
    methods = settings["methods"]
    stats = {"num_targets": 0, "num_applied": 0}
    if not methods:
        return stats

    def record(primary: dict[str, Any], values: dict[str, Any]) -> None:
        primary.setdefault("fit_metrics", {})[ORIENTATION_METRICS_KEY] = values

    primary_by_id: dict[str, dict[str, Any]] = {}
    by_model: dict[str, list[dict[str, Any]]] = {}
    for group in fitted_groups:
        method = methods.get(group["label"] or "")
        if not method:
            continue
        primary = group["primary"]
        stats["num_targets"] += 1
        if method not in SUPPORTED_ORIENTATION_METHODS:
            record(primary, {"method": method, "status": ORIENTATION_STATUS_UNSUPPORTED})
            continue
        # 点数が同じなら代表行を優先する（並び順で結果が変わらないように）
        source = max(
            group["members"],
            key=lambda m: (int(m.get("num_points_depth") or 0)
                           + int(m.get("num_points_lidar") or 0),
                           m is primary, m["instance_tracking_2d_id"]),
        )
        frame = frame_by_token.get(source["sample_data_token"])
        mask = mask_by_instance.get(source["instance_tracking_2d_id"])
        bbox = rle_bbox(mask) if mask else (0, 0, 0, 0)
        if frame is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            record(primary, {"method": method, "status": ORIENTATION_STATUS_FAILED,
                             "error": "画像またはマスクがありません"})
            continue
        primary_id = primary["instance_tracking_2d_id"]
        primary_by_id[primary_id] = primary
        by_model.setdefault(MODEL_FOR_METHOD[method], []).append({
            "method": method, "label": group["label"],
            "primary": {
                "box_fitting_id": primary_id,
                "global_track_id": group["global_track_id"],
                "sample_token": group["sample_token"],
                "center_ego": primary["center_ego"],
            },
            "source": {
                "box_fitting_id": source["instance_tracking_2d_id"],
                "sample_data_token": source["sample_data_token"],
                "channel": frame["channel"],
                "filename": frame["filename"],
                "bbox": list(bbox),
                "mask_rle": mask if method in METHODS_NEEDING_MASK else None,
                "calibrated_sensor": frame["calibrated_sensor"],
                "ego_pose": frame["ego_pose"],
                "reference_ego_pose": (reference_ego_poses or {}).get(
                    group["sample_token"]),
            },
        })

    def emit(result: dict[str, Any]) -> None:
        primary = primary_by_id[result["box_fitting_id"]]
        values = {
            "method": result["method"],
            "status": result["status"],
            # 推定に使ったカメラのインスタンス（box_fittings の行はまだ無い）
            "source_instance_tracking_2d_id": result.get("source_box_fitting_id"),
            **{key: result.get(key) for key in (
                "channel", "bbox", "angle_deg", "confidence", "yaw_ego", "error")},
        }
        if result["status"] == ORIENTATION_STATUS_ESTIMATED:
            size = settings["box_sizes"].get(result["label"] or "")
            values.update(apply_orientation(
                primary, result["method"], float(result["yaw_ego"]),
                standard_width=float(size[1]) if size and len(size) >= 2 else None,
                switch_length_ratio=settings["switch_length_ratio"],
            ))
            stats["num_applied"] += 1
        record(primary, values)

    # モデルごとに載せ替える（MAX_RESIDENT_MODELS=1 でも 1 回ずつで済む）
    for model, groups in by_model.items():
        if cancelled():
            break
        on_model(model)
        estimate_groups(model, groups, dataroot, emit, cancelled=cancelled)
    return stats
