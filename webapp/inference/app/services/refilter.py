"""保存済みの深度マップから、インスタンス点群を再生成して再フィルタする.

パラメータ調整のたびに数分のパイプラインを回し直さずに済むようにする。
点群は「深度マップをクロージング後マスクで切り出したもの」なので、
保存済みの .npz とマスクがあれば完全に再現できる。追加保存は不要。

**モデルはロードしない。** ROR / DBSCAN に DA3 は要らないので、
model_registry を経由すると調整のたびに VRAM を占有することになる。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.services.depth_ops import (
    depth_map_to_point_cloud,
    instance_points_from_depth,
    resize_mask_nearest,
    rle_to_depth_mask,
)
from common.point_ops import downsample_pair, points_to_json
from common.transform3d import camera_to_ego, scale_intrinsic

logger = get_logger(__name__)


def _load_depth(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """深度マップ .npz を読む（内部パラメータと空マスクは無い場合もある）."""
    with np.load(path) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
        intrinsic = (
            np.asarray(data["intrinsic"], dtype=np.float64)
            if "intrinsic" in data.files else None
        )
        non_sky = (
            np.asarray(data["non_sky"], dtype=bool)
            if "non_sky" in data.files else None
        )
    return depth, intrinsic, non_sky


def refilter_frame(
    depth_path: Path,
    instances: list[dict[str, Any]],
    *,
    calibrated_sensor: dict[str, Any],
    depth_params: dict[str, Any],
    nb_points_ratio: dict[str, float] | None = None,
    stored_points_max: int = 500,
    max_depth: float | None = None,
    camera_intrinsic: Any | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[dict[str, Any]]:
    """1 フレーム分のインスタンスを、新しいパラメータで作り直す.

    Args:
        camera_intrinsic / image_width / image_height:
            .npz に内部パラメータが入っていない場合のフォールバック。
            元画像の K を深度マップの解像度へスケールして使う

    Returns:
        インスタンスごとの ``{id, num_points_raw, num_points_kept,
        points_raw_ego, points_filtered_ego}``
    """
    depth, intrinsic, non_sky = _load_depth(depth_path)
    depth_height, depth_width = depth.shape

    if intrinsic is None:
        # 保存時の内部パラメータが無い run（古い形式）向けの経路。
        # 元画像の K を深度側の解像度へ合わせる
        if camera_intrinsic is None or not image_width or not image_height:
            raise ValueError(
                "深度マップに内部パラメータが無く、"
                "フォールバック用の camera_intrinsic も渡されていません"
            )
        intrinsic = scale_intrinsic(
            camera_intrinsic, depth_width / image_width, depth_height / image_height
        )

    if non_sky is not None and non_sky.shape != depth.shape:
        non_sky = resize_mask_nearest(non_sky, depth_height, depth_width)

    # 全画素の点群は 1 回だけ作り、各インスタンスのマスクで切り出す
    all_points = depth_map_to_point_cloud(depth, intrinsic, depth_threshold=max_depth)

    results: list[dict[str, Any]] = []
    for instance in instances:
        mask = rle_to_depth_mask(
            instance["mask_rle_closed"], depth_height, depth_width
        )
        filtered, raw = instance_points_from_depth(
            all_points, mask,
            common_mask=non_sky,
            ror_nb_points=int(depth_params.get("ror_nb_points", 0)),
            ror_radius=float(depth_params.get("ror_radius", 0.0)),
            dbscan_eps=float(depth_params.get("dbscan_eps", 0.0)),
            dbscan_min_samples=int(depth_params.get("dbscan_min_samples", 0)),
            label=instance.get("label"),
            nb_points_ratio=nb_points_ratio,
        )

        raw_ego = camera_to_ego(
            raw, calibrated_sensor["translation"], calibrated_sensor["rotation"]
        )
        filtered_ego = camera_to_ego(
            filtered, calibrated_sensor["translation"], calibrated_sensor["rotation"]
        )
        # 前後で同じボクセルサイズを使う。別々に間引くと、広がりの大きい
        # フィルタ前が粗くなり「フィルタして点が増えた」ように見える
        reduced_raw, reduced_filtered = downsample_pair(
            raw_ego, filtered_ego, stored_points_max
        )

        results.append({
            "id": instance["id"],
            "track_id": instance.get("track_id"),
            "label": instance.get("label"),
            "num_points_raw": int(raw.shape[0]),
            "num_points_kept": int(filtered.shape[0]),
            "points_raw_ego": points_to_json(reduced_raw),
            "points_filtered_ego": points_to_json(reduced_filtered),
        })
    return results


def refilter(
    derived_root: Path,
    frames: list[dict[str, Any]],
    *,
    depth_params: dict[str, Any],
    nb_points_ratio: dict[str, float] | None = None,
    stored_points_max: int = 500,
    max_depth: float | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """複数フレームをまとめて再フィルタする.

    Returns:
        (インスタンスごとの結果, 所要秒)
    """
    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    for frame in frames:
        path = derived_root / frame["depth_path"]
        if not path.exists():
            logger.warning("depth map not found: %s", path)
            continue
        try:
            results.extend(refilter_frame(
                path, frame["instances"],
                calibrated_sensor=frame["calibrated_sensor"],
                depth_params=depth_params,
                nb_points_ratio=nb_points_ratio,
                stored_points_max=stored_points_max,
                max_depth=max_depth,
                camera_intrinsic=(frame.get("calibrated_sensor") or {}).get(
                    "camera_intrinsic"
                ),
                image_width=frame.get("width"),
                image_height=frame.get("height"),
            ))
        except Exception as exc:  # noqa: BLE001
            # 1 フレームの失敗で全体を落とさない
            logger.warning("refilter failed for %s: %s", frame.get("depth_path"), exc)

    return results, time.perf_counter() - started
