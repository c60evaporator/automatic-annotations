"""LiDAR 点群の読み込み・sweep 統合・地面除去.

nuScenes の sweep は、それぞれ別の時刻の LiDAR 座標で記録されている。
自車が動くため、sweep 同士は **global 座標を経由しないと関係づけられない**:

    sweep の lidar → sweep の ego → global → キーフレームの ego → キーフレームの lidar

キーフレーム自身はこの連鎖が恒等変換になるので、変換は不要。

地面除去（Patchwork++）は sweep ごとに実行する。
統合してから一括で処理すると、時刻の違う点が混ざった状態で地面を推定
することになり、自車のピッチ変化があると精度が落ちる。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from common.transform3d import invert_transform, make_transform, transform_points

logger = get_logger(__name__)

# nuScenes の .pcd.bin は float32 × 5（x, y, z, intensity, ring）
LIDAR_POINT_DIMS = 5


def load_lidar_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """nuScenes の .pcd.bin を読む.

    Returns:
        (points (N, 3), intensity (N,))
    """
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, LIDAR_POINT_DIMS)
    return raw[:, :3].astype(np.float64), raw[:, 3].astype(np.float64)


def sweep_to_keyframe_lidar(
    points_lidar: np.ndarray,
    sweep_calib: dict[str, Any],
    sweep_ego: dict[str, Any],
    key_calib: dict[str, Any],
    key_ego: dict[str, Any],
) -> np.ndarray:
    """sweep の LiDAR 座標をキーフレームの LiDAR 座標へ移す.

    自車が sweep とキーフレームの間で動くため、
    2 つの ego 座標系は global を経由してしか関係づけられない。
    """
    lidar_to_ego = make_transform(sweep_calib["rotation"], sweep_calib["translation"])
    ego_to_global = make_transform(sweep_ego["rotation"], sweep_ego["translation"])
    key_ego_to_global = make_transform(key_ego["rotation"], key_ego["translation"])
    key_lidar_to_ego = make_transform(key_calib["rotation"], key_calib["translation"])

    chain = (
        invert_transform(key_lidar_to_ego)
        @ invert_transform(key_ego_to_global)
        @ ego_to_global
        @ lidar_to_ego
    )
    return transform_points(points_lidar, chain)


def estimate_ground(
    points: np.ndarray,
    intensity: np.ndarray,
    *,
    sensor_height: float = 1.723,
) -> tuple[np.ndarray, np.ndarray]:
    """Patchwork++ で地面と非地面に分ける.

    Args:
        points: **LiDAR 座標**の点群 (N, 3)。ego 座標を渡してはいけない
            （Patchwork++ は sensor_height を基準に地面を探すため）

    Returns:
        (ground (G, 3), nonground (M, 3))
    """
    import pypatchworkpp

    params = pypatchworkpp.Parameters()
    params.sensor_height = float(sensor_height)
    params.verbose = False
    estimator = pypatchworkpp.patchworkpp(params)

    # Patchwork++ は (N, 4) の x, y, z, intensity を要求する
    cloud = np.hstack([points, intensity.reshape(-1, 1)]).astype(np.float64)
    estimator.estimateGround(cloud)
    ground = np.asarray(estimator.getGround(), dtype=np.float64).reshape(-1, 3)
    nonground = np.asarray(estimator.getNonground(), dtype=np.float64).reshape(-1, 3)
    return ground, nonground


def merge_sweeps_with_ground(
    sweeps: list[dict[str, Any]],
    dataroot: Path,
    *,
    sensor_height: float = 1.723,
    remove_ground: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """sweep を統合し、地面判定を付けた点群を返す.

    Args:
        sweeps: 古い順のフレーム情報（末尾がキーフレーム）。
            各要素は filename / calibrated_sensor / ego_pose を持つ

    Returns:
        (points (N, 3) キーフレームの LiDAR 座標, ground_mask (N,))

    地面点も捨てずに返すのは、UI が比較用に地面を表示できるようにするため。
    """
    if not sweeps:
        return np.empty((0, 3)), np.empty((0,), dtype=bool)

    key_frame = sweeps[-1]
    key_calib = key_frame["calibrated_sensor"]
    key_ego = key_frame["ego_pose"]

    ground_chunks: list[np.ndarray] = []
    nonground_chunks: list[np.ndarray] = []

    for sweep in sweeps:
        path = dataroot / sweep["filename"]
        if not path.exists():
            logger.warning("lidar file not found: %s", path)
            continue
        points, intensity = load_lidar_points(path)

        if remove_ground:
            # 地面推定は LiDAR 座標のまま、sweep ごとに行う。
            # 統合後にまとめて推定すると、時刻の違う点が混ざった状態で
            # 地面を当てることになり、ピッチ変化があると精度が落ちる
            ground, nonground = estimate_ground(
                points, intensity, sensor_height=sensor_height
            )
        else:
            ground, nonground = np.empty((0, 3)), points

        # キーフレームの LiDAR 座標へ揃える（キーフレーム自身は恒等変換）
        if sweep["sample_data_token"] != key_frame["sample_data_token"]:
            args = (sweep["calibrated_sensor"], sweep["ego_pose"], key_calib, key_ego)
            if ground.shape[0]:
                ground = sweep_to_keyframe_lidar(ground, *args)
            if nonground.shape[0]:
                nonground = sweep_to_keyframe_lidar(nonground, *args)

        ground_chunks.append(ground)
        nonground_chunks.append(nonground)

    ground_all = (
        np.vstack(ground_chunks) if ground_chunks else np.empty((0, 3))
    )
    nonground_all = (
        np.vstack(nonground_chunks) if nonground_chunks else np.empty((0, 3))
    )

    points = np.vstack([nonground_all, ground_all])
    ground_mask = np.concatenate([
        np.zeros(nonground_all.shape[0], dtype=bool),
        np.ones(ground_all.shape[0], dtype=bool),
    ])
    return points, ground_mask


def lidar_to_ego(points: np.ndarray, calib: dict[str, Any]) -> np.ndarray:
    """キーフレームの LiDAR 座標を ego 座標へ移す（保存用）."""
    if points.shape[0] == 0:
        return points
    return transform_points(
        points, make_transform(calib["rotation"], calib["translation"])
    )
