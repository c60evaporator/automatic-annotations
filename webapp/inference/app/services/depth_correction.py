"""LiDAR を基準に、単眼深度推定の点群を補正する.

## 方式

LiDAR 点を **深度マップの画素へ投影**し、その画素の予測深度と
LiDAR の実測深度を比べる。対応点は探索不要で厳密に決まる。

ICP は使わない。単眼深度の誤差は「視線方向の伸縮とずれ」が支配的で、
一般の 6 自由度のずれではない。疎な LiDAR（車 1 台で 10〜100 点）から
6 自由度を決めると、L 字の面に沿って滑る自由度が残る。

## 補正モデル

インスタンス単位では **深度の範囲が狭い**（車 1 台で 4〜5 m）。
そのためスケールとオフセットがほぼ区別できず、どちらも
「その範囲を持ち上げる」効果にしかならない。1 パラメータで十分で、
点数が少ないときはむしろ安定する。

  scale  : z' = s * z        s = median(z_lidar / z_pred)   （既定）
  shift  : z' = z + o        o = median(z_lidar - z_pred)
  affine : z' = a * z + b    最小二乗（点数が多いとき・範囲が広いとき向け）

実測（LiDAR 3 点、補正前 1.2〜3.8 m の誤差）:
  scale 0.16〜0.17 m / shift 0.15〜0.22 m / affine 0.22〜0.25 m

scale は視線方向の伸縮なので、点が視線から外れない（幾何的に自然）。
中央値を使うので、マスクへ混入した背景の LiDAR 点にも強い。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

CORRECTION_SCALE = "scale"
CORRECTION_SHIFT = "shift"
CORRECTION_AFFINE = "affine"
CORRECTION_METHODS = (CORRECTION_SCALE, CORRECTION_SHIFT, CORRECTION_AFFINE)

# affine は 2 パラメータなので、これ未満の点数では scale に落とす
MIN_POINTS_FOR_AFFINE = 5
# 補正係数の妥当範囲。これを外れる推定は何かが壊れているので採用しない
SCALE_LIMITS = (0.3, 3.0)


def sample_depth_at(
    points_camera: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    near_plane: float = 0.1,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """カメラ座標の点を深度マップへ投影し、その画素の予測深度を返す.

    Args:
        points_camera: LiDAR 点（**カメラ座標**）``(N, 3)``
        depth: 深度マップ（保存済みの解像度）
        intrinsic: **深度マップの解像度に合わせた** 内部パラメータ
        mask: 指定するとマスク内の画素だけを採用する（深度マップの解像度）

    Returns:
        ``(z_lidar, z_pred)``。対応が取れた点だけ。
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.shape[0] == 0:
        return np.empty(0), np.empty(0)

    in_front = points_camera[:, 2] > near_plane
    points = points_camera[in_front]
    if points.shape[0] == 0:
        return np.empty(0), np.empty(0)

    projected = points @ np.asarray(intrinsic, dtype=np.float64).T
    us = np.rint(projected[:, 0] / projected[:, 2]).astype(int)
    vs = np.rint(projected[:, 1] / projected[:, 2]).astype(int)

    height, width = depth.shape
    inside = (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
    if mask is not None:
        inside_idx = np.flatnonzero(inside)
        keep = mask[vs[inside_idx], us[inside_idx]]
        inside[inside_idx[~keep]] = False
    if not np.any(inside):
        return np.empty(0), np.empty(0)

    z_pred = depth[vs[inside], us[inside]].astype(np.float64)
    z_lidar = points[inside, 2]
    # 予測が無効な画素（0 や非有限）は対応から外す
    valid = np.isfinite(z_pred) & (z_pred > near_plane)
    return z_lidar[valid], z_pred[valid]


def fit_correction(
    z_lidar: np.ndarray,
    z_pred: np.ndarray,
    method: str = CORRECTION_SCALE,
) -> dict[str, Any] | None:
    """LiDAR と予測深度の対応から補正係数を求める.

    Returns:
        ``{"method", "scale", "offset", "num_points", "residual_before",
        "residual_after"}``。求められない場合は None。
    """
    z_lidar = np.asarray(z_lidar, dtype=np.float64)
    z_pred = np.asarray(z_pred, dtype=np.float64)
    n = int(z_lidar.shape[0])
    if n == 0:
        return None

    used = method
    if method == CORRECTION_AFFINE and n < MIN_POINTS_FOR_AFFINE:
        # 2 パラメータを少数の点で決めると不安定になる
        used = CORRECTION_SCALE

    if used == CORRECTION_SHIFT:
        scale, offset = 1.0, float(np.median(z_lidar - z_pred))
    elif used == CORRECTION_AFFINE:
        design = np.column_stack([z_pred, np.ones(n)])
        (scale, offset), *_ = np.linalg.lstsq(design, z_lidar, rcond=None)
        scale, offset = float(scale), float(offset)
    else:
        scale, offset = float(np.median(z_lidar / z_pred)), 0.0

    if not (SCALE_LIMITS[0] <= scale <= SCALE_LIMITS[1]):
        logger.warning(
            "depth correction rejected: scale=%.3f is out of range (n=%d)", scale, n
        )
        return None

    corrected = scale * z_pred + offset
    return {
        "method": used,
        "requested_method": method,
        "scale": round(scale, 5),
        "offset": round(offset, 4),
        "num_points": n,
        # 補正前後の LiDAR との差（中央値）。効き具合を UI で確認するため
        "residual_before": round(float(np.median(np.abs(z_pred - z_lidar))), 4),
        "residual_after": round(float(np.median(np.abs(corrected - z_lidar))), 4),
    }


def apply_correction(
    points_camera: np.ndarray, correction: dict[str, Any]
) -> np.ndarray:
    """カメラ座標の点群に補正を適用する.

    **点を視線に沿って動かす**（方向を変えず距離だけを変える）。
    z だけを書き換えると、x, y がそのままなので視線から外れてしまう。
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.shape[0] == 0 or not correction:
        return points_camera

    z = points_camera[:, 2]
    new_z = correction["scale"] * z + correction["offset"]
    ratio = np.divide(new_z, z, out=np.ones_like(z), where=z > 1e-6)
    return points_camera * ratio[:, None]
