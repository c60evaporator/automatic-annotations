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


# 最終補正の既定値（webapp から lidar_params で渡ってくる）
DEFAULT_CORRESPONDENCE_RADIUS = 2   # 近傍の半径 [深度マップの画素]
DEFAULT_MIN_CORRESPONDENCES = 3     # 点ごとの対応で補正するのに必要な対応点の数


def neighbor_depth_at(
    points_camera: np.ndarray,
    kept_camera: np.ndarray,
    intrinsic: np.ndarray,
    shape: tuple[int, int],
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """LiDAR 点ごとに、周囲の**残った深度点**の予測深度の中央値を返す.

    予測深度は必ずフィルタ後に残った深度点から取る（LiDAR 点が当たった
    画素そのものは読まない）。その画素がフィルタで落としたノイズでも、
    ノイズの深度が補正に入り込まないようにするため。

    Args:
        points_camera: LiDAR 点（カメラ座標）
        kept_camera: フィルタ後に残った深度点（カメラ座標。深度マップの
            画素中心から逆投影したもの）
        radius: 近傍の半径 [画素]。(2r+1)×(2r+1) の窓を見る

    Returns:
        ``(z_lidar, z_pred)``。窓に残った深度点が無い LiDAR 点は除く
    """
    height, width = shape
    kept = np.asarray(kept_camera, dtype=np.float64).reshape(-1, 3)
    kept = kept[np.isfinite(kept).all(axis=1) & (kept[:, 2] > 0)]
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1)]
    if kept.shape[0] == 0 or points.shape[0] == 0:
        return np.empty(0), np.empty(0)

    K = np.asarray(intrinsic, dtype=np.float64)

    def _pixels(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj = p @ K.T
        return (np.rint(proj[:, 0] / proj[:, 2]).astype(int),
                np.rint(proj[:, 1] / proj[:, 2]).astype(int))

    # 残った深度点の予測深度だけを書いた画像（それ以外は NaN）
    image = np.full(shape, np.nan)
    ku, kv = _pixels(kept)
    inside = (ku >= 0) & (ku < width) & (kv >= 0) & (kv < height)
    image[kv[inside], ku[inside]] = kept[inside, 2]

    us, vs = _pixels(points)
    r = max(int(radius), 0)
    z_lidar, z_pred = [], []
    for u, v, z in zip(us, vs, points[:, 2]):
        if u < -r or u >= width + r or v < -r or v >= height + r:
            continue
        window = image[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
        values = window[np.isfinite(window)]
        if values.size:
            z_lidar.append(float(z))
            z_pred.append(float(np.median(values)))
    return np.asarray(z_lidar), np.asarray(z_pred)


def refine_correction(
    lidar_camera: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    kept_camera: np.ndarray,
    method: str,
    initial: dict[str, Any] | None = None,
    *,
    radius: int = DEFAULT_CORRESPONDENCE_RADIUS,
    min_correspondences: int = DEFAULT_MIN_CORRESPONDENCES,
) -> dict[str, Any] | None:
    """フィルタ後に残った点だけで、最終の補正係数を求める.

    仮の補正（クロージング後マスク全体の画素で求めたもの）はフィルタの
    判定にだけ使い、**最終の補正には使わない**。マスクの縁で背景の深度を
    拾った画素などが係数を引きずるため。

      1. 近傍対応（``kept_neighbors``）: LiDAR 点ごとに、周囲 radius 画素に
         ある残った深度点の予測深度の中央値と対応させ、対応点が
         min_correspondences 以上なら従来どおり点ごとの比から係数を求める
      2. 中央値の比（``median_ratio``）: 対応点が足りなければ、
         「残った LiDAR 点の深度の中央値」と「残った深度点の予測深度の
         中央値」の比で求める。画素の対応が要らないので、遠方の小さな
         物体でも求まる（物体の奥行きは距離に比べて小さいため）

    推論（boxfitting_pipeline）と再フィルタ（refilter）で**同じ関数**を使うこと。

    Args:
        initial: 仮の補正（診断用に initial_* として記録するだけ）

    Returns:
        補正係数。``correspondence`` に求め方を入れる。求められなければ None
    """
    initial_info = {
        "initial_scale": initial["scale"],
        "initial_offset": initial["offset"],
        "initial_num_points": initial["num_points"],
    } if initial is not None else {}
    kept_camera = np.asarray(kept_camera, dtype=np.float64).reshape(-1, 3)
    if kept_camera.shape[0] == 0:
        return None

    # 1. 近傍対応
    z_lidar, z_pred = neighbor_depth_at(
        lidar_camera, kept_camera, intrinsic, depth.shape, radius,
    )
    if z_lidar.shape[0] >= max(int(min_correspondences), 1):
        refined = fit_correction(z_lidar, z_pred, method)
        if refined is not None:
            return {**refined, "correspondence": "kept_neighbors",
                    "radius": int(radius), **initial_info}

    # 2. 中央値の比
    lidar = np.asarray(lidar_camera, dtype=np.float64).reshape(-1, 3)
    lidar_z = lidar[np.isfinite(lidar).all(axis=1) & (lidar[:, 2] > 0.1), 2]
    kept_z = kept_camera[np.isfinite(kept_camera).all(axis=1) & (kept_camera[:, 2] > 0), 2]
    if lidar_z.size == 0 or kept_z.size == 0:
        return None
    median_lidar, median_pred = float(np.median(lidar_z)), float(np.median(kept_z))
    fitted = fit_correction(np.array([median_lidar]), np.array([median_pred]), method)
    if fitted is None:
        return None
    return {
        **fitted,
        "correspondence": "median_ratio",
        # 1 組の比なので、点数は使った点群の大きさを記録する
        "num_points": int(z_lidar.shape[0]),
        "num_lidar": int(lidar_z.size),
        "num_depth": int(kept_z.size),
        "median_lidar": round(median_lidar, 3),
        "median_pred": round(median_pred, 3),
        **initial_info,
    }
