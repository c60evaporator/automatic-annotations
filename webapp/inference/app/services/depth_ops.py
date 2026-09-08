"""深度マップからインスタンスごとの点群を作る処理.

参考実装（common/geometry/depth.py 相当）を、このサーバーの
データ形式（COCO 非圧縮 RLE の全画面マスク）に合わせて移植したもの。

マスクは元画像の解像度で来るが、深度マップは DA3 の出力解像度なので、
**マスクを深度側の解像度へ合わせてから**投影する。
逆に深度をマスク側へ拡大すると、無い情報を作ることになる。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from common.mask_rle import decode_rle

# 深度がこの値以下の画素は無効とみなす
MIN_VALID_DEPTH = 0.1


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """bool マスクを最近傍補間でリサイズする."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if height < 1 or width < 1:
        raise ValueError(f"size must be >= 1, got {width}x{height}")
    h, w = mask.shape
    rows = ((np.arange(height) + 0.5) * h / height).astype(np.int64).clip(0, h - 1)
    cols = ((np.arange(width) + 0.5) * w / width).astype(np.int64).clip(0, w - 1)
    return np.ascontiguousarray(mask[rows][:, cols], dtype=np.bool_)


def close_mask(
    mask: np.ndarray, dilation: int = 1, erosion: int = 1
) -> np.ndarray:
    """マスクのクロージング（膨張 → 収縮）.

    インスタンスマスクの穴を埋め、細かい欠けを繋ぐ。
    膨張と収縮のカーネルサイズを別々に指定できるようにしてあるのは、
    膨張を大きめにして穴を確実に埋めつつ、収縮を控えめにして
    輪郭を戻しすぎないようにする調整をしたいため。

    NOTE: ここで返したマスクがそのまま DB に保存され、UI の
    「Closed」表示に使われる。webapp 側では再計算できない。
    """
    import cv2

    result = mask.astype(np.uint8)
    if dilation > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        result = cv2.dilate(result, kernel)
    if erosion > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion))
        result = cv2.erode(result, kernel)
    return result.astype(bool)


def depth_map_to_point_cloud(
    depth_map: np.ndarray,
    camera_intrinsic: np.ndarray,
    *,
    depth_threshold: float | None = None,
) -> np.ndarray:
    """深度マップ全体をカメラ座標の点群にする.

    Args:
        camera_intrinsic: **深度マップの解像度に対する** 3x3 内部パラメータ

    Returns:
        shape ``(H*W, 3)``。画素順（行優先）を保つので、
        マスクを flatten して選択できる。
    """
    depth_map = np.asarray(depth_map, dtype=np.float64)
    height, width = depth_map.shape
    us, vs = np.meshgrid(np.arange(width), np.arange(height))

    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    z = depth_map.flatten()
    x = (us.flatten() - cx) * z / fx
    y = (vs.flatten() - cy) * z / fy
    points = np.column_stack([x, y, z])

    if depth_threshold is not None:
        # 画素順を保つため、除外する点は NaN にして後段で落とす
        points[z > depth_threshold] = np.nan
    points[z <= MIN_VALID_DEPTH] = np.nan
    return points


def masked_points(
    all_points: np.ndarray,
    mask: np.ndarray,
    common_mask: np.ndarray | None = None,
) -> np.ndarray:
    """全画素の点群からマスク内の点だけを取り出す.

    Args:
        all_points: depth_map_to_point_cloud の戻り値（画素順）
        common_mask: 空マスクなど、全インスタンス共通で適用する条件

    無効画素（NaN）はここで落とす。
    """
    selector = mask.flatten()
    if common_mask is not None:
        selector = np.logical_and(selector, common_mask.flatten())
    picked = all_points[selector]
    return picked[np.isfinite(picked).all(axis=1)]


def remove_radius_outliers(
    points: np.ndarray, nb_points: int, radius: float
) -> np.ndarray:
    """Radius Outlier Removal.

    半径 radius 内に nb_points 個の近傍を持たない点を落とす。
    深度推定の点群は物体の輪郭から尾を引くように誤差が出るため、
    クラスタリングの前段でこれを落としておく。
    """
    if points.shape[0] == 0 or nb_points < 1:
        return points
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    _, keep = cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return points[np.asarray(keep, dtype=np.int64)] if keep else np.empty((0, 3))


def largest_dbscan_cluster(
    points: np.ndarray, eps: float, min_samples: int
) -> np.ndarray:
    """DBSCAN で最大クラスタだけを残す.

    マスクが背景を巻き込むと、対象物体の手前や奥に別の塊ができる。
    最大クラスタに絞ることで、尾状に伸びたノイズを落とす。
    """
    if points.shape[0] == 0:
        return points
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(
        cloud.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False)
    )
    valid = labels >= 0
    if not valid.any():
        # すべてノイズ判定なら、絞り込まずに返す（点を失うより情報を残す）
        return points
    counts = np.bincount(labels[valid])
    return points[labels == int(np.argmax(counts))]


def instance_points_from_depth(
    all_points: np.ndarray,
    mask: np.ndarray,
    *,
    common_mask: np.ndarray | None = None,
    ror_nb_points: int = 0,
    ror_radius: float = 0.0,
    dbscan_eps: float = 0.0,
    dbscan_min_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """1 インスタンス分の点群を作り、外れ値を除去する.

    Returns:
        (フィルタ後の点群, フィルタ前の点群)

    フィルタ前も返すのは、UI が ROR / DBSCAN の効き具合を
    見比べられるようにするため（webapp 側では再計算できない）。
    """
    raw = masked_points(all_points, mask, common_mask)
    if raw.shape[0] == 0:
        return raw, raw

    points = raw
    if ror_nb_points >= 1 and ror_radius > 0:
        points = remove_radius_outliers(points, ror_nb_points, ror_radius)
    if dbscan_eps > 0 and dbscan_min_samples >= 2:
        points = largest_dbscan_cluster(points, dbscan_eps, dbscan_min_samples)
    return points, raw


def rle_to_depth_mask(
    mask_rle: dict[str, Any], depth_height: int, depth_width: int
) -> np.ndarray:
    """全画面 RLE のマスクを深度マップの解像度へ合わせる."""
    mask = decode_rle(mask_rle)
    if mask.shape == (depth_height, depth_width):
        return mask
    return resize_mask_nearest(mask, depth_height, depth_width)
