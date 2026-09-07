"""点群の間引き.

inference 側（DB へ保存する前に上限まで減らす）と
webapp 側（Plotly へ渡す前に表示上限まで減らす）の両方で使うため common に置く。

Plotly の 3D 散布図はブラウザへ JSON で送られるので、点数がそのまま
転送量になる（10 万点で約 1.6MB、34 万点で約 5.5MB）。
深度画像 1 枚を点群化すると 144 万点になるため、間引きは必須。
"""
from __future__ import annotations

import numpy as np

# ボクセルサイズを探索する際の反復上限。
# 目標点数に厳密に合わせる必要はないので、少ない回数で打ち切る
MAX_VOXEL_ITERATIONS = 12

# ボクセル化の前に等間隔で粗く間引く倍率。
# 目標点数の何倍まで落としてからボクセル化するか
PRE_STRIDE_FACTOR = 8


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """ボクセルグリッドで間引く（各ボクセルの代表点を 1 つ残す）.

    ランダム抽出と違い、密な領域だけが薄くなって形状が保たれる。
    ボックスフィッティングの当てはまりを目視で確認する用途に向く。

    Args:
        points: shape ``(N, 3)``
        voxel_size: 一辺の長さ [m]。0 以下なら間引かない
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if voxel_size <= 0 or points.shape[0] == 0:
        return points

    keys = np.floor(points / voxel_size).astype(np.int64)
    # 各ボクセルで最初に現れた点を残す。np.unique は入力順を保たないので
    # インデックスを取り、元の順序に戻してから使う
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first_indices)]


def downsample_to_max(
    points: np.ndarray,
    max_points: int,
    *,
    initial_voxel_size: float | None = None,
) -> np.ndarray:
    """点数が上限を超える場合だけ、ボクセル間引きで上限以下に落とす.

    ボクセルサイズを倍々にしながら、点数が収まるまで繰り返す。
    それでも収まらない場合（極端に平坦な点群など）は等間隔の間引きで
    確実に上限へ落とす。

    Args:
        max_points: 残す点数の上限。0 以下なら間引かない
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if max_points <= 0 or points.shape[0] <= max_points:
        return points

    # 巨大な点群（深度画像 1 枚で 144 万点）に対して、いきなり細かい
    # ボクセルで unique を取ると数秒かかる。先に等間隔で粗く間引いてから
    # ボクセル化すると、形状をほぼ保ったまま桁違いに速くなる
    working = points
    if working.shape[0] > max_points * PRE_STRIDE_FACTOR:
        stride = int(np.ceil(working.shape[0] / (max_points * PRE_STRIDE_FACTOR)))
        working = working[::stride]

    if initial_voxel_size is None:
        # 点群の広がりから初期ボクセルサイズを見積もる。
        # 「目標点数ぶんの立方格子に収まる一辺」を出発点にする
        extent = working.max(axis=0) - working.min(axis=0)
        volume = float(np.prod(np.maximum(extent, 1e-6)))
        initial_voxel_size = max((volume / max_points) ** (1.0 / 3.0), 1e-3)

    voxel_size = initial_voxel_size
    reduced = working
    for _ in range(MAX_VOXEL_ITERATIONS):
        reduced = voxel_downsample(working, voxel_size)
        if reduced.shape[0] <= max_points:
            return reduced
        voxel_size *= 1.5

    # 保険: 等間隔に間引いて必ず上限以下にする
    stride = int(np.ceil(reduced.shape[0] / max_points))
    return reduced[::stride][:max_points]


def points_to_json(points: np.ndarray, decimals: int = 2) -> dict:
    """DB の JSON 列へ入れる形にする.

    小数 2 桁（cm 精度）に丸める。丸めるだけで文字列長が 1/3 になり、
    表示用途では精度も十分。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return {"points": []}
    return {"points": np.round(points, decimals).tolist()}


def points_from_json(data: dict | None) -> np.ndarray:
    """JSON 列から点群を取り出す（空なら shape (0, 3)）."""
    if not data or not data.get("points"):
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(data["points"], dtype=np.float64)
