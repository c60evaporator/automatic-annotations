"""DERIVED_ROOT に置かれた派生ファイルの読み込み.

推論サーバーが書いた深度マップと LiDAR 点群を webapp から読む。
DATA_ROOT（読み取り専用のデータセット）とは別のマウントなので、
パス解決も別に用意する。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DerivedPathError(ValueError):
    """DERIVED_ROOT の外を指すパスが渡された."""


def resolve_derived_path(relative_path: str) -> Path:
    """DERIVED_ROOT 相対のパスを実パスへ解決する.

    DB 由来の値をそのまま結合するため、範囲外を指していないか検証する。
    失敗した推論の行はパスが空文字なので、それも弾く
    （空のまま結合すると DERIVED_ROOT 自体を指してしまう）。
    """
    if not relative_path:
        raise DerivedPathError("パスが空です（推論が失敗した行の可能性）")
    root = get_settings().DERIVED_ROOT.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise DerivedPathError(f"DERIVED_ROOT の外を指しています: {relative_path}")
    return path


def load_depth_map(relative_path: str) -> np.ndarray | None:
    """深度マップ .npz を読む.

    保存は元画像の 1/2 解像度なので、内部パラメータを使う側で
    同じ倍率にスケールする必要がある（DepthEstimation.depth_width 参照）。
    """
    try:
        path = resolve_derived_path(relative_path)
    except DerivedPathError as exc:
        logger.warning("invalid derived path: %s", exc)
        return None
    if not path.exists():
        logger.warning("depth map not found: %s", path)
        return None
    with np.load(path) as data:
        return np.asarray(data["depth"], dtype=np.float32)


def load_lidar_pointcloud(
    relative_path: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """LiDAR 統合点群 .npz を読む.

    Returns:
        (points (N, 3), ground_mask (N,))。見つからなければ None。

    地面マスクを一緒に保存しているのは、webapp に Patchwork++ が無く
    地面判定を再現できないため。
    """
    try:
        path = resolve_derived_path(relative_path)
    except DerivedPathError as exc:
        logger.warning("invalid derived path: %s", exc)
        return None
    if not path.exists():
        logger.warning("lidar pointcloud not found: %s", path)
        return None
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        ground = np.asarray(data["ground_mask"], dtype=bool)
    return points, ground


def derived_file_exists(relative_path: str) -> bool:
    try:
        return resolve_derived_path(relative_path).exists()
    except DerivedPathError:
        return False
