"""深度マップから点群を作る.

「Show Raw Depth」で深度画像そのものを 3D で確認するために使う。
インスタンスごとの点群は推論サーバーが作って DB に入れているので、
こちらは全画素を対象にした確認用。
"""
from __future__ import annotations

import numpy as np

from app.services.geometry.transform import make_transform

# 深度がこの値以下の画素は捨てる（無効値・極近傍）
MIN_VALID_DEPTH = 0.1


def depth_to_camera_points(
    depth: np.ndarray,
    camera_intrinsic: np.ndarray,
    *,
    full_width: int,
    full_height: int,
    stride: int = 1,
    max_depth: float | None = None,
) -> np.ndarray:
    """深度マップをカメラ座標の点群にする.

    Args:
        depth: shape ``(h, w)``。元画像より小さいことがある（保存倍率）
        camera_intrinsic: **元画像の解像度に対する** 3x3 内部パラメータ
        full_width / full_height: 元画像の解像度
        stride: 画素の間引き。大きいほど軽い

    内部パラメータは保存倍率に合わせてスケールする。
    ここを忘れると、点群が実際より広がった形になる。
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got {depth.shape}")

    height, width = depth.shape
    scale_x = width / float(full_width)
    scale_y = height / float(full_height)

    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64).copy()
    intrinsic[0, :] *= scale_x
    intrinsic[1, :] *= scale_y

    if stride > 1:
        depth = depth[::stride, ::stride]
        intrinsic[0, :] /= stride
        intrinsic[1, :] /= stride
        height, width = depth.shape

    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    valid = depth > MIN_VALID_DEPTH
    if max_depth is not None:
        valid &= depth <= max_depth
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    z = depth[valid].astype(np.float64)
    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack([x, y, z])


def camera_points_to_ego(
    points_camera: np.ndarray,
    camera_translation,
    camera_quaternion,
) -> np.ndarray:
    """カメラ座標の点群を ego 座標へ変換する.

    CalibratedSensor は camera → ego の変換なので、そのまま適用する
    （ego → camera の逆変換ではない点に注意）。
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.shape[0] == 0:
        return points_camera
    camera_to_ego = make_transform(camera_quaternion, camera_translation)
    return points_camera @ camera_to_ego[:3, :3].T + camera_to_ego[:3, 3]


def depth_to_ego_points(
    depth: np.ndarray,
    frame: dict,
    *,
    stride: int = 4,
    max_depth: float | None = 60.0,
) -> np.ndarray:
    """深度マップを ego 座標の点群にする（表示用の入口）.

    Args:
        frame: calibrated_sensor と width/height を持つフレーム情報
    """
    calib = frame.get("calibrated_sensor") or {}
    intrinsic = calib.get("camera_intrinsic")
    if intrinsic is None or not frame.get("width") or not frame.get("height"):
        return np.empty((0, 3), dtype=np.float64)

    camera_points = depth_to_camera_points(
        depth, intrinsic,
        full_width=frame["width"], full_height=frame["height"],
        stride=stride, max_depth=max_depth,
    )
    return camera_points_to_ego(
        camera_points, calib["translation"], calib["rotation"]
    )
