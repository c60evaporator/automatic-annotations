"""3D ボックスの底面高さの推定.

上面と底面を点群の z の分位点で決めると、LiDAR の上下方向のビームが疎な
ため、**底面が実際より高く（箱が低く）**なりやすい。底面はラベルごとの
方式で別に決める（webapp の config の BOX_BOTTOM_ESTIMATIONS: {label: 方式}）。

方式:
  percentile … 点群の z の分位点（Bottom Estimation の z_percentile）。
               当てはめ（MOA）が分位点で出した下端をそのまま使う
  ground     … Patchwork++ が判定した地面点のうち、**箱の外形から XY で
               Ground Estimation Radius 以内**の点から求める
               （Use LiDAR のときだけ。地面点が Min Ground Points 以下なら
               percentile に切り替える）
                 percentile    : 地面点の z の分位点（Ground Percentile。既定 30%）。
                                 中央値より低くするのは、歩道や縁石の点が
                                 混ざると路面より高く出るため
                 least_squares : 地面点に平面 z = a·x + b·y + c を最小二乗で
                                 当てはめ、箱の XY 中心での高さ
  ego        … 基準 ego 座標の z = 0（自車の接地面にそろえる）

範囲を「箱の外形から」測るのは、重心からの固定半径だと大きな車両では
円が車体に収まって周囲の地面が入らず、小さな物体では遠くの地面まで
入るため。車体の下は LiDAR の影になるので、実際に使われるのは周囲の帯。

上面は従来どおり分位点（Top Estimation の z_percentile）で決める。

推論（結合しない場合のパイプラインと、カメラ間結合後の merge_and_fit）で
**同じ関数**を使うこと。
"""
from __future__ import annotations

from typing import Any

import numpy as np

BOTTOM_PERCENTILE = "percentile"
BOTTOM_GROUND = "ground"
BOTTOM_EGO = "ego"
BOTTOM_METHODS = (BOTTOM_PERCENTILE, BOTTOM_GROUND, BOTTOM_EGO)

GROUND_HEIGHT_PERCENTILE = "percentile"
GROUND_HEIGHT_LEAST_SQUARES = "least_squares"
# 旧名（中央値）。古い run は percentile 50 として扱う
GROUND_HEIGHT_MEDIAN = "median"

DEFAULT_GROUND_RADIUS = 2.0
DEFAULT_GROUND_PERCENTILE = 30.0
DEFAULT_MIN_GROUND_POINTS = 10
# 底面を下げた結果、箱がこれより低くなる（上下が逆転するなど）場合は
# 底面を変えない
MIN_BOX_HEIGHT = 0.1


def bottom_settings(
    box_fitting_params: dict[str, Any] | None, label: str | None
) -> dict[str, Any]:
    """run のパラメータから、このラベルの底面推定の設定を組み立てる.

    ``box_fitting_params["bottom"]`` が無い古い run は percentile（従来どおり）。
    """
    bottom = (box_fitting_params or {}).get("bottom") or {}
    method = (bottom.get("methods") or {}).get(label or "") or BOTTOM_PERCENTILE
    if method not in BOTTOM_METHODS:
        method = BOTTOM_PERCENTILE
    return {
        "method": method,
        "ground_radius": float(bottom.get("ground_radius", DEFAULT_GROUND_RADIUS)),
        "min_ground_points": int(
            bottom.get("min_ground_points", DEFAULT_MIN_GROUND_POINTS)
        ),
        **_ground_height_settings(bottom),
    }


def _ground_height_settings(bottom: dict[str, Any]) -> dict[str, Any]:
    method = str(bottom.get("ground_height_method", GROUND_HEIGHT_PERCENTILE))
    if method == GROUND_HEIGHT_MEDIAN:
        # 旧名。中央値＝50% 点
        return {"ground_height_method": GROUND_HEIGHT_PERCENTILE,
                "ground_percentile": 50.0}
    return {
        "ground_height_method": method,
        "ground_percentile": float(
            bottom.get("ground_percentile", DEFAULT_GROUND_PERCENTILE)
        ),
    }


def distance_to_box_xy(
    points_xy: np.ndarray,
    center_xy: tuple[float, float],
    length: float,
    width: float,
    yaw: float,
) -> np.ndarray:
    """BEV の点から、向き付きの矩形（箱の外形）までの距離（内側は 0）."""
    rel = np.asarray(points_xy, dtype=float)[:, :2] - np.asarray(center_xy, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    along = rel[:, 0] * c + rel[:, 1] * s
    across = -rel[:, 0] * s + rel[:, 1] * c
    dx = np.maximum(np.abs(along) - length / 2.0, 0.0)
    dy = np.maximum(np.abs(across) - width / 2.0, 0.0)
    return np.hypot(dx, dy)


def _ground_height(
    ground: np.ndarray,
    center_xy: tuple[float, float],
    method: str,
    percentile: float,
) -> tuple[float, str]:
    """地面点から箱の XY 中心での地面の高さを求める. (高さ, 使った方法)."""
    if method == GROUND_HEIGHT_LEAST_SQUARES:
        design = np.column_stack([ground[:, 0], ground[:, 1], np.ones(ground.shape[0])])
        coef, _res, rank, _sv = np.linalg.lstsq(design, ground[:, 2], rcond=None)
        if rank == 3:
            a, b, c = (float(v) for v in coef)
            return a * center_xy[0] + b * center_xy[1] + c, GROUND_HEIGHT_LEAST_SQUARES
        # 点が一直線上に並ぶなど、平面が決まらない場合は分位点
    return float(np.percentile(ground[:, 2], percentile)), GROUND_HEIGHT_PERCENTILE


def estimate_bottom(
    fit: dict[str, Any],
    settings: dict[str, Any],
    *,
    ground_points: np.ndarray | None,
    use_lidar: bool,
) -> tuple[float | None, dict[str, Any]]:
    """底面の高さを決める.

    Args:
        fit: fit_box の戻り値（center_ego / size_wlh / yaw_ego）
        settings: bottom_settings の戻り値
        ground_points: その sample の地面点（基準 ego 座標）

    Returns:
        (底面の z。percentile なら None＝当てはめの値のまま, 診断情報)
    """
    method = settings["method"]
    info: dict[str, Any] = {"method": method}
    if method == BOTTOM_EGO:
        return 0.0, info
    if method != BOTTOM_GROUND:
        return None, info

    if not use_lidar:
        info.update({"fallback": BOTTOM_PERCENTILE, "reason": "lidar_not_used"})
        return None, info
    if ground_points is None or len(ground_points) == 0:
        info.update({"fallback": BOTTOM_PERCENTILE, "reason": "no_ground_points"})
        return None, info

    center = fit["center_ego"]
    width, length, _height = (float(v) for v in fit["size_wlh"])
    ground = np.asarray(ground_points, dtype=float).reshape(-1, 3)
    near = distance_to_box_xy(
        ground[:, :2], (center[0], center[1]), length, width, float(fit["yaw_ego"]),
    ) <= settings["ground_radius"]
    selected = ground[near]
    info["num_ground_points"] = int(selected.shape[0])
    if selected.shape[0] <= settings["min_ground_points"]:
        info.update({"fallback": BOTTOM_PERCENTILE, "reason": "too_few_ground_points"})
        return None, info

    z, used = _ground_height(
        selected, (float(center[0]), float(center[1])),
        settings["ground_height_method"], settings["ground_percentile"],
    )
    info.update({"ground_height_method": used, "ground_z": round(z, 3)})
    if used == GROUND_HEIGHT_PERCENTILE:
        info["ground_percentile"] = settings["ground_percentile"]
    return z, info


def apply_bottom(
    fit: dict[str, Any],
    box_fitting_params: dict[str, Any] | None,
    label: str | None,
    *,
    ground_points: np.ndarray | None,
    use_lidar: bool,
) -> dict[str, Any]:
    """当てはめ結果の底面を差し替える（上面は分位点のまま）.

    center_ego の z と size_wlh の高さを書き換え、fit_metrics["bottom"] に
    決め方を記録する。fit はその場で書き換えて返す。
    """
    settings = bottom_settings(box_fitting_params, label)
    z_bottom, info = estimate_bottom(
        fit, settings, ground_points=ground_points, use_lidar=use_lidar,
    )
    center = list(fit["center_ego"])
    size = list(fit["size_wlh"])
    top = float(center[2]) + float(size[2]) / 2.0
    original_bottom = float(center[2]) - float(size[2]) / 2.0
    info["percentile_bottom"] = round(original_bottom, 3)
    if z_bottom is not None:
        if top - z_bottom < MIN_BOX_HEIGHT:
            # 地面が上面より上に出たなど。底面は変えない
            info.update({"fallback": BOTTOM_PERCENTILE, "reason": "inverted_height"})
        else:
            center[2] = round((top + z_bottom) / 2.0, 3)
            size[2] = round(top - z_bottom, 3)
            fit["center_ego"] = center
            fit["size_wlh"] = size
            info["bottom_z"] = round(z_bottom, 3)
    fit.setdefault("fit_metrics", {})["bottom"] = info
    return fit
