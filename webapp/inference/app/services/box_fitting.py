"""3D ボックス当てはめの入口（手法の切り替え）.

手法は**ラベルごと**に webapp の config（BOXFIT_METHODS: {label: method}）で
決まり、run の box_fitting_params["methods"] として渡ってくる。
実装は手法ごとに別モジュールへ置き、ここでは共通の戻り値へ整える。

手法:
  convex_hull_moa             … 凸包 + 最小オクルージョン面積（MOA）
  twowheeler_convex_hull_moa  … 二輪車用。MOA で当てはめた後、前後面しか
                                見えていない場合は向きを視線方向へ直す。
                                前後面だけでは幅（横の広がり）しか分からず、
                                MOA は広い方を長辺＝進行方向とみなして
                                向きを 90° 取り違えるため

戻り値の約束:
  center_ego / size_wlh / yaw_ego  … DB の対応する列へそのまま入る
  fitting_score                    … **大きいほど良い** 0〜1 の品質指標。
                                     手法をまたいで比較できるよう統一する
  fit_metrics                      … 手法固有の指標（小さいほど良い値など）
  hull_xy                          … BEV 表示に重ねる凸包
"""
from __future__ import annotations

from typing import Any

import numpy as np

BOXFIT_METHOD_CONVEX_HULL_MOA = "convex_hull_moa"
BOXFIT_METHOD_TWOWHEELER_MOA = "twowheeler_convex_hull_moa"
BOXFIT_METHODS = (BOXFIT_METHOD_CONVEX_HULL_MOA, BOXFIT_METHOD_TWOWHEELER_MOA)
# methods に無いラベル（と、手法の指定が無い古い run）に使う手法
BOXFIT_METHOD_FALLBACK = BOXFIT_METHOD_CONVEX_HULL_MOA


def method_for_label(params: dict[str, Any] | None, label: str | None) -> str:
    """ラベルに対応する当てはめ手法.

    ``params["methods"]``（{label: method}）を優先し、無ければ
    古い run の ``params["method"]``、それも無ければ convex_hull_moa。
    """
    params = params or {}
    methods = params.get("methods") or {}
    if label in methods:
        return str(methods[label])
    return str(params.get("method") or BOXFIT_METHOD_FALLBACK)


def _method_param(params: dict[str, Any], method: str, key: str, default: Any) -> Any:
    """手法ごとの設定（params[method][key]）を引く。古い run は直下の値を使う."""
    sub = params.get(method) or {}
    if key in sub:
        return sub[key]
    return params.get(key, default)


def convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    """BEV の凸包（時計回り）。スタブや表示から使う."""
    from app.services.box_fitting_moa import _convex_hull_clockwise

    return _convex_hull_clockwise(np.asarray(points_xy, dtype=float))


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def fit_box(
    points_ego: np.ndarray,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    sensor_origin_xy: tuple[float, float] = (0.0, 0.0),
    z_range: tuple[float, float] | None = None,
    label: str | None = None,
) -> dict[str, Any] | None:
    """設定された手法でボックスを当てはめる.

    Args:
        points_ego: ego 座標のインスタンス点群
        method: 当てはめ手法（通常は method_for_label で決める）
        label: 物体のラベル。twowheeler_convex_hull_moa が
            想定サイズ（params["box_sizes"]）を引くのに使う
        sensor_origin_xy: ego 座標でのセンサー位置。
            MOA はセンサーから見た隠れ方を使うため、これを誤ると向きが崩れる
        z_range: 高さを外から与える（カメラ跨ぎの結合時に使う）

    Returns:
        当てはめ結果。点が少ない・破綻した場合は None
        （呼び出し側は status を not_fitted にする）。
    """
    params = params or {}
    if method not in BOXFIT_METHODS:
        raise ValueError(f"未対応の Box Fitting 手法です: {method}")

    from app.services.box_fitting_moa import fit_convex_hull_moa

    percentiles = params.get("z_percentiles") or (1.0, 99.0)
    try:
        result = fit_convex_hull_moa(
            points_ego,
            angle_step_deg=float(
                _method_param(params, method, "angle_step_deg", 0.5)
            ),
            sensor_origin_xy=sensor_origin_xy,
            z_percentiles=(float(percentiles[0]), float(percentiles[1])),
            z_range=z_range,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        # 点が少ない・向きが決まらない等。呼び出し側で not_fitted にする
        return None

    twowheeler_info: dict[str, Any] | None = None
    if method == BOXFIT_METHOD_TWOWHEELER_MOA:
        result, twowheeler_info = _twowheeler_adjust(
            result, sensor_origin_xy, params.get("box_sizes"), label
        )

    # 矩形に対する凸包の占有率を品質指標にする。
    # occlusion_area は「小さいほど良い」うえに面積の絶対値なので、
    # 物体の大きさが違うと比較できない
    hull = np.asarray(result["hull_xy"]["points"], dtype=float)
    width, length, _height = result["size_wlh"]
    rect_area = float(width) * float(length)
    score = _polygon_area(hull) / rect_area if rect_area > 0 else 0.0

    fit_metrics = {
        "method": method,
        "occlusion_area": result["occlusion_area"],
        "theta_search": result["theta_search"],
        "corners_xy": result["corners_xy"]["points"],
    }
    if twowheeler_info is not None:
        fit_metrics["twowheeler"] = twowheeler_info
    return {
        "center_ego": result["center_ego"],
        "size_wlh": result["size_wlh"],
        "yaw_ego": result["yaw_ego"],
        "fitting_score": round(min(max(score, 0.0), 1.0), 3),
        "hull_xy": result["hull_xy"],
        "fit_metrics": fit_metrics,
    }


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _axis_gap(a: float, b: float) -> float:
    """2 本の軸（向きを区別しない）のなす角 [rad]（0〜π/2）."""
    d = abs(_wrap_pi(a - b)) % np.pi
    return float(min(d, np.pi - d))


def rectangle_corners(
    center_xy: Any, length: float, width: float, yaw: float
) -> np.ndarray:
    """中心・長さ・幅・向きから BEV の 4 頂点を作る（MOA と同じ回り順）."""
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-np.sin(yaw), np.cos(yaw)])
    c = np.asarray(center_xy, dtype=float)[:2]
    hl, hw = length / 2.0, width / 2.0
    return np.stack([
        c - u * hl - v * hw,
        c + u * hl - v * hw,
        c + u * hl + v * hw,
        c - u * hl + v * hw,
    ])


def _twowheeler_adjust(
    result: dict[str, Any],
    sensor_origin_xy: tuple[float, float],
    box_sizes: dict[str, Any] | None,
    label: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """二輪車用の向きの補正（MOA の結果を受け取って直す）.

    1. 見かけの幅（凸包の方位角の広がり × 箱の中心までの距離）が
       (length + width) / 2 を超えれば側面が見えているとみなし、MOA のまま
    2. そうでなければ前後面が見えている。二輪車は視線方向に
       （こちらへ向かって／遠ざかって）並んでいるとみなし、
       **MOA の 2 軸のうち視線に近い方を進行方向**に選び直して、
       向きを視線方向へ合わせる。大きさは MOA の各軸の長さをそのまま使う
       （進行方向に選んだ軸の長さが length）

    距離フィルタと同じ判定（distance_filter.side_visible）を使う。
    想定サイズが無いラベルは補正しない。
    """
    from app.services.distance_filter import (
        box_size_lw, perceived_width, side_visible,
    )

    size = box_size_lw(box_sizes, label)
    if size is None:
        return result, {"applied": False, "reason": "no_box_size"}

    origin = np.asarray(sensor_origin_xy, dtype=float)
    center = np.asarray(result["center_ego"][:2], dtype=float)
    ref = float(np.hypot(*(center - origin)))
    hull = np.asarray(result["hull_xy"]["points"], dtype=float)
    # 凸包の頂点は輪郭そのものなので、両端を捨てずに広がりを測る
    width_seen = perceived_width(hull, tuple(origin), ref, angle_quantile=0.0)
    side = side_visible(width_seen, size)
    info: dict[str, Any] = {
        "applied": True,
        "side_visible": side,
        "perceived_width": round(width_seen, 3),
        "yaw_corrected": False,
    }
    if side:
        return result, info

    moa_width, moa_length, height = (float(x) for x in result["size_wlh"])
    moa_yaw = float(result["yaw_ego"])
    ray = float(np.arctan2(center[1] - origin[1], center[0] - origin[0]))
    # MOA の長辺方向と短辺方向のうち、視線に近い方を進行方向にする
    if _axis_gap(moa_yaw, ray) <= _axis_gap(moa_yaw + np.pi / 2.0, ray):
        axis, length, width = moa_yaw, moa_length, moa_width
    else:
        axis, length, width = moa_yaw + np.pi / 2.0, moa_width, moa_length
    # 向きは視線方向。前後（180°）は決められないので、選んだ軸に近い方
    yaw = ray if abs(_wrap_pi(ray - axis)) <= np.pi / 2.0 else ray + np.pi
    yaw = _wrap_pi(yaw)

    corners = rectangle_corners(center, length, width, yaw)
    adjusted = {
        **result,
        "size_wlh": [round(width, 3), round(length, 3), round(height, 3)],
        "yaw_ego": round(yaw, 4),
        "corners_xy": {"points": np.round(corners, 3).tolist()},
    }
    info.update({
        "yaw_corrected": True,
        "moa_yaw": round(moa_yaw, 4),
        "yaw_change_deg": round(float(np.rad2deg(_wrap_pi(yaw - moa_yaw))), 2),
    })
    return adjusted, info
