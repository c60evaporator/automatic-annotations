"""3D ボックス当てはめの入口（手法の切り替え）.

手法は**ラベルごと**に webapp の config（BOXFIT_METHODS: {label: method}）で
決まり、run の box_fitting_params["methods"] として渡ってくる。
実装は手法ごとに別モジュールへ置き、ここでは共通の戻り値へ整える。

手法:
  convex_hull_moa … 凸包 + 最小オクルージョン面積（MOA）

NOTE: 二輪車用の twowheeler_convex_hull_moa（前後面が見えているときに向きを
視線方向へ直す）は廃止した。深度点群の形と実際の向きが一致しないことが多く、
見かけの幅による側面／前後面の判定も実データのサイズのばらつきで崩れたため。
向きは今後、画像からの向き推定（PartHOE / Orient Anything など）で補う予定

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

from app.core.logging import get_logger

logger = get_logger(__name__)

BOXFIT_METHOD_CONVEX_HULL_MOA = "convex_hull_moa"
BOXFIT_METHODS = (BOXFIT_METHOD_CONVEX_HULL_MOA,)
# methods に無いラベル（と、手法の指定が無い古い run）に使う手法
BOXFIT_METHOD_FALLBACK = BOXFIT_METHOD_CONVEX_HULL_MOA


def method_for_label(params: dict[str, Any] | None, label: str | None) -> str:
    """ラベルに対応する当てはめ手法.

    ``params["methods"]``（{label: method}）を優先し、無ければ
    古い run の ``params["method"]``、それも無ければ convex_hull_moa。
    """
    params = params or {}
    methods = params.get("methods") or {}
    method = str(methods.get(label) or params.get("method") or BOXFIT_METHOD_FALLBACK)
    if method not in BOXFIT_METHODS:
        # 廃止した手法（twowheeler_convex_hull_moa）が記録された古い run 向け
        logger.warning("unknown box fitting method %s; falling back to %s",
                       method, BOXFIT_METHOD_FALLBACK)
        return BOXFIT_METHOD_FALLBACK
    return method


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
        label: 物体のラベル（今は使わない。手法ごとの拡張のために受け取る）
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
    return {
        "center_ego": result["center_ego"],
        "size_wlh": result["size_wlh"],
        "yaw_ego": result["yaw_ego"],
        "fitting_score": round(min(max(score, 0.0), 1.0), 3),
        "hull_xy": result["hull_xy"],
        "fit_metrics": fit_metrics,
    }
