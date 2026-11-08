"""AI モデルによるボックス向きの再推定（モデル非依存の前後処理）.

global_track_id × sample ごとに 1 つのカメラを選び、そのカメラ画像から
物体の向きを推定して、基準 ego 座標の yaw に直す。

カメラの選び方: 当てはめに使った点群の実点数（深度 + LiDAR）が最大のもの。
深度点群を使わなかった（LiDAR が十分）カメラも、深度の点数を数える。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

ORIENTATION_METHOD_PART_HOE = "part_hoe"
ORIENTATION_METHOD_ORIENT_ANYTHING = "orient_anything"
# 推定は orient_anything と同じ。ボックスへの反映方法だけが異なる
# （長辺が十分長ければ元のボックスの前後どちらかを選ぶ）
ORIENTATION_METHOD_ORIENT_ANYTHING_COMBINED = "orient_anything_combined"
# 実装済みの方式
SUPPORTED_ORIENTATION_METHODS = (
    ORIENTATION_METHOD_PART_HOE,
    ORIENTATION_METHOD_ORIENT_ANYTHING,
    ORIENTATION_METHOD_ORIENT_ANYTHING_COMBINED,
)
# 方式ごとに使うモデル（model_registry の名前）。同じモデルの方式はまとめて推定する
MODEL_PART_HOE = "part_hoe"
MODEL_ORIENT_ANYTHING = "orient_anything"
MODEL_FOR_METHOD = {
    ORIENTATION_METHOD_PART_HOE: MODEL_PART_HOE,
    ORIENTATION_METHOD_ORIENT_ANYTHING: MODEL_ORIENT_ANYTHING,
    ORIENTATION_METHOD_ORIENT_ANYTHING_COMBINED: MODEL_ORIENT_ANYTHING,
}
# 背景をマスクで落とす（白にする）方式。webapp はこの方式のラベルだけマスクを送る
METHODS_NEEDING_MASK = (
    ORIENTATION_METHOD_ORIENT_ANYTHING, ORIENTATION_METHOD_ORIENT_ANYTHING_COMBINED,
)

# Orient-Anything（V1）の方位角を、PartHOE と同じ視線基準の角度
# （camera_relative_to_yaw の angle_deg）へ直す: view = SIGN * azimuth + OFFSET。
# 方位角は 0 度でカメラの方を向き、PartHOE とは逆回りに増える
# （nuScenes の GT 車両との比較で決めた。0 度 = 視線基準の 180 度。V2 も同じ定義）
ORIENT_ANYTHING_AZIMUTH_SIGN = -1
ORIENT_ANYTHING_AZIMUTH_OFFSET_DEG = 180.0

ORIENTATION_STATUS_ESTIMATED = "estimated"
ORIENTATION_STATUS_NO_BOX = "no_box"
ORIENTATION_STATUS_UNSUPPORTED = "unsupported"
ORIENTATION_STATUS_FAILED = "failed"

# ボックスへの反映方法（fit_metrics["orientation_estimation"]["applied"] に残す）
#   yaw_only     … 回転が 45 度未満（前後反転を含む）。XY 中心と寸法を固定し、yaw だけを推定値にする
#   swap_axis    … 回転が 45〜135 度（長短辺の向きが入れ替わる）。8 頂点を動かさずに
#                  元の ±Y のうち推定に近い方を X 軸に取り直してから（w と l を入れ替え）、
#                  XY 中心を固定して yaw を推定値にする
#   （yaw_only / swap_axis は、回したあと新しい向きの軸で点群の凸包に当て直す。
#     中心のまわりに回しただけでは面が点群を斜めに横切り、後段の寸法補正が
#     点群の表面ではない面を固定してしまうため）
#   long_axis    … orient_anything_combined で長辺が十分長いとき。元のボックスの長辺の
#                  前後のうち推定に近い方を新しい X 軸にする（8 頂点は動かさない）
ORIENTATION_APPLIED_YAW_ONLY = "yaw_only"
ORIENTATION_APPLIED_SWAP_AXIS = "swap_axis"
ORIENTATION_APPLIED_LONG_AXIS = "long_axis"
# 長短辺の向きが入れ替わるとみなす回転角 [度]（45〜135 度、225〜315 度）
SWAP_AXIS_MIN_DEG = 45.0
# orient_anything_combined で長辺を使うかの倍率の既定値（webapp の
# ORIENT_ANYTHING_SWITCH_LENGTH_RATIO が box_fitting_params で渡ってくる）
DEFAULT_SWITCH_LENGTH_RATIO = 1.2
# long_axis で、推定と選んだ軸のなす角がこれ以上なら「角度推定直角ずれ」として印を付ける
# （webapp の ORIENT_INACCURATE_THRESHOLD。後段の複数フレームを考慮した補正の対象にする）
DEFAULT_INACCURATE_THRESHOLD_DEG = 60.0


def group_candidates(
    candidates: list[dict[str, Any]], methods: dict[str, str | None]
) -> list[dict[str, Any]]:
    """候補を global_track_id × sample にまとめ、推定に使うカメラを選ぶ.

    Returns:
        推定対象のグループ（方式が None のラベルは含めない）。
        ``{"method", "label", "primary", "source"}``。primary はボックスを持つ行
        （無ければ None）、source は推定に使うカメラの行
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        groups[(cand["global_track_id"], cand["sample_token"])].append(cand)

    result = []
    for members in groups.values():
        primary = next((m for m in members if m.get("is_primary")), None)
        # ラベルはグループで共通（結合の多数決後）。念のため代表行を優先する
        label = (primary or members[0])["label"]
        method = methods.get(label)
        if not method:
            continue
        # 点数が同じなら代表行を優先する（並び順で結果が変わらないように）
        source = max(
            members,
            key=lambda m: (int(m.get("num_points") or 0), bool(m.get("is_primary")),
                           m["box_fitting_id"]),
        )
        result.append({
            "method": method, "label": label,
            "primary": primary, "source": source,
        })
    return result


def orient_anything_to_view_angle(azimuth_deg: float) -> float:
    """Orient-Anything の方位角 [deg] を視線基準の角度 [deg] にする."""
    return (ORIENT_ANYTHING_AZIMUTH_SIGN * azimuth_deg
            + ORIENT_ANYTHING_AZIMUTH_OFFSET_DEG) % 360.0


def camera_relative_to_yaw(
    angle_deg: float,
    camera_xy: tuple[float, float],
    center_xy: tuple[float, float],
) -> float:
    """カメラから見た向き（PartHOE / MEBOW の定義）を ego の yaw [rad] に直す.

    角度はカメラから物体への視線を基準にした、上から見た向き:
      0 度   視線と同じ向き（カメラに背を向ける）
      90 度  画像の左向き
      180 度 カメラの方を向く
      270 度 画像の右向き

    光軸ではなく**物体への視線**を基準にする（画像の端の人物でも
    「カメラの方を向く」が 180 度になるように学習されているため）。
    """
    fx = center_xy[0] - camera_xy[0]
    fy = center_xy[1] - camera_xy[1]
    norm = math.hypot(fx, fy)
    if norm < 1e-6:
        raise ValueError("ボックス中心がカメラ位置と一致しています")
    fx, fy = fx / norm, fy / norm
    # 視線の右方向（z が上の ego 座標で、視線を時計回りに 90 度回したもの）
    rx, ry = fy, -fx
    theta = math.radians(angle_deg)
    dx = math.cos(theta) * fx - math.sin(theta) * rx
    dy = math.cos(theta) * fy - math.sin(theta) * ry
    return math.atan2(dy, dx)


def wrap_angle(rad: float) -> float:
    """角度を [-π, π) にする."""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def apply_orientation(
    box: dict[str, Any],
    method: str,
    yaw_estimated: float,
    *,
    standard_width: float | None,
    switch_length_ratio: float = DEFAULT_SWITCH_LENGTH_RATIO,
    inaccurate_threshold_deg: float = DEFAULT_INACCURATE_THRESHOLD_DEG,
    hull_xy: Any = None,
) -> dict[str, Any]:
    """推定した向きをボックスへ反映する（box の yaw_ego / size_wlh / center_ego をその場で書き換える）.

    Args:
        box: ``center_ego`` / ``size_wlh``（w, l, h。l が yaw 方向）/ ``yaw_ego`` を持つ
        method: BOX_ORIENTATION_ESTIMATIONS の方式
        yaw_estimated: 推定した向き（基準 ego 座標の yaw [rad]）
        standard_width: ラベルの短辺の標準値（DEFAULT_BOX_SIZES の width）。
            orient_anything_combined の切り替え判定に使う。None なら長辺を使わない
        switch_length_ratio: 長辺が standard_width × これ以上なら long_axis
        inaccurate_threshold_deg: long_axis で推定と選んだ軸のなす角がこれ以上なら
            ``perpendicular_flag`` を立てる
        hull_xy: 当てはめに使った点群の BEV 凸包（``[[x, y], ...]``）。yaw_only /
            swap_axis で回したあと、新しい向きの軸でこの範囲に当て直す。
            None なら当て直さない（中心と寸法を固定したまま回す）

    Returns:
        記録用の ``{"applied", "yaw_before", "rotation_deg", "long_side",
        "switch_length", "perpendicular_flag"}``

    ## 回転が大きいときに軸を取り直す理由

    前面だけが写った車のように、点群が片方の面に沿って細長いボックスを
    中心のまわりに 90 度近く回すと、ボックスが点群から外れる。さらに回した
    ことで側面が見えている扱いになり、後段の幅の補正で自車から離れる側へ
    伸ばされる。長短辺が入れ替わる回転では、先に 8 頂点を動かさずに軸を
    取り直す（点群に沿ったまま）ことで、残りの回転を 45 度未満に抑える。
    """
    yaw_before = float(box["yaw_ego"])
    width, length, height = (float(v) for v in box["size_wlh"])
    long_side = max(width, length)
    switch_length = (
        standard_width * switch_length_ratio if standard_width else None
    )
    record: dict[str, Any] = {
        "yaw_before": yaw_before,
        # 元のボックスからの回転角 [度]（-180〜180）
        "rotation_deg": math.degrees(wrap_angle(yaw_estimated - yaw_before)),
        "long_side": long_side,
        "switch_length": switch_length,
        "perpendicular_flag": False,
    }

    if (method == ORIENTATION_METHOD_ORIENT_ANYTHING_COMBINED
            and switch_length is not None and long_side >= switch_length):
        # 長辺の向き（size_wlh の l が yaw 方向。w の方が長ければ 90 度回した向き）
        long_yaw = yaw_before if length >= width else yaw_before + math.pi / 2.0
        # 長辺の前後 2 方向のうち、推定に近い方を X 軸（length の向き）にする
        candidates = (long_yaw, long_yaw + math.pi)
        best = min(candidates, key=lambda y: abs(wrap_angle(y - yaw_estimated)))
        box["yaw_ego"] = wrap_angle(best)
        box["size_wlh"] = [min(width, length), long_side, height]
        record["applied"] = ORIENTATION_APPLIED_LONG_AXIS
        # 推定が長辺と直交に近い（前後の判定が当てにならない）
        record["estimate_axis_diff_deg"] = math.degrees(
            abs(wrap_angle(best - yaw_estimated)))
        record["perpendicular_flag"] = (
            record["estimate_axis_diff_deg"] >= inaccurate_threshold_deg)
        return record

    if abs(record["rotation_deg"]) >= SWAP_AXIS_MIN_DEG and \
            abs(record["rotation_deg"]) <= 180.0 - SWAP_AXIS_MIN_DEG:
        # 8 頂点を動かさずに、元の ±Y（推定に近い方）を X 軸に取り直す（w と l が入れ替わる）。
        # そのうえで XY 中心を固定して推定の向きまで回す（残りの回転は 45 度未満）
        box["size_wlh"] = [length, width, height]
        box["yaw_ego"] = wrap_angle(yaw_estimated)
        record["applied"] = ORIENTATION_APPLIED_SWAP_AXIS
    else:
        # XY 中心と寸法はそのまま、yaw だけ差し替える（前後反転もここ）
        box["yaw_ego"] = wrap_angle(yaw_estimated)
        record["applied"] = ORIENTATION_APPLIED_YAW_ONLY
    record.update(refit_to_hull(box, hull_xy))
    return record


def refit_to_hull(box: dict[str, Any], hull_xy: Any) -> dict[str, Any]:
    """box の yaw のまま、点群の凸包を囲む最小の矩形へ当て直す（box をその場で書き換える）.

    中心と長さ・幅を凸包の範囲から決め直す（高さと z は変えない）。
    凸包が無い・潰れている場合は何もしない。

    Returns:
        記録用の ``{"refit", "refit_center_shift"}``
    """
    points = np.asarray(hull_xy if hull_xy is not None else [], dtype=float)
    if points.ndim != 2 or points.shape[0] < 3:
        return {"refit": False}
    yaw = float(box["yaw_ego"])
    u = np.array([math.cos(yaw), math.sin(yaw)])   # 長さ方向
    v = np.array([-math.sin(yaw), math.cos(yaw)])  # 幅方向
    pu, pv = points @ u, points @ v
    length, width = float(pu.max() - pu.min()), float(pv.max() - pv.min())
    if length <= 1e-3 or width <= 1e-3:
        return {"refit": False}
    center_xy = ((pu.max() + pu.min()) / 2.0) * u + ((pv.max() + pv.min()) / 2.0) * v
    old_xy = np.asarray(box["center_ego"][:2], dtype=float)
    box["center_ego"] = [float(center_xy[0]), float(center_xy[1]),
                         *[float(z) for z in box["center_ego"][2:]]]
    box["size_wlh"] = [width, length, float(box["size_wlh"][2])]
    metrics = box.get("fit_metrics")
    if isinstance(metrics, dict) and "corners_xy" in metrics:
        hl, hw = length / 2.0, width / 2.0
        corners = [center_xy - u * hl - v * hw, center_xy + u * hl - v * hw,
                   center_xy + u * hl + v * hw, center_xy - u * hl + v * hw]
        metrics["corners_xy"] = np.round(np.stack(corners), 3).tolist()
    return {"refit": True,
            "refit_center_shift": float(np.linalg.norm(center_xy - old_xy))}
