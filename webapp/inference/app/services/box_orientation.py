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

ORIENTATION_METHOD_PART_HOE = "part_hoe"
ORIENTATION_METHOD_ORIENT_ANYTHING = "orient_anything"
# 推定は orient_anything と同じ。ボックスへの反映方法だけが異なる
# （長辺が十分長ければ元のボックスの前後どちらかを選ぶ。反映は未実装）
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
