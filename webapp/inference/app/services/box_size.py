"""グローバルトラック単位のボックスの大きさ（高さ・幅・長さ）の補正.

各フレームの箱は、隠れや見えている面の少なさで**小さめに出る**のが普通
（LiDAR の上下ビームが疎で上面が下がる、奥の面が見えない等）。剛体の
大きさはフレームによらず一定なので、グローバルトラックの全フレームから
共通の大きさを決め、各フレームの箱に当て直す。

方式（webapp の config の BOX_HEIGHT_CORRECTIONS / BOX_WIDTH_CORRECTIONS /
BOX_LENGTH_CORRECTIONS: {label: 方式}。辞書に無い・None は補正しない）:

  percentile … 全フレームのその寸法の分位点（Height / Width / Length
               Estimation の Percentile）。フレームが Min Frames に
               届かなければ補正しない
  default    … config の DEFAULT_BOX_SIZES（length, width, height）

## 当て直し方

- 高さ: **底面を固定**して上面を動かす（底面は地面などから別に決めている）
- 幅・長さ: **センサーから見えている面を固定**して反対側の面を動かす。
  軸ごとに次のように判定する（u は箱の軸の単位ベクトル、e はその軸の
  半分の長さ、d は箱の中心からセンサーへのベクトル）:

    a = d·u
    |a| > e なら、センサーは +u 側（a > 0）か −u 側（a < 0）の面の外にいる
      → その面は見えている。見えている面を固定し、反対側の面を動かす
    |a| ≤ e なら、センサーはその軸の 2 面の間（面を真横から見る位置）にいる
      → この軸の面はどちらも見えていない。中心を固定して左右対称に伸ばす

  凸な箱の面が見えるのは「面の外側の半空間にセンサーがある」ときなので、
  これで厳密に判定できる。2 軸とも見えていれば角を固定して 2 面を動かし、
  1 軸だけなら、その面を固定しつつ、もう一方の軸は（見えている面の中心
  ＝箱の中心を固定して）対称に伸ばすことになる。

NOTE: 長さと幅は各フレームの MOA の「長辺 = length、短辺 = width」を
そのまま集計する（簡易版）。後ろ姿だけのフレームでは長辺と短辺が入れ替わり
向きも 90° ずれることがあるが、トラック全体で向きをそろえる処理を入れる
まではそのままにしている。
"""
from __future__ import annotations

from typing import Any

import numpy as np

SIZE_CORRECTION_PERCENTILE = "percentile"
SIZE_CORRECTION_DEFAULT = "default"
SIZE_CORRECTION_METHODS = (SIZE_CORRECTION_PERCENTILE, SIZE_CORRECTION_DEFAULT)

DEFAULT_PERCENTILES = {"height": 80.0, "width": 70.0, "length": 80.0}
DEFAULT_MIN_FRAMES = 3
# size_wlh の並び
_INDEX = {"width": 0, "length": 1, "height": 2}
# DEFAULT_BOX_SIZES (length, width, height) の並び
_DEFAULT_INDEX = {"length": 0, "width": 1, "height": 2}


def correction_settings(
    box_fitting_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """run のパラメータから補正の設定を組み立てる（無ければ補正しない）."""
    params = (box_fitting_params or {}).get("size_correction") or {}
    return {
        dim: {
            "methods": dict((params.get(dim) or {}).get("methods") or {}),
            "percentile": float(
                (params.get(dim) or {}).get("percentile", DEFAULT_PERCENTILES[dim])
            ),
        }
        for dim in ("height", "width", "length")
    } | {
        "min_frames": int(params.get("min_frames", DEFAULT_MIN_FRAMES)),
        "box_sizes": (box_fitting_params or {}).get("box_sizes") or {},
    }


def _target_size(
    dim: str,
    label: str | None,
    values: list[float],
    settings: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """グローバルトラック共通の寸法. (値 or None＝補正しない, 診断)."""
    method = settings[dim]["methods"].get(label or "")
    if method not in SIZE_CORRECTION_METHODS:
        return None, {"method": None}
    if method == SIZE_CORRECTION_DEFAULT:
        size = settings["box_sizes"].get(label or "")
        if not size or len(size) < 3:
            return None, {"method": method, "reason": "no_box_size"}
        return float(size[_DEFAULT_INDEX[dim]]), {"method": method}
    if len(values) < settings["min_frames"]:
        return None, {"method": method, "reason": "too_few_frames"}
    pct = settings[dim]["percentile"]
    return float(np.percentile(values, pct)), {"method": method, "percentile": pct}


def _resize_axis(
    center_xy: np.ndarray,
    axis: np.ndarray,
    half: float,
    new_half: float,
    origin_xy: np.ndarray,
) -> tuple[np.ndarray, str]:
    """1 軸の大きさを変える. 見えている面を固定し、見えていなければ対称.

    Returns:
        (新しい中心, 固定の仕方 "near_face" / "center")
    """
    a = float(np.dot(origin_xy - center_xy, axis))
    if abs(a) > half:
        side = 1.0 if a > 0 else -1.0
        # 見えている面（センサー側）の位置を保ったまま、反対側の面を動かす
        near = center_xy + side * half * axis
        return near - side * new_half * axis, "near_face"
    return center_xy, "center"


def _corners(center_xy: np.ndarray, length: float, width: float, yaw: float) -> list:
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-np.sin(yaw), np.cos(yaw)])
    hl, hw = length / 2.0, width / 2.0
    pts = [center_xy - u * hl - v * hw, center_xy + u * hl - v * hw,
           center_xy + u * hl + v * hw, center_xy - u * hl + v * hw]
    return np.round(np.stack(pts), 3).tolist()


def apply_size(
    fit: dict[str, Any],
    targets: dict[str, float | None],
    origin_xy: tuple[float, float],
) -> dict[str, Any]:
    """1 フレームの箱に共通の寸法を当て直す（fit をその場で書き換える）.

    Returns:
        診断（固定の仕方と補正前の寸法）
    """
    width, length, height = (float(v) for v in fit["size_wlh"])
    cx, cy, cz = (float(v) for v in fit["center_ego"])
    yaw = float(fit["yaw_ego"])
    info: dict[str, Any] = {"before_wlh": [width, length, height]}

    center = np.array([cx, cy])
    origin = np.asarray(origin_xy, dtype=float)
    u = np.array([np.cos(yaw), np.sin(yaw)])   # 長さ方向
    v = np.array([-np.sin(yaw), np.cos(yaw)])  # 幅方向

    new_length = targets.get("length") or length
    new_width = targets.get("width") or width
    anchors: dict[str, str] = {}
    if targets.get("length"):
        center, anchors["length"] = _resize_axis(
            center, u, length / 2.0, new_length / 2.0, origin,
        )
    if targets.get("width"):
        # 長さ方向で動かした後の中心で判定する（幅方向の位置は変わらない）
        center, anchors["width"] = _resize_axis(
            center, v, width / 2.0, new_width / 2.0, origin,
        )

    new_height = height
    if targets.get("height"):
        # 底面を固定して上面を動かす
        bottom = cz - height / 2.0
        new_height = float(targets["height"])
        cz = bottom + new_height / 2.0
        anchors["height"] = "bottom"

    fit["center_ego"] = [round(float(center[0]), 3), round(float(center[1]), 3),
                         round(cz, 3)]
    fit["size_wlh"] = [round(new_width, 3), round(new_length, 3), round(new_height, 3)]
    metrics = fit.setdefault("fit_metrics", {})
    if "corners_xy" in metrics:
        metrics["corners_xy"] = _corners(center, new_length, new_width, yaw)
    info["anchors"] = anchors
    return info


def correct_box_sizes(
    fitted: list[tuple[str, str | None, dict[str, Any], tuple[float, float]]],
    box_fitting_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """グローバルトラック単位で寸法を補正する.

    Args:
        fitted: ``(global_track_id, label, 当てはめ結果の行, センサー位置)`` のリスト。
            行（fit の値を持つ dict）は**その場で書き換える**

    Returns:
        ``{"num_tracks", "num_boxes"}``（補正した数）
    """
    settings = correction_settings(box_fitting_params)
    if not any(settings[d]["methods"] for d in ("height", "width", "length")):
        return {"num_tracks": 0, "num_boxes": 0}

    by_track: dict[str, list[tuple[str | None, dict[str, Any], tuple[float, float]]]] = {}
    for global_id, label, row, origin in fitted:
        by_track.setdefault(global_id, []).append((label, row, origin))

    num_tracks = num_boxes = 0
    for global_id, rows in by_track.items():
        label = rows[0][0]
        targets: dict[str, float | None] = {}
        details: dict[str, Any] = {"num_frames": len(rows)}
        for dim in ("height", "width", "length"):
            values = [float(r["size_wlh"][_INDEX[dim]]) for _, r, _ in rows]
            target, detail = _target_size(dim, label, values, settings)
            targets[dim] = target
            if target is not None:
                detail["value"] = round(target, 3)
            details[dim] = detail
        if not any(targets.values()):
            continue
        num_tracks += 1
        for _label, row, origin in rows:
            info = apply_size(row, targets, origin)
            row.setdefault("fit_metrics", {})["size_correction"] = {**details, **info}
            num_boxes += 1
    return {"num_tracks": num_tracks, "num_boxes": num_boxes}
