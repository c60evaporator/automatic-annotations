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

## 補正の順番

高さ → 角度（services/box_orientation_correction.py）→ 見えている面の判定 → 幅 → 長さ。
幅・長さの共通値は、角度を補正した後の寸法から決める（角度の補正で
幅と長さが入れ替わることがあるため）。

## 当て直し方

- 高さ: **底面を固定**して上面を動かす（底面は地面などから別に決めている）
- 幅・長さ: **センサーから見えている面を固定**して反対側の面を動かす。
  見えている面は、幅・長さを動かす前の箱で 2 軸まとめて判定する
  （幅方向に中心を動かしても長さ方向の a = d·u は変わらないので、
  逐次判定と結果は同じ）。軸ごとに次のように判定する（u は箱の軸の単位ベクトル、e はその軸の
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


def _visible_side(
    center_xy: np.ndarray, axis: np.ndarray, half: float, origin_xy: np.ndarray,
) -> float:
    """その軸の見えている面. +1 / -1（+axis / -axis 側の面）、0 は見えていない."""
    a = float(np.dot(origin_xy - center_xy, axis))
    if abs(a) > half:
        return 1.0 if a > 0 else -1.0
    return 0.0


def _resize_axis(
    center_xy: np.ndarray,
    axis: np.ndarray,
    half: float,
    new_half: float,
    side: float,
) -> tuple[np.ndarray, str]:
    """1 軸の大きさを変える. 見えている面（side）を固定し、見えていなければ対称.

    Returns:
        (新しい中心, 固定の仕方 "near_face" / "center")
    """
    if side:
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


def apply_height(fit: dict[str, Any], height: float) -> None:
    """底面を固定して上面を動かす（fit をその場で書き換える）."""
    cx, cy, cz = (float(v) for v in fit["center_ego"])
    old = float(fit["size_wlh"][2])
    bottom = cz - old / 2.0
    fit["center_ego"] = [cx, cy, round(bottom + height / 2.0, 3)]
    fit["size_wlh"] = [float(fit["size_wlh"][0]), float(fit["size_wlh"][1]),
                       round(float(height), 3)]


def apply_width_length(
    fit: dict[str, Any],
    targets: dict[str, float | None],
    origin_xy: tuple[float, float],
) -> dict[str, Any]:
    """見えている面を判定してから、幅 → 長さの順に当て直す（fit をその場で書き換える）.

    Returns:
        診断（見えている面と固定の仕方）
    """
    width, length, height = (float(v) for v in fit["size_wlh"])
    cx, cy, cz = (float(v) for v in fit["center_ego"])
    yaw = float(fit["yaw_ego"])

    center = np.array([cx, cy])
    origin = np.asarray(origin_xy, dtype=float)
    u = np.array([np.cos(yaw), np.sin(yaw)])   # 長さ方向
    v = np.array([-np.sin(yaw), np.cos(yaw)])  # 幅方向

    # 見えている面の判定（幅・長さを動かす前の箱で 2 軸まとめて）
    visible = {
        "width": _visible_side(center, v, width / 2.0, origin),
        "length": _visible_side(center, u, length / 2.0, origin),
    }
    new_width = targets.get("width") or width
    new_length = targets.get("length") or length
    anchors: dict[str, str] = {}
    if targets.get("width"):
        center, anchors["width"] = _resize_axis(
            center, v, width / 2.0, new_width / 2.0, visible["width"],
        )
    if targets.get("length"):
        center, anchors["length"] = _resize_axis(
            center, u, length / 2.0, new_length / 2.0, visible["length"],
        )

    fit["center_ego"] = [round(float(center[0]), 3), round(float(center[1]), 3),
                         round(cz, 3)]
    fit["size_wlh"] = [round(new_width, 3), round(new_length, 3), round(height, 3)]
    metrics = fit.setdefault("fit_metrics", {})
    if "corners_xy" in metrics:
        metrics["corners_xy"] = _corners(center, new_length, new_width, yaw)
    return {"visible": {k: int(side) for k, side in visible.items()}, "anchors": anchors}


def correct_box_sizes(
    fitted: list[tuple[str, str | None, dict[str, Any], tuple[float, float], str]],
    box_fitting_params: dict[str, Any] | None,
    *,
    reference_ego_poses: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """グローバルトラック単位で高さ・角度・幅・長さを補正する.

    Args:
        fitted: ``(global_track_id, label, 当てはめ結果の行, センサー位置, sample_token)``
            のリスト。行（fit の値を持つ dict）は**その場で書き換える**
        reference_ego_poses: sample_token → 基準 ego_pose（角度補正に使う）

    Returns:
        ``{"num_tracks", "num_boxes", "num_orientation_corrected"}``（補正した数）
    """
    from app.services.box_fitting import BOX_STAGE_ORIENTATION_CORRECTED, record_box_stage
    from app.services.box_orientation_correction import (
        correct_track_orientation,
        correction_settings as orientation_correction_settings,
    )

    settings = correction_settings(box_fitting_params)
    orientation_settings = orientation_correction_settings(box_fitting_params)

    by_track: dict[str, list[tuple[str | None, dict[str, Any], tuple[float, float], str]]] = {}
    for global_id, label, row, origin, sample_token in fitted:
        by_track.setdefault(global_id, []).append((label, row, origin, sample_token))

    num_tracks = num_boxes = num_orientation_corrected = 0
    for global_id, rows in by_track.items():
        label = rows[0][0]
        details: dict[str, Any] = {"num_frames": len(rows)}
        befores = [[float(x) for x in r["size_wlh"]] for _, r, _, _ in rows]

        def target(dim: str) -> float | None:
            values = [float(r["size_wlh"][_INDEX[dim]]) for _, r, _, _ in rows]
            value, detail = _target_size(dim, label, values, settings)
            if value is not None:
                detail["value"] = round(value, 3)
            details[dim] = detail
            return value

        # 1. 高さ（底面を固定して上面を動かす）
        height = target("height")
        if height is not None:
            for _label, row, _origin, _sample in rows:
                apply_height(row, height)

        # 2. 角度（BOX_ORIENTATION_CORRECTIONS）
        num_orientation_corrected += correct_track_orientation(
            label,
            [row for _, row, _, _ in rows],
            [sample for _, _, _, sample in rows],
            [(reference_ego_poses or {}).get(sample) for _, _, _, sample in rows],
            orientation_settings,
        )
        for _label, row, _origin, _sample in rows:
            record_box_stage(row, BOX_STAGE_ORIENTATION_CORRECTED)

        # 3〜5. 見えている面の判定 → 幅 → 長さ。共通値は角度補正後の寸法から決める
        targets = {"width": target("width"), "length": target("length")}
        if height is None and not any(targets.values()):
            continue
        num_tracks += 1
        for (_label, row, origin, _sample), before in zip(rows, befores, strict=True):
            info = apply_width_length(row, targets, origin)
            if height is not None:
                info["anchors"] = {"height": "bottom", **info["anchors"]}
            row.setdefault("fit_metrics", {})["size_correction"] = {
                **details, "before_wlh": before, **info,
            }
            num_boxes += 1
    return {"num_tracks": num_tracks, "num_boxes": num_boxes,
            "num_orientation_corrected": num_orientation_corrected}
