"""ボックスの規定長と向きに基づくカメラ間の同一インスタンス結合.

凸包の IoS による結合（inter_cam_merge）は、カメラの視野の重なりが小さいと
凸包が重ならず、同じ物体の前後（長い車両の前部と後部など）を結合できない。
角度を揃えた後のボックスどうしで「同じラベル・同じ向き・合体しても規定長に
収まる」組を結合し、global_track_id を割り当てる。

単位は hull_merged_track × キーフレームのボックス（代表行）。そのキーフレームで
使っているカメラの集合が重なる組は照合しない。

## 合体後のボックス（照合の暫定ボックスと結合後のボックスで共通）

- yaw: 結合前の点数で重み付けした平均（180 度違う向きは同じ軸とみなして揃える）
- XY: 両方の凸包をまとめ、その yaw の軸で囲む矩形
- z: 下面は低い方、上面は高い方
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from app.services.box_orientation import wrap_angle

DEFAULT_ORIENTATION_DIFF_DEG = 15.0
DEFAULT_LENGTH_RATIO = 1.1
DEFAULT_WIDTH_RATIO = 1.1
DEFAULT_MAX_HULL_GAP = 0.5
DEFAULT_MIN_MATCH_FRAMES = 1

# 結果の記録先（行の fit_metrics のキー）
BOX_MERGE_METRICS_KEY = "box_merge"
# 診断に残すトラック対の上限と、残す範囲（凸包の隙間）
MAX_DIAGNOSTIC_EDGES = 300
DIAGNOSTIC_GAP_LIMIT = 3.0


def merge_settings(
    merge_params: dict[str, Any] | None, box_fitting_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """run のパラメータから設定を取り出す（ラベルが無ければ結合しない）."""
    params = (merge_params or {}).get("box_orient") or {}
    return {
        "labels": set(params.get("labels") or ()),
        "orientation_diff_deg": float(
            params.get("orientation_diff_deg", DEFAULT_ORIENTATION_DIFF_DEG)),
        "length_ratio": float(params.get("length_ratio", DEFAULT_LENGTH_RATIO)),
        "width_ratio": float(params.get("width_ratio", DEFAULT_WIDTH_RATIO)),
        "max_hull_gap": float(params.get("max_hull_gap", DEFAULT_MAX_HULL_GAP)),
        "min_match_frames": int(params.get("min_match_frames", DEFAULT_MIN_MATCH_FRAMES)),
        # 同一カメラのトラックを同居させてよい隔たり（凸包の結合と同じ値）
        "max_same_camera_gap": (merge_params or {}).get("max_same_camera_gap"),
        # (length, width, height)
        "box_sizes": (box_fitting_params or {}).get("box_sizes") or {},
    }


def axis_diff_deg(yaw_a: float, yaw_b: float) -> float:
    """2 つの向きの軸の差 [度]（180 度違いは同じとみなす。0〜90）."""
    diff = abs(wrap_angle(yaw_a - yaw_b)) % math.pi
    return math.degrees(min(diff, math.pi - diff))


def merged_box(units: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """複数のボックスを合体したボックス {center_ego, size_wlh, yaw_ego, hull}.

    units は ``{"row", "weight"}``（row は center_ego / size_wlh / yaw_ego / hull_xy を持つ）。
    凸包が無ければ None。
    """
    hulls = [np.asarray((u["row"].get("hull_xy") or {}).get("points") or [], dtype=float)
             for u in units]
    hulls = [h[:, :2] for h in hulls if h.ndim == 2 and h.shape[0]]
    if not hulls:
        return None
    # 重みが最大のボックスの向きを基準に、180 度違う向きを同じ側へ揃えてから平均する
    reference = max(units, key=lambda u: u["weight"])
    base = float(reference["row"]["yaw_ego"])
    sin_sum = cos_sum = 0.0
    for unit in units:
        yaw = float(unit["row"]["yaw_ego"])
        if abs(wrap_angle(yaw - base)) > math.pi / 2.0:
            yaw += math.pi
        weight = max(float(unit["weight"]), 1e-6)
        sin_sum += weight * math.sin(yaw)
        cos_sum += weight * math.cos(yaw)
    yaw = math.atan2(sin_sum, cos_sum)

    points = np.vstack(hulls)
    u = np.array([math.cos(yaw), math.sin(yaw)])   # 長さ方向
    v = np.array([-math.sin(yaw), math.cos(yaw)])  # 幅方向
    pu, pv = points @ u, points @ v
    center_xy = ((pu.max() + pu.min()) / 2.0) * u + ((pv.max() + pv.min()) / 2.0) * v
    bottoms, tops = [], []
    for unit in units:
        cz, h = float(unit["row"]["center_ego"][2]), float(unit["row"]["size_wlh"][2])
        bottoms.append(cz - h / 2.0)
        tops.append(cz + h / 2.0)
    bottom, top = min(bottoms), max(tops)
    return {
        "center_ego": [float(center_xy[0]), float(center_xy[1]), (bottom + top) / 2.0],
        "size_wlh": [float(pv.max() - pv.min()), float(pu.max() - pu.min()), top - bottom],
        "yaw_ego": yaw,
        "hull": points,
    }


def select_merge_boxes(
    same: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """同じキーフレーム・同じ global_track_id のボックスのうち、合体に使うものを選ぶ.

    凸包による結合の ``inter_cam_merge.select_merge_members`` と同じ規則:

    - LiDAR 点群を持つボックスがあれば、それらだけを合体する（持たないものは捨てる）
    - どれも持たなければ、点数が最も多いボックスだけを使う（合体しない）

    LiDAR 点群の有無は、当てはめた点群に LiDAR 点が含まれるか（``has_lidar``）。

    Returns:
        (合体に使うボックス, 選び方 MERGE_SELECTION_*)
    """
    from app.services.inter_cam_merge import (
        MERGE_SELECTION_ALL,
        MERGE_SELECTION_LIDAR,
        MERGE_SELECTION_MOST_POINTS,
    )

    with_lidar = [g for g in same if g.get("has_lidar")]
    if len(with_lidar) == len(same):
        return list(same), MERGE_SELECTION_ALL
    if with_lidar:
        return with_lidar, MERGE_SELECTION_LIDAR
    return [max(same, key=lambda g: float(g.get("num_points") or 0))], \
        MERGE_SELECTION_MOST_POINTS


def _check_pair(
    a: dict[str, Any], b: dict[str, Any], settings: dict[str, Any],
) -> tuple[str | None, dict[str, float]]:
    """1 つのキーフレームで 2 つのボックスを照合する. (落ちた理由 or None, 値)."""
    from app.services.inter_cam_merge import hull_gap

    gap = hull_gap(_hull_of(a["row"]), _hull_of(b["row"]))
    values: dict[str, float] = {"gap": gap}
    if gap > settings["max_hull_gap"]:
        return "gap_too_large", values
    if a["label"] != b["label"]:
        return "label_mismatch", values
    diff = axis_diff_deg(float(a["row"]["yaw_ego"]), float(b["row"]["yaw_ego"]))
    values["orientation_diff"] = diff
    if diff > settings["orientation_diff_deg"]:
        return "orientation_diff", values
    size = settings["box_sizes"].get(a["label"] or "")
    box = merged_box([a, b])
    if box is None or not size or len(size) < 2:
        return "too_long", values
    values["length"], values["width"] = box["size_wlh"][1], box["size_wlh"][0]
    if box["size_wlh"][1] > float(size[0]) * settings["length_ratio"]:
        return "too_long", values
    if box["size_wlh"][0] > float(size[1]) * settings["width_ratio"]:
        return "too_wide", values
    return None, values


def _hull_of(row: dict[str, Any]) -> np.ndarray:
    points = np.asarray((row.get("hull_xy") or {}).get("points") or [], dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.empty((0, 3))
    return np.column_stack([points[:, :2], np.zeros(points.shape[0])])


def _match_frame(
    units: list[dict[str, Any]], settings: dict[str, Any],
) -> tuple[list[tuple[int, int, dict[str, float]]], list[tuple[int, int, str, dict]]]:
    """1 つのキーフレームの照合. (対応付けた組, 落ちた組と理由).

    主カメラ（代表行のカメラ）の対ごとに Hungarian algorithm で 1 対 1 に対応付ける
    （コストは凸包の隙間、同じなら向きの差）。
    """
    from scipy.optimize import linear_sum_assignment

    by_camera: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        by_camera.setdefault(unit["camera"], []).append(index)
    cameras = sorted(by_camera)

    matched: list[tuple[int, int, dict[str, float]]] = []
    failed: list[tuple[int, int, str, dict]] = []
    for ci, cam_a in enumerate(cameras):
        for cam_b in cameras[ci + 1:]:
            rows, cols = by_camera[cam_a], by_camera[cam_b]
            cost = np.full((len(rows), len(cols)), np.inf)
            values: dict[tuple[int, int], dict[str, float]] = {}
            for i, a in enumerate(rows):
                for j, b in enumerate(cols):
                    if units[a]["cameras"] & units[b]["cameras"]:
                        continue  # 同じカメラを使っている組は照合しない
                    reason, vals = _check_pair(units[a], units[b], settings)
                    if reason is not None:
                        failed.append((a, b, reason, vals))
                        continue
                    values[(a, b)] = vals
                    cost[i, j] = vals["gap"] + 1e-3 * vals["orientation_diff"]
            if not values:
                continue
            finite = np.where(np.isfinite(cost), cost, 1e6)
            for i, j in zip(*linear_sum_assignment(finite), strict=True):
                a, b = rows[i], cols[j]
                if (a, b) in values:
                    matched.append((a, b, values.pop((a, b))))
            # 条件は満たしたが 1 対 1 の割り当てで別の相手に取られた組
            for (a, b), vals in values.items():
                failed.append((a, b, "taken_by_other", vals))
    return matched, failed


def assign_box_merged_ids(
    groups: Sequence[dict[str, Any]],
    track_members: dict[str, list[tuple[str, set[int]]]],
    merge_params: dict[str, Any] | None,
    box_fitting_params: dict[str, Any] | None,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """ボックスどうしを照合して global_track_id を割り当て、結合したボックスを作り直す.

    Args:
        groups: 当てはめたボックス（``{"hull_merged_track_id", "sample_token",
            "label", "primary", "members", "num_points"}``）。primary は
            その場で書き換える（結合されて代表でなくなった行は status="merged"）
        track_members: hull_merged_track_id → [(カメラ, 存在するキーフレーム番号の集合)]
            （同一カメラのトラックの同居を判定する）
        enabled: False なら照合せず hull_merged_track_id をそのまま使う

    Returns:
        ``{"global_ids": {hull_merged_track_id: global_track_id},
        "num_merged_boxes", "diagnostics"}``
    """
    from app.services.inter_cam_merge import UnionFind, _channel_of, group_matched_pairs

    settings = merge_settings(merge_params, box_fitting_params)
    hull_ids = sorted(track_members, key=_id_sort_key)
    identity = {hull_id: hull_id for hull_id in hull_ids}
    if not enabled or not settings["labels"]:
        return {"global_ids": identity, "num_merged_boxes": 0, "diagnostics": None}

    # --- キーフレームごとの照合 -------------------------------------------
    units_by_sample: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        if group["label"] not in settings["labels"]:
            continue
        primary = group["primary"]
        units_by_sample.setdefault(group["sample_token"], []).append({
            "track": group["hull_merged_track_id"],
            "label": group["label"],
            "row": primary,
            "weight": float(group.get("num_points") or 0),
            "camera": _channel_of(primary),
            "cameras": {_channel_of(m) for m in group["members"]},
        })

    # {(トラック, トラック): {"frames", "min_gap", "samples"}}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    # サンプルごとの照合結果（診断用。{(トラック, トラック): [{"sample_token", "result", "values"}]}）
    sample_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    diagnostic_gap_limit = max(DIAGNOSTIC_GAP_LIMIT, 3.0 * settings["max_hull_gap"])

    def record(pair: tuple[str, str], sample_token: str, result: str, vals: dict) -> None:
        sample_results.setdefault(pair, []).append({
            "sample_token": sample_token, "result": result,
            "values": {k: round(float(v), 3) for k, v in vals.items()},
        })

    for sample_token, units in units_by_sample.items():
        matched, failed = _match_frame(units, settings)
        for a, b, vals in matched:
            pair = tuple(sorted((units[a]["track"], units[b]["track"])))
            entry = edges.setdefault(pair, {"frames": 0, "min_gap": math.inf, "samples": []})
            entry["frames"] += 1
            entry["min_gap"] = min(entry["min_gap"], vals["gap"])
            entry["samples"].append(sample_token)
            record(pair, sample_token, "matched", vals)
        for a, b, reason, vals in failed:
            if vals.get("gap", math.inf) > diagnostic_gap_limit:
                continue
            pair = tuple(sorted((units[a]["track"], units[b]["track"])))
            record(pair, sample_token, reason, vals)

    # --- トラック対を隙間の小さい順に結合 -----------------------------------
    index_of = {hull_id: i for i, hull_id in enumerate(hull_ids)}
    union = UnionFind(len(hull_ids))
    members = {i: {i} for i in range(len(hull_ids))}
    max_same_camera_gap = settings["max_same_camera_gap"]
    rejected: dict[tuple[str, str], str] = {}

    def conflict(group_a: set[int], group_b: set[int]) -> str | None:
        for i in group_a:
            for j in group_b:
                for channel_a, frames_a in track_members[hull_ids[i]]:
                    for channel_b, frames_b in track_members[hull_ids[j]]:
                        if channel_a != channel_b:
                            continue
                        if frames_a & frames_b:
                            return "同一カメラで同時に存在"
                        if not max_same_camera_gap:
                            return "同一カメラが重複"
                        if frames_a and frames_b:
                            gap = max(min(frames_b) - max(frames_a),
                                      min(frames_a) - max(frames_b))
                            if gap > int(max_same_camera_gap):
                                return f"同一カメラで時間が離れすぎ（{gap} フレーム）"
        return None

    for pair, stats in sorted(edges.items(), key=lambda kv: (kv[1]["min_gap"], -kv[1]["frames"])):
        if stats["frames"] < settings["min_match_frames"]:
            rejected[pair] = f"マッチしたフレーム数が {settings['min_match_frames']} 未満"
            continue
        a, b = index_of[pair[0]], index_of[pair[1]]
        root_a, root_b = union.find(a), union.find(b)
        if root_a == root_b:
            continue
        reason = conflict(members[root_a], members[root_b])
        if reason:
            rejected[pair] = reason
            continue
        union.union(a, b)
        root = union.find(a)
        members[root] = members[root_a] | members[root_b]

    global_ids: dict[str, str] = {}
    group_rows = []
    for indexes in union.groups():
        ids = sorted((hull_ids[i] for i in indexes), key=_id_sort_key)
        for hull_id in ids:
            global_ids[hull_id] = ids[0]
        group_rows.append({"global_track_id": ids[0], "members": ids})
    # グループ内で対応付けられた対と、そのサンプル（結合が認められたサンプル）
    sample_order = {token: i for i, token in enumerate(units_by_sample)}
    matched_samples = {tuple(sorted(pair, key=_id_sort_key)): stats["samples"]
                       for pair, stats in edges.items()}
    for row in group_rows:
        row["accepted_pairs"] = group_matched_pairs(
            row["members"], matched_samples, list(units_by_sample))

    # --- 結合したキーフレームのボックスを作り直す ------------------------------
    num_merged_boxes = 0
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for group in groups:
        key = (group["sample_token"], global_ids.get(group["hull_merged_track_id"],
                                                     group["hull_merged_track_id"]))
        by_key.setdefault(key, []).append(group)
    for same in by_key.values():
        if len(same) < 2:
            continue
        # 合体に使うボックス（凸包による結合の select_merge_members と同じ規則）
        used, selection = select_merge_boxes(same)
        keep = max(used, key=lambda g: float(g.get("num_points") or 0))
        primary = keep["primary"]
        before = {key: primary.get(key) for key in ("center_ego", "size_wlh", "yaw_ego")}
        if len(used) > 1:
            box = merged_box([{"row": g["primary"], "weight": float(g.get("num_points") or 0)}
                              for g in used])
            if box is None:
                continue
            primary["center_ego"] = [round(v, 3) for v in box["center_ego"]]
            primary["size_wlh"] = [round(v, 3) for v in box["size_wlh"]]
            primary["yaw_ego"] = box["yaw_ego"]
            from app.services.box_fitting import convex_hull_xy

            primary["hull_xy"] = {**(primary.get("hull_xy") or {}),
                                  "points": np.round(convex_hull_xy(box["hull"]), 3).tolist()}
        primary.setdefault("fit_metrics", {})[BOX_MERGE_METRICS_KEY] = {
            "merged_from": [g["primary"]["instance_tracking_2d_id"]
                            for g in same if g is not keep],
            "hull_merged_track_ids": [g["hull_merged_track_id"] for g in same],
            # どのボックスを合体したか（all / lidar / most_points）。捨てたボックスも残す
            "selection": selection,
            "used": [g["primary"]["instance_tracking_2d_id"] for g in used],
            "dropped": [g["primary"]["instance_tracking_2d_id"]
                        for g in same if not any(g is u for u in used)],
            "before": before,
        }
        for group in same:
            if group is keep:
                continue
            row = group["primary"]
            row["status"] = "merged"
            row["is_primary"] = False
            row.setdefault("fit_metrics", {})[BOX_MERGE_METRICS_KEY] = {
                "merged_into": primary["instance_tracking_2d_id"],
            }
        num_merged_boxes += len(same) - 1

    # --- 診断 ---------------------------------------------------------------
    # トラック対ごとに 1 行。サンプルごとの照合結果を "samples" に全件持つ
    edge_rows = []
    for pair, results in sample_results.items():
        stats = edges.get(pair)
        same_group = global_ids[pair[0]] == global_ids[pair[1]]
        if stats and same_group:
            status, reason = "accepted", ""
        elif stats:
            status, reason = "rejected", rejected.get(pair, "")
        elif same_group:
            status, reason = "accepted", "他のトラックを介して結合"
        else:
            gated = all(r["result"] in ("label_mismatch", "gap_too_large") for r in results)
            status = "gated" if gated else "rejected"
            reason = "対応付けられたサンプルが無い"
        results.sort(key=lambda r: sample_order[r["sample_token"]])
        edge_rows.append({
            "pair": list(pair), "frames": stats["frames"] if stats else 0,
            "min_gap": round(min(r["values"].get("gap", math.inf) for r in results), 3),
            "status": status, "reason": reason,
            "samples": [
                {**r, "reason": "" if r["result"] == "matched"
                 else _reason_text(r["result"], r["values"], settings)}
                for r in results
            ],
        })
    edge_rows.sort(key=lambda r: ({"accepted": 0, "rejected": 1, "gated": 2}[r["status"]],
                                  r["min_gap"] if r["min_gap"] is not None else math.inf))
    diagnostics = {
        "groups": [g for g in group_rows if len(g["members"]) > 1],
        "num_groups": len(group_rows),
        "edges": edge_rows[:MAX_DIAGNOSTIC_EDGES],
        "num_edges_total": len(edge_rows),
        "num_merged_boxes": num_merged_boxes,
        "params": {
            "labels": sorted(settings["labels"]),
            "orientation_diff_deg": settings["orientation_diff_deg"],
            "length_ratio": settings["length_ratio"],
            "width_ratio": settings["width_ratio"],
            "max_hull_gap": settings["max_hull_gap"],
            "min_match_frames": settings["min_match_frames"],
            "max_same_camera_gap": max_same_camera_gap,
        },
    }
    return {"global_ids": global_ids, "num_merged_boxes": num_merged_boxes,
            "diagnostics": diagnostics}


def _reason_text(reason: str, values: dict[str, float], settings: dict[str, Any]) -> str:
    if reason == "label_mismatch":
        return "ラベルが違う"
    if reason == "gap_too_large":
        return f"凸包の隙間 {values.get('gap')} m が上限（{settings['max_hull_gap']} m）を超えた"
    if reason == "orientation_diff":
        return (f"向きの差 {values.get('orientation_diff')} 度が上限"
                f"（{settings['orientation_diff_deg']} 度）を超えた")
    if reason == "too_long":
        return f"合体後の長さ {values.get('length', '-')} m が規定長 × {settings['length_ratio']} を超えた"
    if reason == "too_wide":
        return f"合体後の幅 {values.get('width', '-')} m が規定幅 × {settings['width_ratio']} を超えた"
    return "1 対 1 の割り当てで別の相手に取られた"


def _id_sort_key(track_id: str) -> tuple[int, int, str]:
    """ID の並び（数字の ID は数値順）."""
    return (0, int(track_id), "") if str(track_id).isdigit() else (1, 0, str(track_id))
