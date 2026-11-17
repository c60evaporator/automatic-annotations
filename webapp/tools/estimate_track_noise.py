"""Track Smoother の観測ノイズとプロセスノイズを GT から推定する.

- 観測ノイズ（config の TRACK_SMOOTHER_MEASUREMENT_NOISE）:
  Depth Boxfitting の run の box_merged 段階のボックス（Track Smoother へ渡す観測）と
  GT の中心の残差を、ボックスの軸（長さ方向・幅方向）に分解し、LiDAR 点の有無ごとに
  距離の区間で σ = 1.4826 × median(|残差|) を求めて、σ = a + b·d を区間の件数で重み付けた
  最小二乗で当てはめる（中央値の周りではなく 0 の周りで測るのは、長さ方向の
  系統的な偏りも観測の誤差に含めるため）。
- プロセスノイズ（config の TRACK_SMOOTHER_PROCESS_NOISE）:
  GT のインスタンスごとに連続する 3 フレームから速度の差 Δv を求め、等速度モデルの
  Var(Δv_x) = Var(Δv_y) ≈ q·Δt から、ラベルごとに q = median(|Δv|² / Δt) / 1.386 を出す
  （|Δv|²/(q·Δt) は自由度 2 の χ² に従い、その中央値が 2·ln2 = 1.386。外れ値に
  引きずられないよう中央値で測る）。止まっている物体の Δv ≈ 0 で中央値が 0 に潰れるので、
  速さ（前後 2 区間の平均）が MIN_MOVING_SPEED 以上の 3 フレームだけを使う
  （q は移動中の物体の加速度の強さなので）。

webapp コンテナの中で動かす。使い方（docker-compose.yml のある webapp フォルダで）:

    docker compose run --rm -T webapp python - [オプション] < tools/estimate_track_noise.py

オプション（いずれも省略可）:
  --runs RUN_ID ...   観測ノイズを測る run（前方一致）。省略すると、ボックスの規定長と向きに
                      基づく結合（box_merge_diagnostics）を記録している成功 run を、シーンごとに
                      最新の 1 件ずつ使う
  --max-match M       ボックスと GT を対応付ける中心距離の上限 [m]（既定 3.0）

注意: GT との対応付けは最も近い GT で行うので、max-match を超える大きな誤差は数えない
（σ は実際よりやや小さく出る）。
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import read_only_session
from app.models.ann_intermediate import DepthEstimationParams
from app.models.scene import Scene
from app.repositories.annotation import AnnotationRepository
from app.repositories.sensor import SensorRepository
from app.services.depth_boxfitting_service import load_box_fittings
from app.services.geometry.pointcloud import transform_ego_to_global
from app.services.geometry.transform import quaternion_to_yaw
from app.services.label_service import nusc_category_to_label

DISTANCE_BINS = [0.0, 10.0, 20.0, 30.0, 45.0, 80.0]
MAD_SCALE = 1.4826
CHI2_2_MEDIAN = 2.0 * math.log(2.0)
MIN_MOVING_SPEED = 1.0  # [m/s]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="estimate_track_noise")
    parser.add_argument("--runs", nargs="*", help="run id（前方一致）")
    parser.add_argument("--max-match", type=float, default=3.0)
    return parser.parse_args()


def _select_runs(prefixes: list[str] | None) -> list[tuple[str, str]]:
    """(run id, scene_token) のリスト."""
    with read_only_session() as session:
        rows = session.execute(
            select(DepthEstimationParams.id, DepthEstimationParams.scene_token,
                   DepthEstimationParams.box_merge_diagnostics.is_not(None))
            .where(DepthEstimationParams.status == "succeeded")
            .order_by(DepthEstimationParams.started_at.desc())
        ).all()
    if prefixes:
        return [(i, s) for i, s, _ in rows if any(i.startswith(p) for p in prefixes)]
    latest: dict[str, str] = {}
    for run_id, scene, has_box_merge in rows:
        if has_box_merge and scene not in latest:
            latest[scene] = run_id
    return [(run_id, scene) for scene, run_id in latest.items()]


def _fit_line(rows: list[tuple[float, float, int]]) -> tuple[float, float]:
    """(区間の中心距離, σ, 件数) に σ = a + b·d を重み付き最小二乗で当てはめる（a, b ≥ 0）."""
    d = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    w = np.sqrt(np.array([r[2] for r in rows], dtype=float))
    if len(rows) == 1:
        return float(s[0]), 0.0
    a, b = np.linalg.lstsq(np.stack([np.ones_like(d), d], axis=1) * w[:, None], s * w,
                           rcond=None)[0]
    if b < 0:
        return float(np.average(s, weights=w ** 2)), 0.0
    if a < 0:
        b = float((d * s * w ** 2).sum() / (d * d * w ** 2).sum())
        return 0.0, b
    return float(a), float(b)


def measurement_noise(runs: list[tuple[str, str]], max_match: float) -> None:
    channel = get_settings().EGO_REFERENCE_CHANNEL
    # (lidar/depth, length/width) → [(距離, 残差)]
    residuals: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for run_id, scene_token in runs:
        with read_only_session() as session:
            frames = SensorRepository(session).list_frames_by_scene(scene_token)
            gts = AnnotationRepository(session).list_by_scene(
                scene_token, source="imported", include_attributes=False)
        sample_of = {f["token"]: f["sample_token"] for f in frames}
        pose_of = {f["sample_token"]: f["ego_pose"] for f in frames
                   if f["channel"] == channel and f.get("ego_pose")}
        gt_by_sample: dict[str, list] = defaultdict(list)
        for g in gts:
            label = nusc_category_to_label(g["instance"]["category_name"])
            if label:
                gt_by_sample[g["sample_token"]].append((label, np.array(g["translation"][:2])))
        n = 0
        for f in (f for items in load_box_fittings(run_id).values() for f in items):
            box = (f.get("fit_metrics") or {}).get("box_merged") or f
            sample = sample_of.get(f["sample_data_token"])
            pose = pose_of.get(sample)
            if (not f.get("is_primary") or f.get("status") != "fitted" or pose is None
                    or not box.get("center_ego")):
                continue
            center = np.asarray(box["center_ego"], dtype=float)
            xy = transform_ego_to_global(center[None, :], pose["translation"],
                                         pose["rotation"])[0, :2]
            candidates = [c for label, c in gt_by_sample.get(sample, []) if label == f["label"]]
            if not candidates:
                continue
            gt = min(candidates, key=lambda c: np.linalg.norm(c - xy))
            if np.linalg.norm(gt - xy) > max_match:
                continue
            yaw = (box.get("yaw_ego") or 0.0) + quaternion_to_yaw(pose["rotation"])
            u = np.array([math.cos(yaw), math.sin(yaw)])
            v = np.array([-u[1], u[0]])
            kind = "lidar" if int(f.get("num_points_lidar") or 0) > 0 else "depth"
            distance = float(np.linalg.norm(center[:2]))
            residuals[(kind, "length")].append((distance, float((xy - gt) @ u)))
            residuals[(kind, "width")].append((distance, float((xy - gt) @ v)))
            n += 1
        print(f"run {run_id[:8]} scene {scene_token[:8]}: {n} boxes matched")

    print("\n## 観測ノイズ σ [m]（距離の区間ごと、1.4826 × median|残差|）")
    result: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for (kind, axis), values in sorted(residuals.items()):
        values_arr = np.array(values)
        rows = []
        for lo, hi in zip(DISTANCE_BINS, DISTANCE_BINS[1:]):
            sel = values_arr[(values_arr[:, 0] >= lo) & (values_arr[:, 0] < hi)]
            if len(sel) < 10:
                continue
            sigma = MAD_SCALE * float(np.median(np.abs(sel[:, 1])))
            rows.append((float(np.median(sel[:, 0])), sigma, len(sel)))
            print(f"  {kind:5s} {axis:6s} {lo:4.0f}-{hi:3.0f} m: n={len(sel):4d} "
                  f"σ={sigma:.3f} 偏り(median)={np.median(sel[:, 1]):+.3f}")
        if rows:
            a, b = _fit_line(rows)
            result[kind][axis] = [round(a, 3), round(b, 4)]
            print(f"  → {kind} {axis}: a={a:.3f}, b={b:.4f}")
    print("\nTRACK_SMOOTHER_MEASUREMENT_NOISE =", dict(result))


def process_noise() -> None:
    with read_only_session() as session:
        scenes = session.execute(select(Scene.token)).scalars().all()
        stats: dict[str, list[float]] = defaultdict(list)
        repo = AnnotationRepository(session)
        for scene in scenes:
            tracks: dict[str, list] = defaultdict(list)
            for g in repo.list_by_scene(scene, source="imported", include_attributes=False):
                label = nusc_category_to_label(g["instance"]["category_name"])
                if label:
                    tracks[(label, g["instance"]["token"])].append(
                        (g["sample_timestamp"] * 1e-6, np.array(g["translation"][:2])))
            for (label, _), obs in tracks.items():
                obs.sort(key=lambda o: o[0])
                for (t0, p0), (t1, p1), (t2, p2) in zip(obs, obs[1:], obs[2:]):
                    dt1, dt2 = t1 - t0, t2 - t1
                    if dt1 <= 0 or dt2 <= 0 or max(dt1, dt2) > 1.0:
                        continue  # 欠けたフレームをまたぐ差は使わない
                    v1, v2 = (p1 - p0) / dt1, (p2 - p1) / dt2
                    if (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 < MIN_MOVING_SPEED:
                        continue
                    dv = v2 - v1
                    stats[label].append(float(dv @ dv) / ((dt1 + dt2) / 2.0))
    print(f"\n## プロセスノイズ q [m²/s³]（GT の {len(scenes)} シーン、"
          f"速さ {MIN_MOVING_SPEED} m/s 以上、median(|Δv|²/Δt) / 1.386）")
    result = {}
    for label, values in sorted(stats.items(), key=lambda kv: -len(kv[1])):
        q = float(np.median(values)) / CHI2_2_MEDIAN
        result[label] = round(q, 3)
        print(f"  {label:20s} n={len(values):5d} q={q:.3f}")
    print("\nTRACK_SMOOTHER_PROCESS_NOISE =", result)


def main() -> None:
    args = _parse_args()
    runs = _select_runs(args.runs)
    measurement_noise(runs, args.max_match)
    process_noise()


main()
