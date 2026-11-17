"""トラックの動き（静止 / カルマンスムーザー）に応じたボックスの補正.

ボックスの規定長と向きによる結合（box_merged_track_id）と、その単位の寸法補正の
後に行う。手順（詳細は webapp/inference/CLAUDE.md）:

1. box_merged_track ごとに、グローバル座標の XY でカルマンフィルタ＋RTS スムーザー
   による平滑化を行う（外れ値を反復して除く）。Min Frames to Smooth 未満の
   トラック（短いトラック）は平滑化しない
2. 中心のばらつきと平滑化した速さで静止を判定する
3. 静止トラック: 中央値の位置が近いトラック同士（短いトラックを含む）を結合し、
   XY を中央値、底面を分位点、角度を中央値（周囲の静止車両とも比べる）に揃える
4. 移動トラック: 端点の状態の予測で分割されたトラックを結合し（短いトラックを
   含む）、結合したトラックを平滑化し直して、位置と角度を修正する
5. 結合後のトラックに global_track_id を割り当てる

観測ノイズ R はボックスの軸（長さ・幅）ごとに、LiDAR 点の有無とセンサーからの
距離で決める。座標は XY だけを扱い、z は静止トラックの底面にだけ使う。
yaw のグローバル座標への変換は ego_pose の yaw だけを足し引きする（ロール・
ピッチは無視。トラック単位の角度補正と同じ扱い）。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from common.transform3d import invert_transform, make_transform

KIND_STATIC = "static"
KIND_MOVING = "moving"
KIND_SHORT = "short"

CHI2_2_99 = 9.21
CHI2_4_99 = 13.28

DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "min_frames": 3,
    # 平滑化
    "outlier_threshold": CHI2_2_99,
    "num_iters": 3,
    "initial_velocity_var": 100.0,
    # σ = a + b·d [m]（d はセンサーからの距離）。LiDAR 点を含むか（lidar）/ 深度のみ（depth）
    "measurement_noise": {
        "lidar": {"length": [0.3, 0.01], "width": [0.15, 0.005]},
        "depth": {"length": [0.8, 0.03], "width": [0.4, 0.015]},
    },
    # 等速度モデルの加速度の強さ q [m²/s³]。ラベルに無ければ default
    "process_noise": {"default": 1.0},
    # 静止判定
    "static_radius": 1.0,
    "static_spread_percentile": 90.0,
    "static_max_speed": 0.5,
    # 静止トラックの補正
    "static_link_distance": 1.0,
    "static_bottom_percentile": 25.0,
    "track_orient_diff_deg": 15.0,
    "neighbor_orient_diff_deg": 15.0,
    "search_length": 15.0,
    "search_width": 3.0,
    # 移動トラックの結合
    "max_link_gap": 3.0,
    "link_threshold_4d": CHI2_4_99,
    "link_threshold_2d": CHI2_2_99,
    # 移動トラックの修正
    "position_gate": CHI2_2_99,
    "min_position_correction": 0.5,
    "min_speed_heading": 2.0,
    "heading_diff_deg": 20.0,
}


def smoother_settings(box_fitting_params: dict[str, Any] | None) -> dict[str, Any]:
    """run のパラメータ（box_fitting_params["track_smoother"]）に既定値を補う."""
    params = (box_fitting_params or {}).get("track_smoother") or {}
    settings = {**DEFAULT_SETTINGS, **params}
    settings["measurement_noise"] = {
        **DEFAULT_SETTINGS["measurement_noise"], **(params.get("measurement_noise") or {}),
    }
    settings["process_noise"] = {
        **DEFAULT_SETTINGS["process_noise"], **(params.get("process_noise") or {}),
    }
    return settings


# ── 角度 ─────────────────────────────────────────────────────────────────────


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """(-π, π] に収める."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi if isinstance(
        angle, np.ndarray) else (angle + math.pi) % (2.0 * math.pi) - math.pi


def axial_diff(a: float, b: float) -> float:
    """前後を区別しない角度の差 [rad]（0〜π/2）."""
    d = abs(wrap_angle(a - b))
    return min(d, math.pi - d)


def circular_median(angles: np.ndarray) -> float:
    """向きの中央値（円周平均のまわりで差の中央値を取る）."""
    angles = np.asarray(angles, dtype=float)
    mean = math.atan2(float(np.sin(angles).sum()), float(np.cos(angles).sum()))
    return float(wrap_angle(mean + float(np.median(wrap_angle(angles - mean)))))


def axial_median(angles: np.ndarray) -> float:
    """前後を区別しない向きの中央値（角度を 2 倍して円周の中央値を取り、半分に戻す）."""
    return float(wrap_angle(circular_median(2.0 * np.asarray(angles, dtype=float)) / 2.0))


def yaw_of_pose(pose: dict[str, Any]) -> float:
    w, x, y, z = (float(v) for v in pose["rotation"])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ── カルマンフィルタ＋RTS スムーザー ────────────────────────────────────────

_H = np.hstack([np.eye(2), np.zeros((2, 2))])


def transition(dt: float) -> np.ndarray:
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    return f


def process_noise(dt: float, q: float) -> np.ndarray:
    dt = abs(dt)
    i = np.eye(2)
    return q * np.block([[dt ** 3 / 3.0 * i, dt ** 2 / 2.0 * i],
                         [dt ** 2 / 2.0 * i, dt * i]])


def measurement_cov(yaw: float, sigma_length: float, sigma_width: float) -> np.ndarray:
    """ボックスの軸（長さ・幅）ごとの観測ノイズをグローバル XY へ回したもの."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return rot @ np.diag([sigma_length ** 2, sigma_width ** 2]) @ rot.T


def kalman_rts(
    times: np.ndarray,
    zs: np.ndarray,
    rs: np.ndarray,
    included: np.ndarray,
    q: float,
    initial_velocity_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """前向きのカルマンフィルタと後ろ向きの RTS スムーザー.

    除外した観測（included=False）のフレームは予測だけで通過する。

    Returns:
        (平滑化した状態 (N, 4), 共分散 (N, 4, 4))
    """
    n = len(times)
    first = int(np.argmax(included)) if included.any() else 0
    x = np.array([zs[first, 0], zs[first, 1], 0.0, 0.0])
    p = np.zeros((4, 4))
    p[:2, :2] = rs[first]
    p[2:, 2:] = np.eye(2) * initial_velocity_var
    x_pred = np.zeros((n, 4))
    p_pred = np.zeros((n, 4, 4))
    x_filt = np.zeros((n, 4))
    p_filt = np.zeros((n, 4, 4))
    for k in range(n):
        if k > 0:
            dt = float(times[k] - times[k - 1])
            f = transition(dt)
            x = f @ x
            p = f @ p @ f.T + process_noise(dt, q)
        x_pred[k], p_pred[k] = x, p
        if included[k]:
            s = _H @ p @ _H.T + rs[k]
            gain = p @ _H.T @ np.linalg.inv(s)
            x = x + gain @ (zs[k] - _H @ x)
            p = (np.eye(4) - gain @ _H) @ p
            p = (p + p.T) / 2.0
        x_filt[k], p_filt[k] = x, p
    xs, ps = x_filt.copy(), p_filt.copy()
    for k in range(n - 2, -1, -1):
        f = transition(float(times[k + 1] - times[k]))
        c = p_filt[k] @ f.T @ np.linalg.pinv(p_pred[k + 1])
        xs[k] = x_filt[k] + c @ (xs[k + 1] - x_pred[k + 1])
        ps[k] = p_filt[k] + c @ (ps[k + 1] - p_pred[k + 1]) @ c.T
        ps[k] = (ps[k] + ps[k].T) / 2.0
    return xs, ps


def residual_d2(
    zs: np.ndarray, rs: np.ndarray, xs: np.ndarray, ps: np.ndarray, included: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """平滑化した位置との残差とマハラノビス距離の二乗.

    平滑化に使った観測は残差の共分散 S = R − H P H^T（平滑化がその観測に引き寄せ
    られる分だけ R より小さい）、除外した観測は S = H P H^T + R で測る。
    R − H P H^T が正定値でなくなった（数値誤差で潰れた）ときは R で代用する。
    """
    residuals = zs - xs[:, :2]
    d2 = np.zeros(len(zs))
    for k in range(len(zs)):
        hph = _H @ ps[k] @ _H.T
        if included[k]:
            s = rs[k] - hph
            if np.linalg.eigvalsh(s).min() <= 1e-6 * float(np.trace(rs[k])):
                s = rs[k]
        else:
            s = hph + rs[k]
        d2[k] = float(residuals[k] @ np.linalg.solve(s, residuals[k]))
    return residuals, d2


def smooth_with_outliers(
    times: np.ndarray, zs: np.ndarray, rs: np.ndarray, q: float, settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    """外れ値を除いては平滑化し直す（外れ値の集合が変わらなくなるまで、上限 num_iters 回）."""
    included = np.ones(len(times), dtype=bool)
    xs, ps = kalman_rts(times, zs, rs, included, q, settings["initial_velocity_var"])
    residuals, d2 = residual_d2(zs, rs, xs, ps, included)
    for _ in range(int(settings["num_iters"])):
        new_included = d2 <= float(settings["outlier_threshold"])
        # 平滑化には 2 観測以上残す
        if new_included.sum() < 2:
            new_included = np.zeros(len(times), dtype=bool)
            new_included[np.argsort(d2)[:2]] = True
        if np.array_equal(new_included, included):
            break
        included = new_included
        xs, ps = kalman_rts(times, zs, rs, included, q, settings["initial_velocity_var"])
        residuals, d2 = residual_d2(zs, rs, xs, ps, included)
    return {"xs": xs, "ps": ps, "included": included, "residuals": residuals, "d2": d2}


# ── 観測とトラック ───────────────────────────────────────────────────────────


class _Obs:
    """1 フレームのボックス（代表行）をグローバル座標で扱うための入れ物."""

    def __init__(self, row: dict[str, Any], origin: tuple[float, float],
                 pose: dict[str, Any], has_lidar: bool) -> None:
        self.row = row
        self.pose = pose
        self.time = float(pose.get("timestamp") or 0) * 1e-6
        self.ego_to_global = make_transform(pose["rotation"], pose["translation"])
        self.pose_yaw = yaw_of_pose(pose)
        center = np.asarray(row["center_ego"], dtype=float)
        self.center_global = (self.ego_to_global @ np.append(center, 1.0))[:3]
        self.yaw_global = float(wrap_angle(float(row.get("yaw_ego") or 0.0) + self.pose_yaw))
        self.size = [float(v) for v in row["size_wlh"]]
        self.distance = float(np.hypot(center[0] - origin[0], center[1] - origin[1]))
        self.has_lidar = has_lidar
        self.before = {"center_ego": [float(v) for v in center],
                       "yaw_ego": float(row.get("yaw_ego") or 0.0)}

    def noise(self, settings: dict[str, Any]) -> np.ndarray:
        coeff = settings["measurement_noise"]["lidar" if self.has_lidar else "depth"]
        sigma_l = float(coeff["length"][0]) + float(coeff["length"][1]) * self.distance
        sigma_w = float(coeff["width"][0]) + float(coeff["width"][1]) * self.distance
        return measurement_cov(self.yaw_global, sigma_l, sigma_w)

    def set_global(self, center_global: np.ndarray | None = None,
                   yaw_global: float | None = None) -> None:
        """グローバル座標の中心・yaw をこのフレームの基準 ego 座標に戻して行へ書く."""
        if center_global is not None:
            center = (invert_transform(self.ego_to_global) @ np.append(center_global, 1.0))[:3]
            self.row["center_ego"] = [round(float(v), 3) for v in center]
            self.center_global = np.asarray(center_global, dtype=float)
        if yaw_global is not None:
            self.row["yaw_ego"] = round(float(wrap_angle(yaw_global - self.pose_yaw)), 5)
            self.yaw_global = float(wrap_angle(yaw_global))
        metrics = self.row.get("fit_metrics") or {}
        if "corners_xy" in metrics and (center_global is not None or yaw_global is not None):
            from app.services.box_size import _corners

            width, length = float(self.row["size_wlh"][0]), float(self.row["size_wlh"][1])
            metrics["corners_xy"] = _corners(
                np.asarray(self.row["center_ego"][:2], dtype=float), length, width,
                float(self.row["yaw_ego"]))


class _Track:
    def __init__(self, track_id: str, label: str | None, observations: list[_Obs]) -> None:
        self.id = track_id
        self.label = label
        self.obs = sorted(observations, key=lambda o: o.time)
        self.samples = {o.row.get("_sample_token") for o in self.obs}
        self.kind = KIND_SHORT
        self.smoothed: dict[str, np.ndarray] | None = None

    @property
    def start(self) -> float:
        return self.obs[0].time

    @property
    def end(self) -> float:
        return self.obs[-1].time

    def median_xy(self) -> np.ndarray:
        return np.median(np.stack([o.center_global[:2] for o in self.obs]), axis=0)


def _arrays(observations: list[_Obs], settings: dict[str, Any]):
    times = np.array([o.time for o in observations])
    zs = np.stack([o.center_global[:2] for o in observations])
    rs = np.stack([o.noise(settings) for o in observations])
    return times, zs, rs


def _q(label: str | None, settings: dict[str, Any]) -> float:
    noise = settings["process_noise"]
    return float(noise.get(label or "", noise.get("default", 1.0)))


def _smooth(observations: list[_Obs], label: str | None,
            settings: dict[str, Any]) -> dict[str, np.ndarray]:
    times, zs, rs = _arrays(observations, settings)
    result = smooth_with_outliers(times, zs, rs, _q(label, settings), settings)
    result.update(times=times, zs=zs, rs=rs)
    return result


# ── 静止トラック ─────────────────────────────────────────────────────────────


def _is_static(track: _Track, settings: dict[str, Any]) -> tuple[bool, float, float]:
    sm = track.smoothed
    zs = sm["zs"][sm["included"]]
    spread = float(np.percentile(np.linalg.norm(zs - np.median(zs, axis=0), axis=1),
                                 float(settings["static_spread_percentile"])))
    speed = float(np.median(np.linalg.norm(sm["xs"][:, 2:], axis=1)))
    return (spread < float(settings["static_radius"])
            and speed < float(settings["static_max_speed"])), spread, speed


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item


def link_static_tracks(
    static: list[_Track], short: list[_Track], settings: dict[str, Any],
) -> tuple[list[list[_Track]], list[dict[str, Any]]]:
    """中央値の位置が近い静止トラック同士（と短いトラック）を結合する.

    距離の近い対から順に結合し、結合後のグループ内に同じフレームに出現する
    トラックが入る結合は却下する。短いトラック同士は結合しない。

    Returns:
        (グループのリスト（結合しなかった静止トラックも 1 要素のグループ）, 結合した対)
    """
    tracks = static + short
    medians = {t.id: t.median_xy() for t in tracks}
    static_ids = {t.id for t in static}
    by_id = {t.id: t for t in tracks}
    pairs = []
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            if a.id not in static_ids and b.id not in static_ids:
                continue
            if a.label != b.label or a.samples & b.samples:
                continue
            dist = float(np.linalg.norm(medians[a.id] - medians[b.id]))
            if dist <= float(settings["static_link_distance"]):
                pairs.append((dist, a.id, b.id))
    pairs.sort()
    uf = _UnionFind([t.id for t in tracks])
    samples = {t.id: set(t.samples) for t in tracks}
    links = []
    for dist, a, b in pairs:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb or samples[ra] & samples[rb]:
            continue
        uf.parent[rb] = ra
        samples[ra] |= samples.pop(rb)
        links.append({"a": a, "b": b, "distance": round(dist, 3)})
    groups: dict[str, list[_Track]] = {}
    for t in tracks:
        groups.setdefault(uf.find(t.id), []).append(t)
    # 静止トラックを含まないグループ（結合されなかった短いトラック）は返さない
    result = [g for g in groups.values() if any(t.id in static_ids for t in g)]
    for group in result:
        for t in group:
            by_id[t.id].kind = KIND_STATIC
    return result, links


def correct_static_group(
    group: list[_Track], neighbors: list[tuple[np.ndarray, float]], settings: dict[str, Any],
) -> dict[str, Any]:
    """静止グループの XY・底面・角度を揃える（行をその場で書き換える）.

    neighbors は他の静止グループのボックスの (グローバル XY, yaw)（補正前の値）。
    """
    observations = [o for t in group for o in t.obs]
    median_xy = np.median(np.stack([o.center_global[:2] for o in observations]), axis=0)
    bottoms = [o.center_global[2] - o.size[2] / 2.0 for o in observations]
    bottom = float(np.percentile(bottoms, float(settings["static_bottom_percentile"])))
    track_yaw = circular_median(np.array([o.yaw_global for o in observations]))

    # 周囲の静止車両（自分のグループ以外）の向き。中央値の位置からボックスの軸で範囲を取る
    u = np.array([math.cos(track_yaw), math.sin(track_yaw)])
    v = np.array([-u[1], u[0]])
    near = [yaw for xy, yaw in neighbors
            if abs(float((xy - median_xy) @ u)) <= float(settings["search_length"])
            and abs(float((xy - median_xy) @ v)) <= float(settings["search_width"])]
    neighbor_yaw = axial_median(np.array(near)) if near else None

    track_thr = math.radians(float(settings["track_orient_diff_deg"]))
    neighbor_thr = math.radians(float(settings["neighbor_orient_diff_deg"]))
    num_yaw = 0
    for o in observations:
        yaw_corrected = (
            abs(wrap_angle(o.yaw_global - track_yaw)) >= track_thr
            and (neighbor_yaw is None or axial_diff(o.yaw_global, neighbor_yaw) >= neighbor_thr)
        )
        center = np.array([median_xy[0], median_xy[1], bottom + o.size[2] / 2.0])
        o.set_global(center, track_yaw if yaw_corrected else None)
        num_yaw += int(yaw_corrected)
        o.row["_motion"] = {"position_corrected": True, "yaw_corrected": yaw_corrected}
    return {
        "median_xy": [round(float(v), 3) for v in median_xy],
        "bottom_z": round(bottom, 3),
        "median_yaw": round(track_yaw, 5),
        "neighbor_yaw": None if neighbor_yaw is None else round(neighbor_yaw, 5),
        "num_neighbors": len(near),
        "num_yaw_corrected": num_yaw,
    }


# ── 移動トラックの結合 ───────────────────────────────────────────────────────


def _predict(x: np.ndarray, p: np.ndarray, dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    f = transition(dt)
    return f @ x, f @ p @ f.T + process_noise(dt, q)


def link_cost(a: _Track, b: _Track, settings: dict[str, Any]) -> dict[str, Any] | None:
    """先に終わる a と後から始まる b を結合する候補なら、D² と自由度を返す."""
    if a.label != b.label or a.samples & b.samples or a.end >= b.start:
        return None
    gap = b.start - a.end
    if gap > float(settings["max_link_gap"]):
        return None
    q = _q(a.label, settings)
    a_long, b_long = a.smoothed is not None, b.smoothed is not None
    if a_long and b_long:
        xa, pa = _predict(a.smoothed["xs"][-1], a.smoothed["ps"][-1], gap, q)
        e = xa - b.smoothed["xs"][0]
        s = pa + b.smoothed["ps"][0]
        dof, thr = 4, float(settings["link_threshold_4d"])
    elif a_long:
        xa, pa = _predict(a.smoothed["xs"][-1], a.smoothed["ps"][-1], gap, q)
        e = xa[:2] - b.obs[0].center_global[:2]
        s = _H @ pa @ _H.T + b.obs[0].noise(settings)
        dof, thr = 2, float(settings["link_threshold_2d"])
    elif b_long:
        xb, pb = _predict(b.smoothed["xs"][0], b.smoothed["ps"][0], -gap, q)
        e = xb[:2] - a.obs[-1].center_global[:2]
        s = _H @ pb @ _H.T + a.obs[-1].noise(settings)
        dof, thr = 2, float(settings["link_threshold_2d"])
    else:
        return None
    d2 = float(e @ np.linalg.solve(s, e))
    if d2 > thr:
        return None
    return {"d2": d2, "dof": dof, "threshold": thr, "gap": gap}


def link_moving_tracks(
    tracks: list[_Track], settings: dict[str, Any],
) -> tuple[list[list[_Track]], list[dict[str, Any]]]:
    """分割された移動トラックを 1 対 1 に結合し、鎖ごとのリストを返す."""
    from scipy.optimize import linear_sum_assignment

    if not tracks:
        return [], []
    n = len(tracks)
    big = 1e6
    cost = np.full((n, n), big)
    info: dict[tuple[int, int], dict[str, Any]] = {}
    for i, a in enumerate(tracks):
        for j, b in enumerate(tracks):
            if i == j:
                continue
            result = link_cost(a, b, settings)
            if result is not None:
                # 自由度の異なる D² を同じ尺度で比べるため、閾値で割る
                cost[i, j] = result["d2"] / result["threshold"]
                info[(i, j)] = result
    rows, cols = linear_sum_assignment(cost)
    nxt: dict[int, int] = {}
    links = []
    for i, j in zip(rows, cols, strict=True):
        if (i, j) in info:
            nxt[i] = j
            r = info[(i, j)]
            links.append({"a": tracks[i].id, "b": tracks[j].id, "d2": round(r["d2"], 3),
                          "dof": r["dof"], "threshold": r["threshold"],
                          "gap_sec": round(r["gap"], 3)})
    has_prev = set(nxt.values())
    chains = []
    for i in range(n):
        if i in has_prev:
            continue
        chain = [tracks[i]]
        while i in nxt:
            i = nxt[i]
            chain.append(tracks[i])
        chains.append(chain)
    return chains, links


def correct_moving(observations: list[_Obs], label: str | None,
                   settings: dict[str, Any]) -> dict[str, np.ndarray]:
    """平滑化し直して、位置と角度を修正する（行をその場で書き換える）."""
    observations = sorted(observations, key=lambda o: o.time)
    sm = _smooth(observations, label, settings)
    gate = float(settings["position_gate"])
    min_shift = float(settings["min_position_correction"])
    min_speed = float(settings["min_speed_heading"])
    heading_thr = math.radians(float(settings["heading_diff_deg"]))
    for k, o in enumerate(observations):
        position = bool(sm["d2"][k] > gate and np.linalg.norm(sm["residuals"][k]) > min_shift)
        velocity = sm["xs"][k, 2:]
        speed = float(np.linalg.norm(velocity))
        heading = math.atan2(float(velocity[1]), float(velocity[0]))
        yaw = bool(speed >= min_speed and abs(wrap_angle(o.yaw_global - heading)) >= heading_thr)
        center = None
        if position:
            center = np.array([sm["xs"][k, 0], sm["xs"][k, 1], o.center_global[2]])
        o.set_global(center, heading if yaw else None)
        o.row["_motion"] = {"position_corrected": position, "yaw_corrected": yaw}
    return sm


# ── 全体 ─────────────────────────────────────────────────────────────────────


def _record(o: _Obs, track: _Track, global_id: str, sm: dict[str, np.ndarray] | None,
            k: int | None) -> None:
    motion = o.row.pop("_motion", None) or {"position_corrected": False, "yaw_corrected": False}
    record: dict[str, Any] = {
        "kind": track.kind,
        "box_merged_track_id": track.id,
        "global_track_id": global_id,
        "observed_xy": [round(float(v), 3) for v in o.before_global_xy],
        "before": o.before,
        **motion,
    }
    if sm is not None and k is not None:
        record.update(
            smoothed_xy=[round(float(v), 3) for v in sm["xs"][k, :2]],
            smoothed_velocity=[round(float(v), 3) for v in sm["xs"][k, 2:]],
            speed=round(float(np.linalg.norm(sm["xs"][k, 2:])), 3),
            d2=round(float(sm["d2"][k]), 3),
            outlier=not bool(sm["included"][k]),
        )
    o.row.setdefault("fit_metrics", {})["motion_correction"] = record


def correct_track_motion(
    fitted: list[tuple[str, str | None, dict[str, Any], tuple[float, float], str]],
    reference_ego_poses: dict[str, dict[str, Any]] | None,
    box_fitting_params: dict[str, Any] | None,
    has_lidar: dict[tuple[str, str], bool] | None = None,
) -> dict[str, Any]:
    """動きに応じた補正を行い、box_merged_track_id → global_track_id を返す.

    Args:
        fitted: ``(box_merged_track_id, label, 代表行, センサー位置, sample_token)``。
            行は**その場で書き換える**
        reference_ego_poses: sample_token → 基準 ego_pose（timestamp を含む）
        has_lidar: (box_merged_track_id, sample_token) → 当てはめた点群に LiDAR 点を含むか

    Returns:
        ``{"global_ids": {box_merged_track_id: global_track_id}, "diagnostics": {...}}``
    """
    settings = smoother_settings(box_fitting_params)
    poses = reference_ego_poses or {}
    by_track: dict[str, list[_Obs]] = {}
    labels: dict[str, str | None] = {}
    for track_id, label, row, origin, sample_token in fitted:
        pose = poses.get(sample_token)
        if not pose or row.get("center_ego") is None:
            continue
        row["_sample_token"] = sample_token
        obs = _Obs(row, origin, pose, bool((has_lidar or {}).get((track_id, sample_token), False)))
        obs.before_global_xy = obs.center_global[:2].copy()
        by_track.setdefault(track_id, []).append(obs)
        labels.setdefault(track_id, label)
    tracks = [_Track(tid, labels[tid], obs) for tid, obs in by_track.items()]

    # 1〜2) 長いトラックを平滑化し、静止を判定する
    track_info: dict[str, dict[str, Any]] = {}
    for t in tracks:
        if len(t.obs) < int(settings["min_frames"]):
            continue
        t.smoothed = _smooth(t.obs, t.label, settings)
        static, spread, speed = _is_static(t, settings)
        t.kind = KIND_STATIC if static else KIND_MOVING
        track_info[t.id] = {"spread": round(spread, 3), "median_speed": round(speed, 3)}

    static = [t for t in tracks if t.kind == KIND_STATIC]
    short = [t for t in tracks if t.kind == KIND_SHORT]

    # 3) 静止トラックの結合と補正（短いトラックはまずここで結合を試す）
    groups, static_links = link_static_tracks(static, short, settings)
    global_ids: dict[str, str] = {}
    group_info: dict[str, dict[str, Any]] = {}
    # 周囲の静止車両の向きは補正前の値で比べる（補正の順番で結果が変わらないように）
    static_boxes = [(index, o.center_global[:2].copy(), o.yaw_global)
                    for index, g in enumerate(groups) for t in g for o in t.obs]
    for index, group in enumerate(groups):
        group.sort(key=lambda t: t.start)
        gid = group[0].id
        neighbors = [(xy, yaw) for i, xy, yaw in static_boxes if i != index]
        group_info[gid] = correct_static_group(group, neighbors, settings)
        for t in group:
            global_ids[t.id] = gid

    # 4) 移動トラック（と静止の結合に入らなかった短いトラック）の結合と補正
    candidates = [t for t in tracks if t.kind == KIND_MOVING
                  or (t.kind == KIND_SHORT and t.id not in global_ids)]
    chains, moving_links = link_moving_tracks(candidates, settings)
    chain_smoothed: dict[str, tuple[list[_Obs], dict[str, np.ndarray]]] = {}
    for chain in chains:
        gid = chain[0].id
        for t in chain:
            global_ids[t.id] = gid
        if len(chain) == 1 and chain[0].kind == KIND_SHORT:
            continue  # 結合されなかった短いトラックは補正しない
        for t in chain:
            t.kind = KIND_MOVING
        observations = sorted((o for t in chain for o in t.obs), key=lambda o: o.time)
        sm = correct_moving(observations, chain[0].label, settings)
        chain_smoothed[gid] = (observations, sm)

    # 記録（静止は最初の平滑化、移動は結合後の平滑化の結果を残す）
    for t in tracks:
        gid = global_ids.setdefault(t.id, t.id)
        if gid in chain_smoothed:
            observations, sm = chain_smoothed[gid]
            index = {id(o): k for k, o in enumerate(observations)}
            for o in t.obs:
                _record(o, t, gid, sm, index[id(o)])
        else:
            for k, o in enumerate(t.obs):
                _record(o, t, gid, t.smoothed, k if t.smoothed is not None else None)
        for o in t.obs:
            o.row.pop("_sample_token", None)

    diagnostics = {
        "params": {k: v for k, v in settings.items()
                   if k not in ("measurement_noise", "process_noise")},
        "tracks": [
            {
                "box_merged_track_id": t.id,
                "global_track_id": global_ids[t.id],
                "label": t.label,
                "kind": t.kind,
                "num_frames": len(t.obs),
                "num_outliers": (int((~t.smoothed["included"]).sum())
                                 if t.smoothed is not None else 0),
                "median_xy": [round(float(v), 3) for v in
                              np.median(np.stack([o.before_global_xy for o in t.obs]), axis=0)],
                **track_info.get(t.id, {}),
            }
            for t in sorted(tracks, key=lambda t: t.id)
        ],
        "static_groups": [{"global_track_id": gid, **info} for gid, info in group_info.items()],
        "static_links": static_links,
        "moving_links": moving_links,
        "num_static": sum(1 for t in tracks if t.kind == KIND_STATIC),
        "num_moving": sum(1 for t in tracks if t.kind == KIND_MOVING),
        "num_short": sum(1 for t in tracks if t.kind == KIND_SHORT),
    }
    return {"global_ids": global_ids, "diagnostics": diagnostics}
