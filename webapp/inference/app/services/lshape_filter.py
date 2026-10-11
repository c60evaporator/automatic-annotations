"""車両の手前 2 面（L 字）を基準にした、手前側ノイズの除去.

## なぜ必要か

Convex-hull MOA は「センサーから見える手前の輪郭」で向きと大きさを決める。
そのため、手前の面より前へ突き出たノイズに弱い。ROR / DBSCAN で落ちない
程度の小さな突出でも、凸包の手前側の頂点になるので箱が大きく引きずられる。

## 流れ（BEV で行う）

1. **手前輪郭**: センサー原点から見た方位角を細かいビンに分け、各ビンで
   距離の低い分位点を取る。面状に広がった点群（ボンネットの傾斜など）から、
   センサーに見えている輪郭だけを 1 次元の折れ線として取り出す
2. **L 字当てはめ**: 向き θ を 0〜90° で総当たりする。各軸について、
   面の位置を「帯 ``support_band`` 以内に乗る輪郭点の多さ − 面より手前に
   残る輪郭点の数」が最大になる位置に置き、L 字全体でこの評価値が最大の
   θ を選ぶ。最小二乗の線（点群の中心）ではなく外側の線が得られる。
   **面の位置は固定の分位点では決めない**（手前の塊が輪郭の数 % を超えると
   分位点が塊に乗るため）。分位点は「その向きの面が見えているか」の判定に
   だけ使う
3. **角度の微調整**: 単眼深度では 2 面が直交からずれることがあるので、
   もう一方の線を固定したまま、各線の角度を ±``adjust_range_deg`` の
   範囲で個別に調整する
4. **除去**: 各点について、同じ方位で L 字の面に入るまでの距離（基準距離）と
   比べ、**手前側に** ``max(margin_min, margin_ratio × 基準距離)`` より
   大きく出ている点だけを落とす。あわせて、面から垂直方向にも
   ``support_band`` 以上出ていることを求める（浅い角度で見えている面の
   点を誤って消さないため）

輪郭点は、そのビンで輪郭の近くにある点の数で重み付けする。手前の塊が
数ビンにまたがると、その後ろの面は輪郭から隠れるので、ビンを等しく
数えると塊に沿った誤った L 字が選ばれる。

## 判定の要点

- **片側だけ**: 面より奥の点（ボンネット・屋根・車体側面の奥）は消さない
- **視線方向**: 単眼深度の誤差は視線方向に出るので、直線への垂線距離では
  なく、同じ方位で何 m 手前にあるかで測る
- **距離に応じた閾値**: 深度の誤差は距離とともに大きくなる
  （LiDAR は ``margin_ratio=0`` で固定値にする）

NOTE: センサー原点は点群ごとに違う（深度はそのカメラ、LiDAR は LIDAR_TOP）。
カメラが違えば手前の面も変わるので、**カメラ間結合の前**に適用すること。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# 手前輪郭がこの数のビンに満たない場合は、L 字を決められないので何もしない
MIN_CONTOUR_BINS = 3
# 担当する輪郭点がこの数に満たない線は、角度の微調整をしない
MIN_POINTS_FOR_ADJUST = 3


def _wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def near_contour(
    points_xy: np.ndarray,
    origin_xy: np.ndarray,
    *,
    bin_deg: float,
    quantile: float,
    band: float,
) -> tuple[np.ndarray, np.ndarray]:
    """センサーに見えている手前の輪郭を、方位ビンごとの点として返す.

    Args:
        bin_deg: 方位ビンの幅 [度]
        quantile: 各ビンで取る距離の分位点 [%]。最小値（0）だと孤立した
            1 点に輪郭が引っ張られるので、少し内側を取る
        band: 輪郭点の重みを数える範囲 [m]

    Returns:
        (輪郭点 ``(M, 2)``, 重み ``(M,)``)。方位の順に並ぶ。
        重みは「そのビンで輪郭点から ``band`` 以内の距離にある点の数」。
        **ビンを等しく数えると、数ビンにまたがる手前の塊が面と同じ重さに
        なり、塊に沿った誤った L 字が選ばれる**（塊の後ろの面は輪郭から
        隠れるため）。面のビンは点が多く、塊のビンは少ないので、点の数で
        重み付けすると面が優先される
    """
    rel = points_xy - origin_xy
    ranges = np.hypot(rel[:, 0], rel[:, 1])
    azimuth = np.arctan2(rel[:, 1], rel[:, 0])
    # ±π をまたぐ物体（真後ろ）でビンが分断されないよう、平均方位を基準にする
    reference = np.arctan2(np.mean(np.sin(azimuth)), np.mean(np.cos(azimuth)))
    delta = _wrap_pi(azimuth - reference)

    bins = np.floor(delta / np.deg2rad(bin_deg)).astype(np.int64)
    order = np.argsort(bins, kind="stable")
    bins, ranges, delta = bins[order], ranges[order], delta[order]
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    ends = np.r_[starts[1:], len(bins)]

    contour = np.empty((len(starts), 2))
    weights = np.empty(len(starts))
    for i, (s, e) in enumerate(zip(starts, ends)):
        r = float(np.percentile(ranges[s:e], quantile))
        # ビン中心ではなく、ビン内の点の方位の中央値を使う（量子化を避ける）
        a = reference + float(np.median(delta[s:e]))
        contour[i] = origin_xy + r * np.array([np.cos(a), np.sin(a)])
        weights[i] = float(np.sum(np.abs(ranges[s:e] - r) <= band))
    return contour, weights


# 境界線より手前に残る点 1 つあたりの減点の既定値（面に乗る点 1 つに対する比）。
# 大きいと「塊に沿った（手前に何も残らない）L 字」が勝ちやすくノイズを
# 取り逃がし、小さいと L 字が物体の内側へ入り込んで真の点を消しやすい。
# 合成データ（深度風・LiDAR 風 84 ケース）で 0〜1.0 を比べた結果:
#   0.0: ノイズ除去 93% / 誤除去 1.6%、0.25: 87% / 0.06%、0.5: 81% / 0.04%
DEFAULT_VIOLATION_WEIGHT = 0.25


def _best_offset(
    proj: np.ndarray, weights: np.ndarray, band: float,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
) -> tuple[float, float]:
    """面の位置（オフセット）を決める.

    物体は ``proj >= t`` の側にある。**固定の分位点では決めない。**
    手前に突出した塊が輪郭の数 % を超えると、分位点がその塊に乗って
    しまうため。代わりに、候補ごとに

        （面に乗る点の多さ）−（面より ``band`` 以上手前に残る点の数）

    を最大にする位置を選ぶ（いずれも輪郭点の重みで数える）。面の点は
    1 つのオフセットに集まるので強い山になる。面より奥の候補は、面の点を
    手前に残すので減点される（L 字が物体の内側へ入り込むのを防ぐ）。
    """
    candidates = np.unique(proj)
    support = (np.clip(
        1.0 - np.abs(proj[None, :] - candidates[:, None]) / band, 0.0, None
    ) * weights[None, :]).sum(axis=1)
    in_front = ((proj[None, :] < candidates[:, None] - band)
                * weights[None, :]).sum(axis=1)
    score = support - violation_weight * in_front
    best = int(np.argmax(score))
    return float(candidates[best]), float(score[best])


def _visible_faces(
    contour: np.ndarray, weights: np.ndarray, origin: np.ndarray,
    axes: list[np.ndarray], boundary_quantile: float, band: float,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
) -> list[tuple[np.ndarray, float]]:
    """各軸について、センサー側に見えている面の線を返す.

    Returns:
        ``(内向き法線 u, オフセット t)`` のリスト。物体は ``u · x >= t`` 側。
        センサーがその軸方向で物体の範囲内にある場合（真正面・真横から
        見ていて、その向きの面が見えない I 字）は含めない。
        見えているかの判定にだけ分位点を使う
    """
    lines = []
    for axis in axes:
        proj = contour @ axis
        o = float(origin @ axis)
        # センサーに近い側を内向き法線の始点にする
        flip = o >= float(np.median(proj))
        u = -axis if flip else axis
        proj_u = -proj if flip else proj
        if float(origin @ u) >= float(np.percentile(proj_u, boundary_quantile)):
            continue
        offset, _ = _best_offset(proj_u, weights, band, violation_weight)
        lines.append((u, offset))
    return lines


def _evaluate(
    contour: np.ndarray, weights: np.ndarray, lines, band: float,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
) -> tuple[float, np.ndarray]:
    """L 字全体の評価値（大きいほど良い）と、各輪郭点を担当する線の番号.

    評価値 = 線に乗る点の多さ − L 字の手前に ``band`` 以上残る点の数
    （いずれも輪郭点の重みで数える）
    """
    if not lines:
        return -np.inf, np.zeros(len(contour), dtype=np.int64)
    signed = np.stack([contour @ u - t for u, t in lines], axis=1)
    nearest = np.argmin(np.abs(signed), axis=1)
    d = np.abs(signed[np.arange(len(contour)), nearest])
    support = float(np.sum(np.clip(1.0 - d / band, 0.0, None) * weights))
    in_front = float(np.sum((signed.min(axis=1) < -band) * weights))
    return support - violation_weight * in_front, nearest


def fit_lshape(
    contour: np.ndarray,
    weights: np.ndarray,
    origin: np.ndarray,
    *,
    angle_step_deg: float,
    adjust_range_deg: float,
    adjust_step_deg: float,
    boundary_quantile: float,
    support_band: float,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
) -> tuple[list[tuple[np.ndarray, float]], float, float]:
    """手前輪郭に L 字（見えている 1〜2 面）を当てはめる.

    Returns:
        (線のリスト, 直交で当てはめたときの向き θ [度], 評価値)
    """
    best_lines: list[tuple[np.ndarray, float]] = []
    best_score, best_theta = -np.inf, 0.0
    for theta in np.arange(0.0, 90.0, angle_step_deg):
        rad = np.deg2rad(theta)
        axes = [np.array([np.cos(rad), np.sin(rad)]),
                np.array([-np.sin(rad), np.cos(rad)])]
        lines = _visible_faces(
            contour, weights, origin, axes, boundary_quantile, support_band,
            violation_weight,
        )
        score, _ = _evaluate(contour, weights, lines, support_band,
                             violation_weight)
        if score > best_score:
            best_lines, best_score, best_theta = lines, score, float(theta)

    if adjust_range_deg <= 0 or not best_lines:
        return best_lines, best_theta, best_score

    # --- 段階 1: 各線の角度を個別に微調整する ---------------------------
    # 単眼深度では 2 面が直交からずれることがある。線ごとに、もう一方の
    # 線を固定したまま角度を振り、L 字全体の評価値が最大になる角度を選ぶ
    offsets = np.deg2rad(np.arange(
        -adjust_range_deg, adjust_range_deg + 1e-9, adjust_step_deg
    ))
    lines = list(best_lines)
    for index in range(len(lines)):
        _, nearest = _evaluate(contour, weights, lines, support_band,
                               violation_weight)
        if int(np.sum(nearest == index)) < MIN_POINTS_FOR_ADJUST:
            continue
        u0 = lines[index][0]
        base = np.arctan2(u0[1], u0[0])
        best_line, best_line_score = lines[index], -np.inf
        for offset in offsets:
            v = np.array([np.cos(base + offset), np.sin(base + offset)])
            t_new, _ = _best_offset(contour @ v, weights, support_band,
                                    violation_weight)
            if float(origin @ v) >= t_new:
                # センサーが面の内側に入る向きは成り立たない
                continue
            trial = lines[:index] + [(v, t_new)] + lines[index + 1:]
            score, _ = _evaluate(contour, weights, trial, support_band,
                                 violation_weight)
            if score > best_line_score:
                best_line, best_line_score = (v, t_new), score
        lines[index] = best_line
    score, _ = _evaluate(contour, weights, lines, support_band,
                         violation_weight)
    return lines, best_theta, score


def front_deficit(
    points_xy: np.ndarray, origin: np.ndarray, lines
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各点が L 字の面よりどれだけ手前にあるか.

    物体は各線の内側（``u · x >= t``）の共通部分にあるので、視線が物体に
    入るのは全ての線を越えた地点、つまり各線との交点距離の最大値になる
    （基準距離）。視線が線と交わらない（平行・逆向き）線は使わない。

    Returns:
        (基準距離, 視線方向の不足量 = 基準距離 − 点の距離,
         基準距離を決めた線に対する垂直方向の不足量)。
        どの線とも交わらない方位は ``nan``（判定しない）
    """
    rel = points_xy - origin
    ranges = np.hypot(rel[:, 0], rel[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = rel / ranges[:, None]
    ref = np.full(len(points_xy), np.nan)
    perp = np.full(len(points_xy), np.nan)
    for u, t in lines:
        denom = direction @ u
        gap = t - float(origin @ u)          # センサーは外側なので正
        with np.errstate(invalid="ignore", divide="ignore"):
            hit = np.where(denom > 1e-9, gap / denom, np.nan)
        # この線が基準距離を決める（交点が最も遠い）点を更新する
        better = np.isnan(ref) | (hit > ref)
        better &= ~np.isnan(hit)
        ref = np.where(better, hit, ref)
        perp = np.where(better, t - points_xy @ u, perp)
    return ref, ref - ranges, perp


def lshape_filter(
    points: np.ndarray,
    origin_xy: tuple[float, float] | np.ndarray,
    *,
    bin_deg: float,
    quantile: float,
    margin_min: float,
    margin_ratio: float,
    angle_step_deg: float,
    adjust_range_deg: float,
    adjust_step_deg: float,
    boundary_quantile: float,
    support_band: float,
    violation_weight: float = DEFAULT_VIOLATION_WEIGHT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """L 字を基準に、手前へ突き出た点を落とす.

    Args:
        points: ``(N, 3)``（ego 座標。BEV の XY だけを使う）
        origin_xy: 点群と同じ座標系でのセンサー位置
        margin_min / margin_ratio: 除去の閾値
            ``max(margin_min, margin_ratio × 基準距離)`` [m]

    Returns:
        (残す点の bool マスク, 記録用の情報)。L 字を決められない場合は
        全点を残し、``applied=False`` を返す
    """
    points = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    info: dict[str, Any] = {"applied": False, "num_removed": 0}
    if len(points) < MIN_CONTOUR_BINS:
        info["reason"] = "too_few_points"
        return keep, info

    origin = np.asarray(origin_xy, dtype=np.float64)[:2]
    xy = points[:, :2]
    contour, weights = near_contour(
        xy, origin, bin_deg=bin_deg, quantile=quantile, band=support_band
    )
    if len(contour) < MIN_CONTOUR_BINS:
        info["reason"] = "too_few_bins"
        return keep, info

    lines, theta, score = fit_lshape(
        contour, weights, origin,
        angle_step_deg=angle_step_deg,
        adjust_range_deg=adjust_range_deg,
        adjust_step_deg=adjust_step_deg,
        boundary_quantile=boundary_quantile,
        support_band=support_band,
        violation_weight=violation_weight,
    )
    if not lines:
        info["reason"] = "no_visible_face"
        return keep, info

    ref, deficit, perp = front_deficit(xy, origin, lines)
    margin = np.maximum(margin_min, margin_ratio * ref)
    with np.errstate(invalid="ignore"):
        # 手前側（基準より近い）にだけ閾値を当てる。奥の点は消さない。
        # 視線方向の不足量に加えて、面から垂直方向にも support_band 以上
        # 出ていることを求める。**面を浅い角度で見ていると、面上の点でも
        # 視線方向の不足量がいくらでも大きくなる**（交点が視線に沿って
        # 遠くへ逃げる）ため、垂直方向の条件が無いと面そのものを消してしまう
        front = (deficit > margin) & (perp > support_band)
    keep = ~np.nan_to_num(front, nan=False).astype(bool)

    info.update({
        "applied": True,
        "num_removed": int((~keep).sum()),
        "num_contour": int(len(contour)),
        "theta_deg": round(theta, 2),
        "score": round(score / len(contour), 3),
        # BEV に重ねて確認できるよう、線も残す（u·x = t、物体は u·x >= t 側）
        "lines": [
            {"normal": [round(float(u[0]), 4), round(float(u[1]), 4)],
             "offset": round(float(t), 3)}
            for u, t in lines
        ],
    })
    return keep, info


def sensor_origin_xy(
    calibrated_sensor: dict[str, Any],
    sensor_ego_pose: dict[str, Any] | None,
    reference_ego_pose: dict[str, Any] | None,
) -> tuple[float, float]:
    """センサー位置を、点群と同じ（基準 ego）座標系での XY で返す."""
    from common.transform3d import ego_to_ego

    origin = np.asarray([[*calibrated_sensor["translation"][:3]]], dtype=np.float64)
    if reference_ego_pose and sensor_ego_pose:
        origin = ego_to_ego(origin, sensor_ego_pose, reference_ego_pose)
    return (float(origin[0][0]), float(origin[0][1]))


def lshape_settings(
    general: dict[str, Any] | None,
    modality: dict[str, Any] | None,
    *,
    margin_ratio_default: float = 0.0,
) -> dict[str, Any] | None:
    """run のパラメータから lshape_filter() の引数を組み立てる.

    推論と再フィルタで**同じ解釈**をするために 1 箇所にまとめている。

    Args:
        general: box_fitting_params["lshape"]（共通の設定と対象ラベル）
        modality: depth_params または lidar_params（``lshape_*`` キー）

    Returns:
        無効なら None
    """
    general = general or {}
    modality = modality or {}
    if not general.get("enabled"):
        return None
    return {
        "bin_deg": float(modality.get("lshape_bin_deg", 0.5)),
        "quantile": float(modality.get("lshape_quantile", 10.0)),
        # LiDAR は lshape_margin（固定値）、深度は min と ratio
        "margin_min": float(modality.get(
            "lshape_margin_min", modality.get("lshape_margin", 0.3))),
        "margin_ratio": float(modality.get(
            "lshape_margin_ratio", margin_ratio_default)),
        "angle_step_deg": float(general.get("angle_step_deg", 1.0)),
        "adjust_range_deg": float(general.get("adjust_range_deg", 0.0)),
        "adjust_step_deg": float(general.get("adjust_step_deg", 1.0)),
        "boundary_quantile": float(general.get("boundary_quantile", 5.0)),
        "support_band": float(general.get("support_band", 0.3)),
        "violation_weight": float(general.get(
            "violation_weight", DEFAULT_VIOLATION_WEIGHT)),
    }


def lshape_applies(general: dict[str, Any] | None, label: str | None) -> bool:
    """このラベルに L 字フィルタを適用するか."""
    general = general or {}
    return bool(general.get("enabled")) and label in set(general.get("labels") or [])
