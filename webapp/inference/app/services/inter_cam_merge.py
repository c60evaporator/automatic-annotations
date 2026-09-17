"""カメラを跨いだ同一インスタンスの結合.

## なぜ点群化してから結合するのか

2D の段階では、あるカメラのマスクを他カメラへ投影しても一意に決まらない
（深度が分からないとエピポーラ線にしかならない）。しかも隣接カメラの
重なり領域は狭く、そこでは物体が画像端で切れていて形も不完全なので、
**判定材料が最も乏しい場所で最も難しい判定を強いられる**。
3D にしてから位置と形で判定するほうが確実。

## 判定の流れ

1. **フレームごと**にカメラ対で照合（BEV 凸包の重なり、Hungarian、閾値）
2. 辺ごとに「マッチしたフレーム数」と「最大の重なり率」を集計
3. **トラック単位**で結合（min_match_frames 以上マッチした辺を、
   重なり率の高い順に繋ぐ）
4. グローバルトラック全体でラベルを多数決

## 設計上の判断

- **IoU ではなく IoS**（小さい方の面積に対する重なり率）を既定にする。
  カメラごとに物体の違う面しか観測できないため、同一物体でも IoU は
  0.3 程度まで落ちる（IoS なら 0.5 前後）。閾値を切りやすいほうを採る。
- **割合ではなくフレーム数で判定する。** 同じ物体が複数カメラに写る
  キーフレームは高々 1〜2 なので、割合にすると 0 / 0.5 / 1 の 3 値しか
  取らず、閾値に意味が無い。
- **同一カメラの 2 トラックを同じ群に入れない。** 定義上別物体なので、
  重なり率の高い辺から順に繋ぎ、カメラが重複する結合は却下する。
  これをやらないと「フレーム 3 では A、フレーム 7 では B とマッチ」
  のような取り違えで、連鎖して同一カメラの 2 台が 1 つになる。

NOTE: 点群はすべて **sample の基準 ego 座標**へ揃えてから渡すこと
（common.transform3d.ego_to_ego）。カメラごとに sample_data の時刻が
違うため、揃えないと最大 0.6 m ずれて判定が壊れる。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

MERGE_METHOD_BEV_HULL = "BEV convex-hull"
MERGE_METHODS = (MERGE_METHOD_BEV_HULL,)

# 重なりの測り方
OVERLAP_IOS = "ios"      # 小さい方に対する重なり率（既定）
OVERLAP_IOU = "iou"

LABEL_MATCH_LABEL = "label"
LABEL_MATCH_CATEGORY_GROUP = "category_group"
LABEL_MATCH_NONE = "none"


class UnionFind:
    """連結成分をまとめる（3 カメラ以上に写る物体のため）."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, index: int) -> int:
        while self._parent[index] != index:
            # 経路圧縮
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def groups(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = {}
        for index in range(len(self._parent)):
            buckets.setdefault(self.find(index), []).append(index)
        return [sorted(v) for v in buckets.values()]


# ── 重なりの計算 ──────────────────────────────────────────────────────────

def _hull(points_xy: np.ndarray):
    """BEV の凸包ポリゴンを返す（面積が取れない場合は None）."""
    from shapely.geometry import MultiPoint

    if points_xy.shape[0] < 3:
        return None
    hull = MultiPoint([tuple(p) for p in points_xy]).convex_hull
    return hull if getattr(hull, "area", 0.0) > 0 else None


def hull_overlap(
    points_a: np.ndarray, points_b: np.ndarray, *, method: str = OVERLAP_IOS
) -> float:
    """BEV 凸包の重なり率.

    Args:
        method: ``"ios"`` なら小さい方の面積で割る（部分観測に寛容）、
            ``"iou"`` なら和集合で割る
    """
    hull_a = _hull(np.asarray(points_a, dtype=np.float64)[:, :2])
    hull_b = _hull(np.asarray(points_b, dtype=np.float64)[:, :2])
    if hull_a is None or hull_b is None:
        return 0.0

    intersection = hull_a.intersection(hull_b).area
    if intersection <= 0:
        return 0.0
    denominator = (
        hull_a.union(hull_b).area if method == OVERLAP_IOU
        else min(hull_a.area, hull_b.area)
    )
    return float(intersection / denominator) if denominator > 0 else 0.0


def centroid_xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points[:, :2].mean(axis=0) if points.shape[0] else np.zeros(2)


def labels_compatible(
    label_a: str | None,
    label_b: str | None,
    label_match: str,
    label_to_category_group: dict[str, str] | None,
) -> bool:
    """結合してよいラベルの組か（トラッキングと同じ基準を使う）.

    トラック内でラベルが揺れることがあるので、呼び出し側では
    **多数決で決めたラベル**を渡すこと（majority_label）。
    """
    if label_match == LABEL_MATCH_NONE:
        return True
    if label_match == LABEL_MATCH_CATEGORY_GROUP:
        mapping = label_to_category_group or {}
        # グループが引けないラベルは、ラベル名そのものをグループ扱いにする
        return mapping.get(label_a, label_a) == mapping.get(label_b, label_b)
    return label_a == label_b


def majority_label(
    labels: Iterable[str], scores: Iterable[float] | None = None
) -> str | None:
    """ラベルの多数決.

    同数の場合は **スコア合計が大きいほう**を採る。
    スコアが無い場合はラベル名の昇順（結果を決定的にするため）。
    """
    labels = [l for l in labels if l]
    if not labels:
        return None

    counts = Counter(labels)
    top = max(counts.values())
    candidates = sorted(l for l, c in counts.items() if c == top)
    if len(candidates) == 1:
        return candidates[0]

    if scores is None:
        return candidates[0]
    totals: dict[str, float] = {}
    for label, score in zip(labels, scores):
        if label in candidates:
            totals[label] = totals.get(label, 0.0) + float(score or 0.0)
    return max(candidates, key=lambda l: (totals.get(l, 0.0), l))


# ── フレーム単位の照合 ────────────────────────────────────────────────────

def match_camera_pair(
    instances_a: Sequence[dict[str, Any]],
    instances_b: Sequence[dict[str, Any]],
    *,
    overlap_threshold: float,
    max_centroid_distance: float,
    overlap_method: str = OVERLAP_IOS,
    label_match: str = LABEL_MATCH_CATEGORY_GROUP,
    label_to_category_group: dict[str, str] | None = None,
) -> dict[tuple[int, int], float]:
    """2 カメラ間で 1 対 1 の対応を求める.

    Args:
        instances_a / instances_b: ``{"points", "label"}`` を持つ辞書

    Returns:
        ``{(a の index, b の index): 重なり率}``。閾値未満は含まれない。

    **スコアを返すのが要点。** トラック単位の集計で
    「どの辺を優先して繋ぐか」を決めるのに使う。

    Hungarian で全体最適を取り、そのあと閾値未満を落とす
    （必ず最大マッチングが作られるため）。
    """
    if not instances_a or not instances_b:
        return {}

    score = np.zeros((len(instances_a), len(instances_b)))
    for i, a in enumerate(instances_a):
        centroid_a = centroid_xy(a["points"])
        for j, b in enumerate(instances_b):
            if not labels_compatible(
                a.get("label"), b.get("label"), label_match, label_to_category_group
            ):
                continue
            # 重心が離れすぎている組は凸包を作る前に捨てる（計算も減る）
            distance = np.linalg.norm(centroid_a - centroid_xy(b["points"]))
            if distance > max_centroid_distance:
                continue
            score[i, j] = hull_overlap(
                a["points"], b["points"], method=overlap_method
            )

    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-score)
    return {
        (int(i), int(j)): float(score[i, j])
        for i, j in zip(rows, cols)
        if score[i, j] >= overlap_threshold
    }


def match_frame(
    instances: Sequence[dict[str, Any]],
    *,
    overlap_threshold: float,
    max_centroid_distance: float,
    overlap_method: str = OVERLAP_IOS,
    label_match: str = LABEL_MATCH_CATEGORY_GROUP,
    label_to_category_group: dict[str, str] | None = None,
) -> dict[tuple[int, int], float]:
    """1 フレーム分のインスタンスを、カメラ対ごとに照合する.

    Args:
        instances: ``{"channel", "label", "points"}``（基準 ego 座標）

    Returns:
        ``{(index, index): 重なり率}``。index は instances 内の位置。
    """
    by_channel: dict[str, list[int]] = {}
    for index, instance in enumerate(instances):
        by_channel.setdefault(instance["channel"], []).append(index)

    edges: dict[tuple[int, int], float] = {}
    channels = sorted(by_channel)
    for a_pos in range(len(channels)):
        for b_pos in range(a_pos + 1, len(channels)):
            indices_a = by_channel[channels[a_pos]]
            indices_b = by_channel[channels[b_pos]]
            matched = match_camera_pair(
                [instances[i] for i in indices_a],
                [instances[j] for j in indices_b],
                overlap_threshold=overlap_threshold,
                max_centroid_distance=max_centroid_distance,
                overlap_method=overlap_method,
                label_match=label_match,
                label_to_category_group=label_to_category_group,
            )
            for (local_a, local_b), value in matched.items():
                edges[(indices_a[local_a], indices_b[local_b])] = value
    return edges


# ── 結合（カメラ重複を禁止して繋ぐ）──────────────────────────────────────

def merge_by_edges(
    channels: Sequence[str],
    edges: dict[tuple[int, int], dict[str, Any]],
    *,
    min_match_frames: int = 1,
) -> list[list[int]]:
    """辺の集計結果からグループを作る.

    Args:
        channels: ノードごとのカメラ名（同一カメラの重複を防ぐのに使う）
        edges: ``{(i, j): {"frames": int, "max_overlap": float}}``
        min_match_frames: この回数以上マッチした辺だけを繋ぐ

    Returns:
        グループ（ノード index のリスト）。単独のノードも 1 要素で含まれる。

    **重なり率の高い辺から順に繋ぎ、カメラが重複する結合は却下する。**
    同一カメラの 2 トラックは定義上別物体なので、連鎖で同じ群に
    入るのを防ぐ必要がある。
    """
    union = UnionFind(len(channels))
    # 群ごとに「含んでいるカメラ」を持ち、重複を検出する
    members = {index: {channel} for index, channel in enumerate(channels)}

    candidates = sorted(
        (
            (stats.get("max_overlap", 0.0), pair)
            for pair, stats in edges.items()
            if stats.get("frames", 0) >= min_match_frames
        ),
        key=lambda item: -item[0],
    )

    for overlap, (a, b) in candidates:
        root_a, root_b = union.find(a), union.find(b)
        if root_a == root_b:
            continue
        if members[root_a] & members[root_b]:
            logger.debug(
                "却下: ノード %d-%d（重なり %.3f）はカメラが重複", a, b, overlap
            )
            continue
        union.union(a, b)
        root = union.find(a)
        members[root] = members[root_a] | members[root_b]

    return union.groups()


def merge_instances(
    instances: Sequence[dict[str, Any]],
    *,
    overlap_threshold: float = 0.3,
    max_centroid_distance: float = 3.0,
    overlap_method: str = OVERLAP_IOS,
    label_match: str = LABEL_MATCH_CATEGORY_GROUP,
    label_to_category_group: dict[str, str] | None = None,
) -> list[list[int]]:
    """1 フレーム分のインスタンスをカメラ跨ぎで結合する（フレーム内完結）.

    Returns:
        グループ（instances の index のリスト）。

    3 カメラ以上では、フレーム内でも連鎖で同一カメラが重複しうるので
    merge_by_edges を通す。
    """
    if len(instances) < 2:
        return UnionFind(len(instances)).groups()

    edges = match_frame(
        instances,
        overlap_threshold=overlap_threshold,
        max_centroid_distance=max_centroid_distance,
        overlap_method=overlap_method,
        label_match=label_match,
        label_to_category_group=label_to_category_group,
    )
    return merge_by_edges(
        [i["channel"] for i in instances],
        {pair: {"frames": 1, "max_overlap": value} for pair, value in edges.items()},
        min_match_frames=1,
    )


# ── トラック単位の集計 ────────────────────────────────────────────────────

def aggregate_track_edges(
    frame_matches: Iterable[dict[tuple[str, str], float]]
) -> dict[tuple[str, str], dict[str, Any]]:
    """フレームごとの照合結果を、トラック対ごとに集計する.

    Args:
        frame_matches: フレームごとの ``{(トラックキー, トラックキー): 重なり率}``。
            トラックキーは ``f"{channel}:{track_id}"`` のような一意な文字列

    Returns:
        ``{(キー, キー): {"frames": マッチ回数, "max_overlap": 最大の重なり率}}``

    代表スコアに **最大値**を使う。平均にすると、片方のカメラに長く
    写っていたぶんスコアが薄まってしまう。
    """
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for matches in frame_matches:
        for pair, overlap in matches.items():
            key = tuple(sorted(pair))
            entry = stats.setdefault(key, {"frames": 0, "max_overlap": 0.0})
            entry["frames"] += 1
            entry["max_overlap"] = max(entry["max_overlap"], float(overlap))
    return stats


def assign_global_track_ids(
    tracks: Sequence[dict[str, Any]],
    frame_matches: Iterable[dict[tuple[str, str], float]],
    *,
    min_match_frames: int = 1,
    start_id: int = 0,
) -> dict[str, str]:
    """トラックを結合し、グローバル ID を割り当てる.

    Args:
        tracks: ``{"key", "channel"}`` のリスト（key は一意な文字列）
        frame_matches: フレームごとの照合結果（キー対 → 重なり率）

    Returns:
        ``{トラックキー: global_track_id}``
    """
    keys = [t["key"] for t in tracks]
    index_of = {key: i for i, key in enumerate(keys)}
    stats = aggregate_track_edges(frame_matches)

    edges = {
        (index_of[a], index_of[b]): value
        for (a, b), value in stats.items()
        if a in index_of and b in index_of
    }
    groups = merge_by_edges(
        [t["channel"] for t in tracks], edges, min_match_frames=min_match_frames
    )

    assigned: dict[str, str] = {}
    for offset, group in enumerate(groups):
        global_id = str(start_id + offset)
        for index in group:
            assigned[keys[index]] = global_id
    return assigned


# ── パス間で持ち越す要約 ──────────────────────────────────────────────────

def summarize_instance(
    points: np.ndarray,
    *,
    z_percentiles: tuple[float, float] | None = (1.0, 99.0),
) -> dict[str, Any]:
    """インスタンス点群を、結合と当てはめに必要な最小限へ圧縮する.

    Returns:
        ``{"hull_xy", "z_range", "num_points"}``

    **BEV は凸包の頂点だけで厳密に扱える**（凸包の和集合＝結合後の凸包）。
    全点を持ち越すと 1 シーンで 100MB 超になるが、頂点なら 1MB 未満で済む。

    z は ``z_percentiles`` の範囲を保持し、結合時は和集合（min/max）を取る。
    分位点サンプルに圧縮して結合後に再計算する方式は誤差が大きい
    （64 分位点で高さ 0.6m 誤差）。各カメラで個別に外れ値を落としてから
    和集合を取るほうが、誤差も小さく（0.15m）混入にも強い。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return {"hull_xy": np.empty((0, 2)), "z_range": None, "num_points": 0}

    from app.services.box_fitting import convex_hull_xy

    hull = (
        convex_hull_xy(points[:, :2]) if points.shape[0] >= 3
        else points[:, :2].copy()
    )
    if z_percentiles is None:
        z_low, z_high = float(points[:, 2].min()), float(points[:, 2].max())
    else:
        z_low, z_high = (
            float(z) for z in np.percentile(points[:, 2], list(z_percentiles))
        )
    return {
        "hull_xy": np.asarray(hull, dtype=np.float64),
        "z_range": (z_low, z_high),
        "num_points": int(points.shape[0]),
    }


def combine_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """グループ内の要約をまとめる.

    Returns:
        ``{"points", "z_range", "num_points"}``。
        ``points`` は当てはめへ渡す ``(N, 3)``（XY は凸包の頂点の集合、
        z は範囲の中央で埋める。実際の高さは z_range で渡す）。
    """
    hulls = [
        s["hull_xy"] for s in summaries
        if s.get("hull_xy") is not None and len(s["hull_xy"])
    ]
    if not hulls:
        return {"points": np.empty((0, 3)), "z_range": None, "num_points": 0}

    hull_xy = np.vstack(hulls)
    ranges = [s["z_range"] for s in summaries if s.get("z_range")]
    if ranges:
        z_low = min(r[0] for r in ranges)
        z_high = max(r[1] for r in ranges)
    else:
        z_low = z_high = 0.0

    return {
        "points": np.column_stack([hull_xy, np.full(len(hull_xy), (z_low + z_high) / 2)]),
        "z_range": (z_low, z_high),
        "num_points": sum(int(s.get("num_points", 0)) for s in summaries),
    }


def dominant_sensor_origin(
    instances: Sequence[dict[str, Any]], group: Sequence[int]
) -> tuple[float, float]:
    """グループの MOA 用センサー原点を決める.

    点数が最も多いカメラの位置を使う。MOA は単一のセンサー原点を
    前提にしているため、結合後の点群では厳密には成り立たない。
    ただし両側が観測できている状況ではオクルージョン面積が小さくなり、
    向きの推定は原点の取り方に鈍感になる。
    """
    best = max(
        group, key=lambda index: np.asarray(instances[index]["points"]).shape[0]
    )
    origin = instances[best].get("sensor_origin_xy") or (0.0, 0.0)
    return (float(origin[0]), float(origin[1]))
