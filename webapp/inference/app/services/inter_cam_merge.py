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

# 診断に残す候補の下限。これ未満は「重なっていない」として記録しない
CANDIDATE_MIN_OVERLAP = 0.01
# 診断に残す辺の上限。全部残すとインスタンス数の 2 乗に比例して膨らむ
MAX_DIAGNOSTIC_EDGES = 200

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
        ``({(a index, b index): 重なり率}, [候補の記録])``。
        2 つ目は採用されなかった対も含む診断用のリストで、
        ``{"pair", "overlap", "status"}`` を持つ。

        status:
          accepted        … 採用
          taken_by_other  … 閾値は超えたが、Hungarian で相手が別の対に取られた
          below_threshold … 重なりが閾値未満

    **スコアを返すのが要点。** トラック単位の集計で
    「どの辺を優先して繋ぐか」を決めるのに使う。

    Hungarian で全体最適を取り、そのあと閾値未満を落とす
    （必ず最大マッチングが作られるため）。

    NOTE: 「閾値を緩めても結合されない」ときの原因は閾値ではなく、
    **1 対 1 の割り当てで別の相手に取られている**ことが多い。
    それを切り分けられるよう、候補の記録を返している。
    """
    if not instances_a or not instances_b:
        return {}, []

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
    assigned = {(int(i), int(j)) for i, j in zip(rows, cols)}
    matched = {
        pair: float(score[pair])
        for pair in assigned
        if score[pair] >= overlap_threshold
    }

    # 診断用。重なりが 0 の対は数が多いので記録しない
    candidates = []
    for i, j in zip(*np.nonzero(score > CANDIDATE_MIN_OVERLAP)):
        pair = (int(i), int(j))
        if pair in matched:
            status = "accepted"
        elif score[pair] >= overlap_threshold:
            status = "taken_by_other"
        else:
            status = "below_threshold"
        candidates.append(
            {"pair": pair, "overlap": float(score[pair]), "status": status}
        )
    return matched, candidates


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
        ``({(index, index): 重なり率}, [候補の記録])``。
        index は instances 内の位置。候補の記録は診断用で、
        採用されなかった対とその理由を含む。
    """
    by_channel: dict[str, list[int]] = {}
    for index, instance in enumerate(instances):
        by_channel.setdefault(instance["channel"], []).append(index)

    edges: dict[tuple[int, int], float] = {}
    candidates: list[dict[str, Any]] = []
    channels = sorted(by_channel)
    for a_pos in range(len(channels)):
        for b_pos in range(a_pos + 1, len(channels)):
            indices_a = by_channel[channels[a_pos]]
            indices_b = by_channel[channels[b_pos]]
            matched, pair_candidates = match_camera_pair(
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
            for entry in pair_candidates:
                local_a, local_b = entry["pair"]
                candidates.append({
                    **entry,
                    "pair": (indices_a[local_a], indices_b[local_b]),
                })
    return edges, candidates


# ── 結合（カメラ重複を禁止して繋ぐ）──────────────────────────────────────

def merge_by_edges(
    nodes: Sequence[dict[str, Any]],
    edges: dict[tuple[int, int], dict[str, Any]],
    *,
    min_match_frames: int = 1,
    max_same_camera_gap: int | None = None,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """辺の集計結果からグループを作る.

    Args:
        nodes: ``{"key", "channel", "frames"}`` のリスト。
            ``frames`` はそのトラックが存在するフレーム番号の集合（省略可）
        edges: ``{(i, j): {"frames": int, "max_overlap": float}}``
        min_match_frames: この回数以上マッチした辺だけを繋ぐ
        max_same_camera_gap: 同一カメラのトラックを同じグループへ入れてよい
            **時間的な隔たりの上限**（フレーム数）。None なら同一カメラを一切許さない

    Returns:
        ``(グループ, 却下した辺の記録)``。

    ## 同一カメラの扱い

    同一カメラの 2 トラックが同時に存在するなら、それは別物体なので繋がない。
    一方、**存在フレームが重ならない**なら、トラックが時間方向に分断された
    同じ物体でありうる（カメラ A → カメラ B → カメラ A と写り込む場合など）。
    その場合だけ許す。

    ただし「重ならない」だけだと、視界を出た車と後から入ってきた別の車が
    共通の相手を介して繋がりうる。隔たりが max_same_camera_gap を超える組は
    却下する。

    判定は **グループ全体**で行う。辺の両端だけを見ると、相手のグループに
    既に入っている同一カメラのトラックを見落とす。
    """
    union = UnionFind(len(nodes))
    channels = [n["channel"] for n in nodes]
    frames_of = [set(n.get("frames") or ()) for n in nodes]
    members = {index: {index} for index in range(len(nodes))}
    rejected: list[dict[str, Any]] = []

    def conflict(group_a: set[int], group_b: set[int]) -> str | None:
        """同じグループにできない理由。問題なければ None."""
        for i in group_a:
            for j in group_b:
                if channels[i] != channels[j]:
                    continue
                if frames_of[i] & frames_of[j]:
                    return "同一カメラで同時に存在"
                if max_same_camera_gap is None:
                    return "同一カメラが重複"
                if frames_of[i] and frames_of[j]:
                    gap = max(
                        min(frames_of[j]) - max(frames_of[i]),
                        min(frames_of[i]) - max(frames_of[j]),
                    )
                    if gap > max_same_camera_gap:
                        return f"同一カメラで時間が離れすぎ（{gap} フレーム）"
        return None

    candidates = sorted(
        (
            (stats.get("max_overlap", 0.0), pair)
            for pair, stats in edges.items()
            if stats.get("frames", 0) >= min_match_frames
        ),
        key=lambda item: -item[0],
    )
    for pair, stats in edges.items():
        if stats.get("frames", 0) < min_match_frames:
            rejected.append({
                "pair": pair, "max_overlap": stats.get("max_overlap", 0.0),
                "frames": stats.get("frames", 0),
                "reason": f"マッチしたフレーム数が {min_match_frames} 未満",
            })

    for overlap, (a, b) in candidates:
        root_a, root_b = union.find(a), union.find(b)
        if root_a == root_b:
            continue
        reason = conflict(members[root_a], members[root_b])
        if reason:
            rejected.append({
                "pair": (a, b), "max_overlap": overlap,
                "frames": edges[(a, b)].get("frames", 0), "reason": reason,
            })
            continue
        union.union(a, b)
        root = union.find(a)
        members[root] = members[root_a] | members[root_b]

    return union.groups(), rejected


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

    edges, _candidates = match_frame(
        instances,
        overlap_threshold=overlap_threshold,
        max_centroid_distance=max_centroid_distance,
        overlap_method=overlap_method,
        label_match=label_match,
        label_to_category_group=label_to_category_group,
    )
    groups, _rejected = merge_by_edges(
        [{"key": str(i), "channel": inst["channel"]}
         for i, inst in enumerate(instances)],
        {pair: {"frames": 1, "max_overlap": value} for pair, value in edges.items()},
        min_match_frames=1,
    )
    return groups


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
    max_same_camera_gap: int | None = None,
    start_id: int = 0,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """トラックを結合し、グローバル ID を割り当てる.

    Args:
        tracks: ``{"key", "channel", "frames"}`` のリスト（key は一意な文字列）
        frame_matches: フレームごとの照合結果（キー対 → 重なり率）
        max_same_camera_gap: 同一カメラのトラックを同居させてよい隔たり

    Returns:
        ``({トラックキー: global_track_id}, 却下した辺の記録)``
    """
    keys = [t["key"] for t in tracks]
    index_of = {key: i for i, key in enumerate(keys)}
    stats = aggregate_track_edges(frame_matches)

    edges = {
        (index_of[a], index_of[b]): value
        for (a, b), value in stats.items()
        if a in index_of and b in index_of
    }
    groups, rejected = merge_by_edges(
        tracks, edges,
        min_match_frames=min_match_frames,
        max_same_camera_gap=max_same_camera_gap,
    )

    assigned: dict[str, str] = {}
    for offset, group in enumerate(groups):
        global_id = str(start_id + offset)
        for index in group:
            assigned[keys[index]] = global_id
    # 却下の記録はトラックキーで返す（index だと呼び出し側が読めない）
    rejected = [
        {**r, "pair": (keys[r["pair"][0]], keys[r["pair"][1]])} for r in rejected
    ]
    return assigned, rejected


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
    from common.point_ops import dedupe_points

    hulls = [
        s["hull_xy"] for s in summaries
        if s.get("hull_xy") is not None and len(s["hull_xy"])
    ]
    if not hulls:
        return {"points": np.empty((0, 3)), "z_range": None, "num_points": 0}

    # 凸包の頂点も、カメラ跨ぎで同じ点が入りうるのでまとめる
    # （LiDAR 点は両カメラが同じ sweep から選ぶため重複する）。
    # 凸包の計算結果は変わらないが、点数の集計が実態に合う
    hull_xy = dedupe_points(
        np.column_stack([np.vstack(hulls), np.zeros(len(np.vstack(hulls)))])
    )[:, :2]
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


# ── パス 2 のオーケストレーション ────────────────────────────────────────

def _channel_of(entry: dict[str, Any]) -> str:
    """entry からカメラ名を取り出す.

    パイプラインの戻り値には channel が含まれないため、呼び出し側が
    ``_channel`` として付ける。どちらでも読めるようにしておく。
    """
    return entry.get("channel") or entry.get("_channel") or ""


def track_key(channel: str, track_id: Any) -> str:
    """トラックの一意キー。track_id はカメラ内でのみ一意なので channel を含める."""
    return f"{channel}:{track_id}"


def _as_xyz(hull_xy: np.ndarray) -> np.ndarray:
    """凸包の頂点を (N, 3) にする（重なり判定は XY しか見ない）."""
    hull_xy = np.asarray(hull_xy, dtype=np.float64)
    return np.column_stack([hull_xy, np.zeros(len(hull_xy))])


def merge_and_fit(
    entries_by_sample: dict[str, list[dict[str, Any]]],
    *,
    merge_params: dict[str, Any],
    box_fitting_params: dict[str, Any],
    label_to_category_group: dict[str, str] | None = None,
) -> dict[str, Any]:
    """カメラ跨ぎで結合し、グループごとに 3D ボックスを当てはめる.

    Args:
        entries_by_sample: ``{sample_token: [entry, ...]}``。
            entry は ``channel`` / ``track_id`` / ``label`` / ``score`` /
            ``_summary`` を持つ（パイプラインが fit_boxes=False で返したもの）
        merge_params: ``overlap_threshold`` / ``max_centroid_distance`` /
            ``min_match_frames`` / ``label_match``

    Returns:
        ``{"num_groups", "num_merged"}``。entry は **その場で書き換える**
        （``global_track_id`` / ``label`` / ``status`` / ボックスの各値）。

    ## 流れ

    1. sample ごとにカメラ対で照合（BEV 凸包の IoS）
    2. トラック単位で集計し、global_track_id を割り当てる
    3. グローバルトラック全体でラベルを多数決
    4. sample ごとに、グループの要約を結合して当てはめる
       （ボックスは代表 1 行にだけ入れ、他は status="merged"）
    """
    from app.services.box_fitting import fit_box

    overlap_threshold = float(merge_params.get("overlap_threshold", 0.3))
    max_distance = float(merge_params.get("max_centroid_distance", 3.0))
    min_frames = int(merge_params.get("min_match_frames", 1))
    label_match = merge_params.get("label_match", LABEL_MATCH_CATEGORY_GROUP)

    # --- 1) sample ごとの照合 ---------------------------------------------
    # 判定は各トラックの多数決ラベルで行うので、先にラベルを集める
    labels_by_track: dict[str, list[tuple[str, float]]] = {}
    # トラックごとの存在フレーム。sample の並び順を番号として使う
    frames_by_track: dict[str, set[int]] = {}
    for sample_index, entries in enumerate(entries_by_sample.values()):
        for entry in entries:
            if entry.get("_summary") is None:
                continue
            key = track_key(_channel_of(entry), entry["track_id"])
            frames_by_track.setdefault(key, set()).add(sample_index)
            labels_by_track.setdefault(key, []).append(
                (entry.get("label"), float(entry.get("score") or 0.0))
            )
    track_label = {
        key: majority_label([l for l, _ in items], [s for _, s in items])
        for key, items in labels_by_track.items()
    }

    frame_matches: list[dict[tuple[str, str], float]] = []
    # {トラック対: {"overlap": 最大, "status": 最良時の扱い}}
    candidate_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for entries in entries_by_sample.values():
        usable = [e for e in entries if e.get("_summary") is not None]
        if len(usable) < 2:
            continue
        instances = [
            {
                "channel": _channel_of(e),
                # 多数決ラベルで判定する（トラック内で揺れることがある）
                "label": track_label.get(track_key(_channel_of(e), e["track_id"])),
                "points": _as_xyz(e["_summary"]["hull_xy"]),
            }
            for e in usable
        ]
        edges, pair_candidates = match_frame(
            instances,
            overlap_threshold=overlap_threshold,
            max_centroid_distance=max_distance,
            label_match=label_match,
            label_to_category_group=label_to_category_group,
        )
        key_of = lambda k: track_key(_channel_of(usable[k]), usable[k]["track_id"])
        frame_matches.append({
            (key_of(i), key_of(j)): value for (i, j), value in edges.items()
        })
        # 診断用。採用されなかった対も、トラック対ごとに最良の記録だけ残す
        for entry in pair_candidates:
            i, j = entry["pair"]
            pair = tuple(sorted((key_of(i), key_of(j))))
            previous = candidate_stats.get(pair)
            if previous is None or entry["overlap"] > previous["overlap"]:
                candidate_stats[pair] = {
                    "overlap": entry["overlap"], "status": entry["status"]
                }

    # --- 2) トラック単位の集計 -------------------------------------------
    tracks = [
        {
            "key": key, "channel": key.rsplit(":", 1)[0],
            # 同一カメラのトラックが同時に存在するかの判定に使う
            "frames": frames_by_track.get(key, set()),
        }
        for key in sorted(labels_by_track)
    ]
    global_ids, rejected_edges = assign_global_track_ids(
        tracks, frame_matches,
        min_match_frames=min_frames,
        max_same_camera_gap=merge_params.get("max_same_camera_gap"),
    )

    # --- 3) グローバルトラック単位でラベルを多数決 -----------------------
    by_global: dict[str, list[tuple[str, float]]] = {}
    for key, items in labels_by_track.items():
        by_global.setdefault(global_ids.get(key, key), []).extend(items)
    global_label = {
        gid: majority_label([l for l, _ in items], [s for _, s in items])
        for gid, items in by_global.items()
    }

    # --- 4) sample ごとに結合して当てはめる ------------------------------
    num_merged = 0
    for entries in entries_by_sample.values():
        groups: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            summary = entry.get("_summary")
            key = track_key(_channel_of(entry), entry["track_id"])
            global_id = global_ids.get(key, key)
            entry["global_track_id"] = global_id
            # 最終ラベルはグローバルトラック全体の多数決
            entry["label"] = global_label.get(global_id) or entry.get("label")
            if summary is not None:
                groups.setdefault(global_id, []).append(entry)

        for global_id, members in groups.items():
            summaries = [m["_summary"] for m in members]
            combined = combine_summaries(summaries)
            if len(members) > 1:
                num_merged += len(members)

            # 点数が最も多いカメラの位置を原点にする
            origin = dominant_sensor_origin(
                [{"points": np.empty((s["num_points"], 3)),
                  "sensor_origin_xy": s.get("sensor_origin_xy")} for s in summaries],
                list(range(len(summaries))),
            )
            fit = fit_box(
                combined["points"],
                box_fitting_params.get("method", "convex_hull_moa"),
                box_fitting_params,
                sensor_origin_xy=origin,
                z_range=combined["z_range"],
            )
            # ボックスは代表 1 行だけに入れる。全行へ入れると BEV 表示で
            # 同じ箱が重なって見える
            primary = max(members, key=lambda m: m["_summary"]["num_points"])
            for member in members:
                member.pop("_summary", None)
                if member is primary and fit is not None:
                    member.update(fit)
                    member["status"] = "fitted"
                    member["is_primary"] = True
                elif member is primary:
                    member["status"] = "not_fitted"
                    member["is_primary"] = True
                else:
                    # 代表行のボックスに含まれる（点群は自分のものを保持）
                    member["status"] = "merged"
                    member["is_primary"] = False

    # --- 5) 診断情報 -----------------------------------------------------
    # 「なぜ結合されなかったか」を後から確認できるようにする。
    # 閾値を緩めても結合されない場合、原因は閾値ではなく
    # 1 対 1 の割り当てやカメラ重複の保護であることが多い
    groups_by_id: dict[str, list[str]] = {}
    for key, global_id in global_ids.items():
        groups_by_id.setdefault(global_id, []).append(key)

    rejected_by_pair = {tuple(sorted(r["pair"])): r for r in rejected_edges}
    edge_rows: list[dict[str, Any]] = []
    for pair, stats in sorted(
        candidate_stats.items(), key=lambda kv: -kv[1]["overlap"]
    ):
        same_group = global_ids.get(pair[0]) == global_ids.get(pair[1])
        rejected = rejected_by_pair.get(pair)
        if same_group:
            status, reason = "accepted", ""
        elif rejected:
            status, reason = "rejected", rejected["reason"]
        elif stats["status"] == "taken_by_other":
            status = "rejected"
            reason = "1 対 1 の割り当てで別の相手に取られた"
        else:
            status = "rejected"
            reason = f"重なりが閾値（{overlap_threshold}）未満"
        edge_rows.append({
            "pair": list(pair), "max_overlap": round(stats["overlap"], 4),
            "status": status, "reason": reason,
        })

    diagnostics = {
        "groups": [
            {
                "global_track_id": global_id,
                "members": sorted(members_),
                "label": global_label.get(global_id),
                "num_frames": len(
                    set().union(*(frames_by_track.get(k, set()) for k in members_))
                ),
            }
            for global_id, members_ in sorted(groups_by_id.items(), key=lambda kv: kv[0])
        ],
        # 重なりの大きい順。結合されなかった対の理由がここに出る
        "edges": edge_rows[:MAX_DIAGNOSTIC_EDGES],
        "num_edges_total": len(edge_rows),
        "params": {
            "overlap_threshold": overlap_threshold,
            "max_centroid_distance": max_distance,
            "min_match_frames": min_frames,
            "label_match": label_match,
            "max_same_camera_gap": merge_params.get("max_same_camera_gap"),
        },
    }

    return {
        "num_groups": len(set(global_ids.values())),
        "num_merged": num_merged,
        "diagnostics": diagnostics,
    }
