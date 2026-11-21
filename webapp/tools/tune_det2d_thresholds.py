"""Detection2D の後処理パラメータを、キャッシュ（tools/build_det2d_cache.py）を使って Optuna で探索する.

推論はやり直さず、キャッシュした検出に後処理をかけ直して GT と比べる。1 試行は数秒で済む。

## 探索するパラメータ（--space all のとき）

  - スコア閾値（カテゴリグループごと）  DET2D_DEFAULT_SCORE_THRESHOLDS
  - 同一クラス NMS の IoU（グループごと） DET2D_NMS_SAME_CLASS_IOUS
  - クラス間 NMS の IoU                  DET2D_NMS_CROSS_CLASS_IOU
  - IoS 結合の閾値                        DET2D_IOS_DELETE_THRESHOLD
  - 再判定の削除マージン（**config に未実装の新しいパラメータ**）:
    「使わない」候補（RE_CLASSIFICATION_CANDIDATES の値が None）の最大スコアが、残す候補の
    最大スコアよりマージン以上高いときだけ削除する。0 なら現在の挙動（スコア最大の候補で決める）

--space thresholds ならスコア閾値だけを探し、残りは config の値（マージンは 0）に固定する。

## 後処理の再現（推論サーバーと同じ順）

  1. スコアが閾値より大きい検出を残す
  2. 同一ラベルの NMS（IoU が閾値より大きいものを落とす）→ グループ内のクラス間 NMS
  3. 全グループをまとめたクラス間 NMS（IoU が閾値以上のものを落とす）
  4. SigLIP2 の再判定（キャッシュした全候補のスコアから、マージン付きで判定し直す）
  5. IoS 結合（postprocess.suppress_contained_boxes と同じ判定）

検出 0 件のフレームの救済（Whole-Image Fallback）は再現しない。

## 評価

check_det2d_errors.py と同じ数え方: ラベルを問わず IoU（--match-iou）で貪欲に 1 対 1 に対応付け、
ラベルが同じなら正解、違えばラベル違い（検出ラベルの誤検出かつ GT ラベルの見逃し）とする。
visibility が --min-visibility 未満の GT と、検出ラベルに対応しないカテゴリの GT は無視する。

目的関数はラベルごとの Fβ（--beta。既定 2 で recall を重視）の平均。探索に使うシーンで
GT が --min-label-gt 個以上あるラベルだけを平均する（数個しかないラベルで振り回されないため）。

シーンは探索用と確認用（--holdout の割合）に分け、最後に現行の設定と最良の設定を両方で比べる。
確認用で大きく下がるなら、探索用のシーンに合わせすぎている。

試行の記録は Optuna の SQLite（キャッシュのフォルダの optuna.db）に残るので、中断しても同じ
オプションで実行すれば続きから再開する。

## 使い方（docker-compose.yml のある webapp フォルダで）

推論サーバーは不要。

    docker compose run --rm --no-deps -T webapp python - [オプション] < tools/tune_det2d_thresholds.py

オプション（いずれも省略可）:
  --name NAME          キャッシュの名前（既定 default）
  --max-scenes N       使うシーン数（キャッシュを作った順に先頭から。既定: キャッシュ全部）
  --holdout R          確認用に取り分けるシーンの割合（既定 0.3。0 なら全部を探索に使う）
  --trials N           試行回数（既定 500。再開時は合計がこの回数になるまで続ける）
  --startup-trials N   最初にランダムに探す回数（既定 30）
  --space SPACE        all / thresholds（既定 all）
  --beta B             Fβ の β（既定 2.0）
  --min-precision P    全体の precision がこれを下回る試行は、下回った分だけ目的関数から引く
                       （既定 0 = 制約なし。β を大きくすると precision が大きく下がるため）
  --min-label-gt N     目的関数に入れるラベルの GT 数の下限（既定 20）
  --match-iou IOU      GT と対応付ける IoU の下限（既定 0.3）
  --min-visibility V   この visibility 未満の GT は無視（既定 2）
  --seed N             シーンの分け方と Optuna の乱数の種（既定 0）
  --study NAME         Optuna の study 名（既定はオプションから自動で決める）
  --log-interval SEC   進捗を表示する間隔（秒。既定 30）

例:
    docker compose run --rm --no-deps -T webapp python - --max-scenes 10 < tools/tune_det2d_thresholds.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.services.label_service import label_groups, nusc_category_to_label

# 探索範囲（0.01 刻み）
SCORE_RANGE = (0.10, 0.50)
NMS_SAME_RANGE = (0.40, 0.90)
NMS_CROSS_RANGE = (0.50, 0.95)
IOS_RANGE = (0.60, 1.00)
MARGIN_RANGE = (0.00, 0.30)
STEP = 0.01

MULTI_BIG_MIN_CHILDREN = 2  # postprocess.MULTI_BIG_MIN_CHILDREN と同じ


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tune_det2d_thresholds")
    parser.add_argument("--name", default="default")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--startup-trials", type=int, default=30)
    parser.add_argument("--space", choices=("all", "thresholds"), default="all")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-label-gt", type=int, default=20)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--min-visibility", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", default=None)
    parser.add_argument("--log-interval", type=float, default=30.0)
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _fmt(sec: float | None) -> str:
    if sec is None:
        return "--:--"
    sec = int(round(sec))
    h, rest = divmod(sec, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ── キャッシュの読み込みと前計算 ──────────────────────────────────────────

def _pair_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    smaller = np.minimum(area_a[:, None], area_b[None, :])
    ios = np.where(smaller > 0, inter / np.maximum(smaller, 1e-9), 0.0)
    return iou, ios


def _label_id(labels: dict[str, int], name: str) -> int:
    """ラベルの番号. config に無いラベル（プロンプトの語がそのまま残ったもの等）は末尾に足す."""
    return labels.setdefault(name, len(labels))


class Frame:
    """1 フレーム分の検出と GT. 試行ごとに変わらない量（IoU 等）は読み込み時に計算しておく.

    検出はスコアの降順に並べる（NMS・IoS 結合は推論サーバーでもこの順で処理される）。
    """

    def __init__(self, frame: dict, labels: dict[str, int], groups: dict[str, int],
                 candidates: dict, args: argparse.Namespace) -> None:
        boxes = sorted(frame["boxes"], key=lambda b: -b["score"])
        n = len(boxes)
        self.n = n
        xyxy = np.array([[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in boxes],
                        dtype=float).reshape(n, 4)
        self.score = np.array([b["score"] for b in boxes])
        self.group = np.array([groups[b["group"]] for b in boxes], dtype=int)
        self.det_label = np.array([_label_id(labels, b["detection_label"] or b["label"])
                                   for b in boxes], dtype=int)
        iou, ios = _pair_iou(xyxy, xyxy)
        upper = np.triu(np.ones((n, n), dtype=bool), 1)  # 自分より後（スコアが低い）の検出
        self.iou_upper = np.where(upper, iou, -1.0)
        self.ios = ios
        self.same_label = self.det_label[:, None] == self.det_label[None, :]
        self.same_group = self.group[:, None] == self.group[None, :]
        area = (xyxy[:, 2] - xyxy[:, 0]).clip(0) * (xyxy[:, 3] - xyxy[:, 1]).clip(0)
        # (面積, 番号) の順位。面積が同じときも番号で大小が決まる（postprocess と同じ）
        self.rank = np.empty(n, dtype=int)
        self.rank[np.lexsort((np.arange(n), area))] = np.arange(n)
        self.ymin, self.ymax = xyxy[:, 1], xyxy[:, 3]

        # 再判定: 残す候補の最大スコアとその最終ラベル、使わない候補の最大スコア
        self.has_scores = np.zeros(n, dtype=bool)
        self.keep_score = np.full(n, -np.inf)
        self.none_score = np.full(n, -np.inf)
        self.keep_label = self.det_label.copy()
        for i, b in enumerate(boxes):
            scores = b.get("reclassification_scores")
            mapping = candidates.get(b["detection_label"] or b["label"])
            if not scores or not mapping:
                continue
            self.has_scores[i] = True
            for cand, value in scores.items():
                final = mapping.get(cand)
                if final is None:
                    self.none_score[i] = max(self.none_score[i], value)
                elif value > self.keep_score[i]:
                    self.keep_score[i] = value
                    self.keep_label[i] = _label_id(labels, final)

        # GT
        gts = frame["gts"]
        self.gt_label = np.array([labels.get(g["label"], -1) for g in gts], dtype=int)
        self.gt_ignored = np.array([
            (g["visibility"] is not None and g["visibility"] < args.min_visibility)
            or nusc_category_to_label(g["category_name"]) is None
            or g["label"] not in labels
            for g in gts], dtype=bool)
        # 対応付けの候補（IoU の大きい順）。試行ごとに残った検出だけを使う
        self.pairs: list[tuple[int, int]] = []
        if n and gts:
            g_xyxy = np.array([[g["xmin"], g["ymin"], g["xmax"], g["ymax"]] for g in gts], dtype=float)
            m, _ = _pair_iou(xyxy, g_xyxy)
            ii, jj = np.nonzero(m >= args.match_iou)
            order = np.argsort(-m[ii, jj], kind="stable")
            self.pairs = [(int(ii[k]), int(jj[k])) for k in order]


def _load_frames(paths: list[Path], labels, groups, candidates, args) -> list[Frame]:
    frames = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        frames.extend(Frame(f, labels, groups, candidates, args) for f in data["frames"])
    return frames


# ── 後処理の再現 ─────────────────────────────────────────────────────────

def _greedy_nms(keep: np.ndarray, iou_upper: np.ndarray, threshold: np.ndarray | float,
                restrict: np.ndarray | None, inclusive: bool) -> None:
    """スコアの高い順に残し、重なる後続を落とす（keep をその場で書き換える）."""
    for i in range(len(keep)):
        if not keep[i]:
            continue
        t = threshold[i] if isinstance(threshold, np.ndarray) else threshold
        hit = iou_upper[i] >= t if inclusive else iou_upper[i] > t
        if restrict is not None:
            hit &= restrict[i]
        keep &= ~hit


def _ios_suppress(f: Frame, alive: np.ndarray, label: np.ndarray, threshold: float,
                  direction_by_label: dict[int, str], coverage: float) -> np.ndarray:
    """postprocess.suppress_contained_boxes と同じ判定で、削除する検出の印を返す."""
    deleted = np.zeros(f.n, dtype=bool)
    if threshold <= 0:
        return deleted
    for lab, direction in direction_by_label.items():
        idx = np.nonzero(alive & (label == lab))[0]
        if len(idx) < 2:
            continue
        sub = f.ios[np.ix_(idx, idx)] >= threshold
        if direction == "multi_big":
            for a, i in enumerate(idx):
                children = sub[a] & (f.rank[idx] < f.rank[i])
                if children.sum() < MULTI_BIG_MIN_CHILDREN:
                    continue
                c = idx[children]
                top, bottom = f.ymin[i], f.ymax[i]
                covered = min(bottom, f.ymax[c].max()) - max(top, f.ymin[c].min())
                if covered > (bottom - top) * coverage:
                    deleted[i] = True
            continue
        aa, bb = np.nonzero(np.triu(sub, 1))
        for a, b in zip(aa, bb):
            i, j = idx[a], idx[b]
            big, small = (i, j) if f.rank[i] > f.rank[j] else (j, i)
            deleted[small if direction == "small" else big] = True
    return deleted


def _evaluate(frames: list[Frame], p: dict, ctx: dict) -> tuple[Counter, Counter, Counter]:
    """後処理をかけ直し、ラベルごとの (tp, fp, fn) を数える."""
    tp, fp, fn = Counter(), Counter(), Counter()
    thr = ctx["group_array"](p["score"])
    nms_same = ctx["group_array"](p["nms_same"])
    for f in frames:
        if f.n:
            keep = f.score > thr[f.group]
            _greedy_nms(keep, f.iou_upper, nms_same[f.group], f.same_label, inclusive=False)
            _greedy_nms(keep, f.iou_upper, p["nms_cross"], f.same_group, inclusive=False)
            _greedy_nms(keep, f.iou_upper, p["nms_cross"], None, inclusive=True)
            removed = f.has_scores & (f.none_score > f.keep_score + p["margin"])
            label = np.where(f.has_scores, f.keep_label, f.det_label)
            alive = keep & ~removed
            alive &= ~_ios_suppress(f, alive, label, p["ios"], ctx["ios_direction"],
                                    ctx["ios_coverage"])
        else:
            alive, label = np.zeros(0, dtype=bool), np.zeros(0, dtype=int)
        used_d, used_g = set(), set()
        for i, j in f.pairs:
            if not alive[i] or i in used_d or j in used_g:
                continue
            used_d.add(i)
            used_g.add(j)
            if f.gt_ignored[j]:
                continue
            if label[i] == f.gt_label[j]:
                tp[label[i]] += 1
            else:
                fp[label[i]] += 1
                fn[f.gt_label[j]] += 1
        for i in np.nonzero(alive)[0]:
            if i not in used_d:
                fp[label[i]] += 1
        for j in range(len(f.gt_label)):
            if j not in used_g and not f.gt_ignored[j]:
                fn[f.gt_label[j]] += 1
    return tp, fp, fn


def _scores(tp: Counter, fp: Counter, fn: Counter, label_ids: list[int], beta: float,
            min_precision: float = 0.0) -> dict:
    """ラベルごとと全体の precision / recall / Fβ と、目的関数（label_ids の Fβ の平均）.

    全体の precision が min_precision を下回ると、下回った分を目的関数から引く。
    """
    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        b2 = beta * beta
        fb = (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0
        return p, r, fb
    per_label = {lab: prf(tp[lab], fp[lab], fn[lab]) for lab in set(tp) | set(fp) | set(fn)}
    objective = float(np.mean([per_label.get(lab, (0, 0, 0))[2] for lab in label_ids]))
    total = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    objective -= max(0.0, min_precision - total[0])
    return {"objective": objective, "per_label": per_label, "total": total}


# ── 探索 ────────────────────────────────────────────────────────────────

def _current_params(settings, group_names: list[str]) -> dict:
    return {
        "score": {g: settings.DET2D_DEFAULT_SCORE_THRESHOLDS[g] for g in group_names},
        "nms_same": {g: settings.DET2D_NMS_SAME_CLASS_IOUS[g] for g in group_names},
        "nms_cross": settings.DET2D_NMS_CROSS_CLASS_IOU,
        "ios": settings.DET2D_IOS_DELETE_THRESHOLD,
        "margin": 0.0,
    }


def _suggest(trial, space: str, group_names: list[str], current: dict) -> dict:
    p = {"score": {g: trial.suggest_float(f"score_{g}", *SCORE_RANGE, step=STEP) for g in group_names}}
    if space == "thresholds":
        return {**current, **p}
    p["nms_same"] = {g: trial.suggest_float(f"nms_same_{g}", *NMS_SAME_RANGE, step=STEP)
                     for g in group_names}
    p["nms_cross"] = trial.suggest_float("nms_cross", *NMS_CROSS_RANGE, step=STEP)
    p["ios"] = trial.suggest_float("ios", *IOS_RANGE, step=STEP)
    p["margin"] = trial.suggest_float("margin", *MARGIN_RANGE, step=STEP)
    return p


def _flatten(p: dict, space: str, group_names: list[str]) -> dict:
    """Optuna の trial パラメータの形（enqueue_trial 用）."""
    flat = {f"score_{g}": p["score"][g] for g in group_names}
    if space == "all":
        flat.update({f"nms_same_{g}": p["nms_same"][g] for g in group_names})
        flat.update({"nms_cross": p["nms_cross"], "ios": p["ios"], "margin": p["margin"]})
    return flat


def _check_ranges(meta: dict, current: dict, group_names: list[str]) -> None:
    if SCORE_RANGE[0] < meta["min_score"]:
        sys.exit(f"スコアの探索範囲の下限 {SCORE_RANGE[0]} がキャッシュの min_score {meta['min_score']} より低い")
    if NMS_SAME_RANGE[1] > meta["nms_same_max"] or NMS_CROSS_RANGE[1] > meta["nms_cross_max"]:
        sys.exit("NMS の探索範囲の上限がキャッシュの NMS より緩い")
    for g in group_names:
        if not SCORE_RANGE[0] <= current["score"][g] <= SCORE_RANGE[1]:
            sys.exit(f"現行のスコア閾値 {g}={current['score'][g]} が探索範囲の外")


def _print_compare(title: str, rows: list[tuple[str, dict]], label_names: list[str],
                   label_ids: list[int], gt_count: Counter) -> None:
    _log(title)
    head = f"  {'':<22}" + "".join(f"{name:>26}" for name, _ in rows)
    print(head)
    print(f"  {'':<22}" + "".join(f"{'P':>8}{'R':>8}{'Fβ':>10}" for _ in rows))
    for lab in sorted(set(gt_count) | {l for _, s in rows for l in s["per_label"]}):
        if lab < 0:
            continue
        mark = "*" if lab in label_ids else " "
        line = f"  {mark}{label_names[lab]:<16}{gt_count[lab]:>5}"
        for _, s in rows:
            p, r, fb = s["per_label"].get(lab, (0.0, 0.0, 0.0))
            line += f"{p:>8.3f}{r:>8.3f}{fb:>10.3f}"
        print(line)
    line = f"  {'全体（micro）':<20}{sum(gt_count.values()):>5}"
    for _, s in rows:
        p, r, fb = s["total"]
        line += f"{p:>8.3f}{r:>8.3f}{fb:>10.3f}"
    print(line)
    print(f"  {'目的関数（*の平均）':<19}" + "".join(f"{s['objective']:>26.4f}" for _, s in rows))
    print("  （* は目的関数に入れたラベル。数字は GT 数）", flush=True)


def main() -> None:
    args = _parse_args()
    try:
        import optuna
    except ImportError:
        sys.exit("optuna がありません。webapp のイメージを作り直してください"
                 "（docker compose build webapp）")
    settings = get_settings()
    cache_dir = settings.DERIVED_ROOT / "det2d_cache" / args.name
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        sys.exit(f"キャッシュがありません: {cache_dir}（先に tools/build_det2d_cache.py を実行）")
    meta = json.loads(meta_path.read_text())

    order = meta.get("scene_order") or sorted(p.name.split(".")[0] for p in cache_dir.glob("*.json.gz"))
    scenes = [t for t in order if (cache_dir / f"{t}.json.gz").exists()]
    if args.max_scenes is not None:
        if len(scenes) < args.max_scenes:
            sys.exit(f"キャッシュ済みのシーンが {len(scenes)} 件しかありません（--max-scenes {args.max_scenes}）")
        scenes = scenes[:args.max_scenes]
    shuffled = scenes[:]
    random.Random(args.seed).shuffle(shuffled)
    num_holdout = int(round(len(scenes) * args.holdout))
    holdout, tune = shuffled[:num_holdout], shuffled[num_holdout:]
    if not tune:
        sys.exit("探索に使うシーンがありません（--holdout を小さくしてください）")

    group_names = list(label_groups())
    label_names = [lab for g in group_names for lab in label_groups()[g]]
    labels = {name: i for i, name in enumerate(label_names)}
    groups = {name: i for i, name in enumerate(group_names)}
    current = _current_params(settings, group_names)
    _check_ranges(meta, current, group_names)
    ctx = {
        "group_array": lambda d: np.array([d[g] for g in group_names]),
        "ios_direction": {labels[k]: v for k, v in settings.DET2D_IOS_DELETE_DIRECTION.items()
                          if k in labels},
        "ios_coverage": settings.DET2D_IOS_MULTI_BIG_VERTICAL_COVERAGE,
    }
    candidates = meta["re_classification_candidates"]

    started = time.time()
    _log(f"キャッシュ: {cache_dir}（探索 {len(tune)} シーン / 確認 {len(holdout)} シーン）を読み込み中")
    tune_frames = _load_frames([cache_dir / f"{t}.json.gz" for t in tune], labels, groups, candidates, args)
    holdout_frames = _load_frames([cache_dir / f"{t}.json.gz" for t in holdout], labels, groups,
                                  candidates, args)
    gt_tune = Counter(int(l) for f in tune_frames for l, ig in zip(f.gt_label, f.gt_ignored) if not ig)
    gt_holdout = Counter(int(l) for f in holdout_frames for l, ig in zip(f.gt_label, f.gt_ignored) if not ig)
    label_ids = sorted(l for l, c in gt_tune.items() if c >= args.min_label_gt and l >= 0)
    if not label_ids:
        sys.exit(f"GT が {args.min_label_gt} 個以上のラベルがありません")
    label_names = list(labels)  # 読み込みで増えたラベルも含める
    _log(f"読み込み完了 {_fmt(time.time() - started)}: 探索 {len(tune_frames)} 枚 GT {sum(gt_tune.values())} 個"
         f" / 確認 {len(holdout_frames)} 枚 GT {sum(gt_holdout.values())} 個")
    _log(f"目的関数: Fβ(β={args.beta}) の平均（{', '.join(label_names[l] for l in label_ids)}）"
         + (f"、全体の precision が {args.min_precision} 未満なら減点" if args.min_precision else ""))

    t0 = time.time()
    baseline = _scores(*_evaluate(tune_frames, current, ctx), label_ids, args.beta, args.min_precision)
    per_trial = time.time() - t0
    _log(f"現行の設定: 目的関数 {baseline['objective']:.4f}"
         f"（P {baseline['total'][0]:.3f} / R {baseline['total'][1]:.3f}）  1 試行 {per_trial:.2f} 秒")

    study_name = args.study or (f"s{len(scenes)}_h{args.holdout}_b{args.beta}_p{args.min_precision}_{args.space}"
                                f"_seed{args.seed}_vis{args.min_visibility}_iou{args.match_iou}"
                                f"_min{args.min_label_gt}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True,
                                         n_startup_trials=args.startup_trials)
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler,
                                storage=f"sqlite:///{cache_dir / 'optuna.db'}", load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, args.trials - done)
    if done == 0:
        # 1 試行目は現行の設定（比較の基準になる）
        study.enqueue_trial(_flatten(current, args.space, group_names))
    _log(f"study {study_name}: 完了 {done} 試行、今回 {remaining} 試行"
         f"（見込み {_fmt(remaining * per_trial)}）")

    state = {"last_log": time.time(), "start": time.time(), "count": 0,
             "best": study.best_value if done else None}

    def objective(trial):
        p = _suggest(trial, args.space, group_names, current)
        s = _scores(*_evaluate(tune_frames, p, ctx), label_ids, args.beta, args.min_precision)
        trial.set_user_attr("precision", s["total"][0])
        trial.set_user_attr("recall", s["total"][1])
        return s["objective"]

    def callback(study, trial):
        state["count"] += 1
        now = time.time()
        n_done = done + state["count"]
        if state["best"] is None or (trial.value is not None and trial.value > state["best"]):
            state["best"] = trial.value
            _log(f"  ★ 試行 {trial.number}: 最良を更新 {trial.value:.4f}"
                 f"（現行比 {trial.value - baseline['objective']:+.4f}、"
                 f"P {trial.user_attrs['precision']:.3f} / R {trial.user_attrs['recall']:.3f}）")
        if now - state["last_log"] >= args.log_interval or state["count"] == remaining:
            state["last_log"] = now
            elapsed = now - state["start"]
            left = elapsed / state["count"] * (remaining - state["count"])
            _log(f"  試行 {n_done}/{args.trials} ({n_done / args.trials:4.0%})"
                 f"  最良 {study.best_value:.4f}（試行 {study.best_trial.number}）"
                 f"  経過 {_fmt(elapsed)}  残り約 {_fmt(left)}")

    if remaining:
        try:
            study.optimize(objective, n_trials=remaining, callbacks=[callback])
        except KeyboardInterrupt:
            _log("中断しました。同じオプションで実行すると続きから再開します")
            return

    best_flat = study.best_params
    best = {**current, "score": {g: best_flat[f"score_{g}"] for g in group_names}}
    if args.space == "all":
        best.update({
            "nms_same": {g: best_flat[f"nms_same_{g}"] for g in group_names},
            "nms_cross": best_flat["nms_cross"], "ios": best_flat["ios"],
            "margin": best_flat["margin"]})

    rows = [("現行", baseline), ("最良", _scores(*_evaluate(tune_frames, best, ctx), label_ids, args.beta, args.min_precision))]
    _print_compare(f"探索用 {len(tune)} シーンでの比較", rows, label_names, label_ids, gt_tune)
    result = {"study": study_name, "trials": len(study.trials), "best_trial": study.best_trial.number,
              "params": best, "current": current,
              "tune": {"current": baseline["objective"], "best": rows[1][1]["objective"]}}
    if holdout_frames:
        # 確認用のシーンに GT が無いラベルは、Fβ が 0 になって平均を下げるだけなので外す
        hold_ids = [lab for lab in label_ids if gt_holdout[lab]]
        hold_rows = [("現行", _scores(*_evaluate(holdout_frames, current, ctx), hold_ids, args.beta, args.min_precision)),
                     ("最良", _scores(*_evaluate(holdout_frames, best, ctx), hold_ids, args.beta, args.min_precision))]
        _print_compare(f"確認用 {len(holdout)} シーンでの比較（探索に使っていないシーン）",
                       hold_rows, label_names, hold_ids, gt_holdout)
        result["holdout"] = {"current": hold_rows[0][1]["objective"], "best": hold_rows[1][1]["objective"]}

    _log("最良の設定（config.py の形）:")
    r2 = lambda v: round(float(v), 2)  # noqa: E731
    score = {g: r2(best["score"][g]) for g in group_names}
    nms_same = {g: r2(best["nms_same"][g]) for g in group_names}
    print(f"    DET2D_DEFAULT_SCORE_THRESHOLDS = {score}")
    print(f"    DET2D_NMS_SAME_CLASS_IOUS = {nms_same}")
    print(f"    DET2D_NMS_CROSS_CLASS_IOU = {r2(best['nms_cross'])}")
    print(f"    DET2D_IOS_DELETE_THRESHOLD = {r2(best['ios'])}")
    print(f"    再判定の削除マージン（未実装） = {r2(best['margin'])}", flush=True)
    out = cache_dir / f"tune_{study_name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    _log(f"結果を保存しました: {out}（経過 {_fmt(time.time() - started)}）")


main()
