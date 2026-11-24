"""Detection2D の後処理パラメータを、キャッシュ（tools/build_det2d_cache.py）を使って Optuna で探索する.

推論はやり直さず、キャッシュした検出に後処理をかけ直して GT と比べる。1 試行は数秒で済む。

## 探索するパラメータ（--space all のとき）

  - スコア閾値（カテゴリグループごと）  DET2D_DEFAULT_SCORE_THRESHOLDS
  - 同一クラス NMS の IoU（グループごと） DET2D_NMS_SAME_CLASS_IOUS
  - クラス間 NMS の IoU                  DET2D_NMS_CROSS_CLASS_IOU
  - IoS 結合の閾値                        DET2D_IOS_DELETE_THRESHOLD
  - 再判定の削除マージン                  DEFAULT_RECLASSIFICATION_DELETE_MARGIN
    「使わない」候補（RE_CLASSIFICATION_CANDIDATES の値が None）の最大スコアが、残す候補の
    最大スコアをマージンより大きく上回ったときだけ削除する。0 ならスコア最大の候補で決める

--space thresholds ならスコア閾値だけを探し、残りは config の値に固定する。

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
  --split-seed N       探索用と確認用にシーンを分けるときの乱数の種（既定 0）
  --seeds N [N ...]    Optuna（TPE）の乱数の種。複数なら順に探索してまとめを出す
                       （既定 0。0-9 のような範囲も書ける）
  --stable-spread R    まとめの判定: シード間の最良値の範囲がパラメータの探索範囲の幅の
                       この割合以下なら「安定」（既定 0.15）
  --flat-tol D         まとめの判定: 目的関数の変化がこれ未満なら差なしとみなす（既定 0.003）
  --study NAME         Optuna の study 名（既定はオプションから自動で決める。シードが複数なら
                       末尾に _seed{N} を付ける）
  --log-interval SEC   進捗を表示する間隔（秒。既定 30）

## 複数シードのまとめ（--seeds に 2 個以上）

シードごとの結果を今まで通り出したあと、次をまとめて表示し、tune_..._split{S}_seeds{...}.json に保存する。

  - シードごとの最良と、現行比の改善幅（探索用・確認用）の平均±標準偏差
  - 合意の設定: パラメータごとに、各シードの最良の中央値をとった設定
  - パラメータごとの、シード間のばらつきと効き方（合意の設定を基準に 1 つずつ動かす）
      ばらつき       シード間の範囲 ÷ 探索範囲の幅
      範囲内の影響   シード間の最小値・最大値にしたときの、探索用の目的関数の低下
      現行に戻す     そのパラメータだけ現行の値に戻したときの目的関数の変化（探索用・確認用）
    判定は「安定」（ばらつきが小さい）/「平坦」（ばらつくが目的関数は変わらない。どの値でもよい）/
    「不安定」（ばらつき、値で結果も変わる）。現行に戻しても差が無ければ「現行と差なし」を付ける
  - 合意の設定と現行の設定の比較（探索用・確認用）

例:
    docker compose run --rm --no-deps -T webapp python - --max-scenes 10 < tools/tune_det2d_thresholds.py
    docker compose run --rm --no-deps -T webapp python - --max-scenes 30 --seeds 0-4 < tools/tune_det2d_thresholds.py
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
from types import SimpleNamespace

import numpy as np

from app.core.config import get_settings
from app.services.label_service import label_groups, nusc_category_to_label

# 探索範囲（0.01 刻み）。変えたら SPACE_VERSION を上げる（範囲の違う試行を同じ study に混ぜない）
SCORE_RANGE = (0.10, 0.50)
NMS_SAME_RANGE = (0.40, 0.90)
NMS_CROSS_RANGE = (0.30, 0.95)  # v1 は 0.50〜。最良が下限に張り付いたので広げた
IOS_RANGE = (0.60, 1.00)
MARGIN_RANGE = (0.00, 0.80)  # v1 は 〜0.30。最良が上限に張り付いたので広げた
STEP = 0.01
SPACE_VERSION = 2

MULTI_BIG_MIN_CHILDREN = 2  # postprocess.MULTI_BIG_MIN_CHILDREN と同じ


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tune_det2d_thresholds", allow_abbrev=False)
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
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", default=["0"])
    parser.add_argument("--stable-spread", type=float, default=0.15)
    parser.add_argument("--flat-tol", type=float, default=0.003)
    parser.add_argument("--study", default=None)
    parser.add_argument("--log-interval", type=float, default=30.0)
    args = parser.parse_args()
    seeds: list[int] = []
    for token in args.seeds:  # "0-9" は 0 から 9 まで
        try:
            if "-" in token:
                a, b = token.split("-", 1)
                seeds.extend(range(int(a), int(b) + 1))
            else:
                seeds.append(int(token))
        except ValueError:
            parser.error(f"--seeds の値が読めません: {token}（例: 0 1 2 / 0-9）")
    args.seeds = list(dict.fromkeys(seeds))
    if not args.seeds:
        parser.error("--seeds が空です")
    return args


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
        "margin": settings.DEFAULT_RECLASSIFICATION_DELETE_MARGIN,
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


def _param_range(name: str) -> tuple[float, float]:
    if name.startswith("score_"):
        return SCORE_RANGE
    if name.startswith("nms_same_"):
        return NMS_SAME_RANGE
    return {"nms_cross": NMS_CROSS_RANGE, "ios": IOS_RANGE, "margin": MARGIN_RANGE}[name]


def _unflatten(flat: dict, space: str, group_names: list[str], current: dict) -> dict:
    """_flatten の逆。探索しないパラメータは現行の値."""
    p = {**current, "score": {g: flat[f"score_{g}"] for g in group_names}}
    if space == "all":
        p.update({"nms_same": {g: flat[f"nms_same_{g}"] for g in group_names},
                  "nms_cross": flat["nms_cross"], "ios": flat["ios"], "margin": flat["margin"]})
    return p


def _print_config(p: dict, group_names: list[str]) -> None:
    r2 = lambda v: round(float(v), 2)  # noqa: E731
    print(f"    DET2D_DEFAULT_SCORE_THRESHOLDS = { {g: r2(p['score'][g]) for g in group_names} }")
    print(f"    DET2D_NMS_SAME_CLASS_IOUS = { {g: r2(p['nms_same'][g]) for g in group_names} }")
    print(f"    DET2D_NMS_CROSS_CLASS_IOU = {r2(p['nms_cross'])}")
    print(f"    DET2D_IOS_DELETE_THRESHOLD = {r2(p['ios'])}")
    print(f"    DEFAULT_RECLASSIFICATION_DELETE_MARGIN = {r2(p['margin'])}", flush=True)


def _seed_spec(seeds: list[int]) -> str:
    """ファイル名用のシード一覧（連番なら 0-9 の形）."""
    if seeds == list(range(seeds[0], seeds[0] + len(seeds))):
        return f"{seeds[0]}-{seeds[-1]}"
    return ".".join(map(str, seeds))


def _study_name(args: argparse.Namespace, n_scenes: int, seed_part: str) -> str:
    return (f"s{n_scenes}_h{args.holdout}_b{args.beta}_p{args.min_precision}_{args.space}"
            f"_{seed_part}_vis{args.min_visibility}_iou{args.match_iou}_min{args.min_label_gt}"
            f"_v{SPACE_VERSION}")


def _run_seed(optuna, env: SimpleNamespace, seed: int) -> dict | None:
    """1 つの Optuna シードで探索し、結果を表示・保存して返す（中断したら None）."""
    args = env.args
    if args.study:
        study_name = args.study if len(args.seeds) == 1 else f"{args.study}_seed{seed}"
    else:
        # シーンの分け方と Optuna のシードが同じなら、分ける前（--seed 1 つ）の名前のまま（再開できる）
        part = f"seed{seed}" if seed == args.split_seed else f"split{args.split_seed}_seed{seed}"
        study_name = _study_name(args, env.n_scenes, part)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                         n_startup_trials=args.startup_trials)
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler,
                                storage=f"sqlite:///{env.cache_dir / 'optuna.db'}", load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, args.trials - done)
    if done == 0:
        # 1 試行目は現行の設定（比較の基準になる）
        study.enqueue_trial(_flatten(env.current, args.space, env.group_names))
    _log(f"study {study_name}: 完了 {done} 試行、今回 {remaining} 試行"
         f"（見込み {_fmt(remaining * env.per_trial)}）")

    base = env.baseline["objective"]
    state = {"last_log": time.time(), "start": time.time(), "count": 0,
             "best": study.best_value if done else None}

    def objective(trial):
        p = _suggest(trial, args.space, env.group_names, env.current)
        s = env.score_tune(p)
        trial.set_user_attr("precision", s["total"][0])
        trial.set_user_attr("recall", s["total"][1])
        return s["objective"]

    def callback(study, trial):
        state["count"] += 1
        now = time.time()
        n_done = done + state["count"]
        if state["best"] is None or (trial.value is not None and trial.value > state["best"]):
            state["best"] = trial.value
            _log(f"  ★ 試行 {trial.number}: 最良を更新 {trial.value:.4f}（現行比 {trial.value - base:+.4f}、"
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
            return None

    flat = study.best_params
    best = _unflatten(flat, args.space, env.group_names, env.current)
    best_tune = env.score_tune(best)
    _print_compare(f"探索用 {len(env.tune)} シーンでの比較", [("現行", env.baseline), ("最良", best_tune)],
                   env.label_names, env.label_ids, env.gt_tune)
    result = {"study": study_name, "seed": seed, "trials": len(study.trials),
              "best_trial": study.best_trial.number, "flat": flat, "params": best, "current": env.current,
              "tune": {"current": base, "best": best_tune["objective"],
                       "precision": best_tune["total"][0], "recall": best_tune["total"][1]}}
    if env.holdout_frames:
        best_hold = env.score_hold(best)
        _print_compare(f"確認用 {len(env.holdout)} シーンでの比較（探索に使っていないシーン）",
                       [("現行", env.baseline_hold), ("最良", best_hold)],
                       env.label_names, env.hold_ids, env.gt_holdout)
        result["holdout"] = {"current": env.baseline_hold["objective"], "best": best_hold["objective"],
                             "precision": best_hold["total"][0], "recall": best_hold["total"][1]}
    _log("最良の設定（config.py の形）:")
    _print_config(best, env.group_names)
    out = env.cache_dir / f"tune_{study_name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    _log(f"結果を保存しました: {out}")
    return result


def _summarize(env: SimpleNamespace, results: list[dict]) -> None:
    """複数シードの結果をまとめる: 改善幅の分布、パラメータごとのばらつきと効き方、合意の設定."""
    args = env.args
    has_hold = bool(env.holdout_frames)
    seeds = [r["seed"] for r in results]
    print()
    _log(f"===== {len(results)} シード（{', '.join(map(str, seeds))}）のまとめ =====")

    # 1. シードごとの改善幅
    _log("シードごとの最良（Δ は現行比）")
    head = f"  {'seed':>6}{'最良試行':>8}{'探索 目的関数':>14}{'Δ':>9}"
    if has_hold:
        head += f"{'確認 目的関数':>14}{'Δ':>9}{'確認 P':>8}{'確認 R':>8}"
    print(head)
    for r in results:
        line = (f"  {r['seed']:>6}{r['best_trial']:>12}{r['tune']['best']:>16.4f}"
                f"{r['tune']['best'] - r['tune']['current']:>+9.4f}")
        if has_hold:
            h = r["holdout"]
            line += (f"{h['best']:>16.4f}{h['best'] - h['current']:>+9.4f}"
                     f"{h['precision']:>8.3f}{h['recall']:>8.3f}")
        print(line)
    d_tune = np.array([r["tune"]["best"] - r["tune"]["current"] for r in results])
    line = f"  {'平均±標準偏差':<14}{d_tune.mean():>+20.4f}±{d_tune.std(ddof=1):.4f}"
    if has_hold:
        d_hold = np.array([r["holdout"]["best"] - r["holdout"]["current"] for r in results])
        line += f"{d_hold.mean():>+16.4f}±{d_hold.std(ddof=1):.4f}"
    print(line)
    best_values = np.array([r["tune"]["best"] for r in results])
    print(f"  探索用の最良値の幅: {best_values.min():.4f}〜{best_values.max():.4f}"
          f"（差 {np.ptp(best_values):.4f}。小さいのに設定がばらつくなら、目的関数の山が平らということ）")
    if has_hold:
        print(f"  確認用で現行より良くなったシード: {int((d_hold > 0).sum())}/{len(results)}", flush=True)

    # 2. 合意の設定（シードごとの最良の中央値）
    names = list(results[0]["flat"])
    values = {n: np.array([r["flat"][n] for r in results]) for n in names}
    cons_flat = {n: round(round(float(np.median(v)) / STEP) * STEP, 2) for n, v in values.items()}
    cons = _unflatten(cons_flat, args.space, env.group_names, env.current)
    cons_tune = env.score_tune(cons)
    cons_hold = env.score_hold(cons) if has_hold else None
    cur_flat = _flatten(env.current, args.space, env.group_names)

    # 3. パラメータごとのばらつきと効き方（合意の設定を基準に 1 つずつ動かす）
    _log("パラメータごとのばらつきと効き方（基準は合意の設定 = 各シードの最良の中央値）")
    print("  ばらつき: シード間の範囲 ÷ 探索範囲の幅")
    print("  範囲内の影響: そのパラメータだけシード間の最小値・最大値にしたとき、探索用の目的関数が最も下がる量")
    print("  現行に戻す: そのパラメータだけ現行の値に戻したときの目的関数の変化（マイナスならそのパラメータの変更が効いている）")
    head = (f"  {'パラメータ':<22}{'現行':>6}{'中央値':>7}{'平均±標準偏差':>15}{'最小〜最大':>13}"
            f"{'ばらつき':>7}{'範囲内の影響':>10}{'現行に戻す 探索':>12}")
    if has_hold:
        head += f"{'確認':>8}"
    print(head + f"  {'判定'}")
    param_stats = {}
    for n in names:
        v = values[n]
        lo, hi = _param_range(n)
        spread = float(np.ptp(v)) / (hi - lo)
        drops = []
        for x in (float(v.min()), float(v.max())):
            p = _unflatten({**cons_flat, n: x}, args.space, env.group_names, env.current)
            drops.append(cons_tune["objective"] - env.score_tune(p)["objective"])
        spread_drop = max(0.0, max(drops))
        reverted = _unflatten({**cons_flat, n: cur_flat[n]}, args.space, env.group_names, env.current)
        rev_tune = env.score_tune(reverted)["objective"] - cons_tune["objective"]
        rev_hold = (env.score_hold(reverted)["objective"] - cons_hold["objective"]) if has_hold else None
        if spread <= args.stable_spread:
            verdict = "安定"
        elif spread_drop < args.flat_tol:
            verdict = "平坦（どの値でも同じ）"
        else:
            verdict = "不安定（値で結果が変わる）"
        if abs(rev_tune) < args.flat_tol:
            verdict += "・現行と差なし"
        elif has_hold and rev_tune < 0 and rev_hold > -args.flat_tol:
            verdict += "・確認用では効かない"
        line = (f"  {n:<22}{cur_flat[n]:>6.2f}{cons_flat[n]:>7.2f}{v.mean():>9.2f}±{v.std(ddof=1):.2f}"
                f"{v.min():>8.2f}〜{v.max():.2f}{spread:>8.0%}{-spread_drop:>+12.4f}{rev_tune:>+14.4f}")
        if has_hold:
            line += f"{rev_hold:>+10.4f}"
        print(line + f"  {verdict}")
        param_stats[n] = {"current": cur_flat[n], "median": cons_flat[n], "mean": float(v.mean()),
                          "std": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
                          "spread": spread, "spread_drop": spread_drop, "revert_tune": rev_tune,
                          "revert_holdout": rev_hold, "verdict": verdict}
    print(f"  判定: ばらつき {args.stable_spread:.0%} 以下 = 安定 / それ以外で範囲内の影響が {args.flat_tol} 未満 = 平坦"
          f" / それ以外 = 不安定。現行に戻した差が {args.flat_tol} 未満なら「現行と差なし」、"
          f"探索用では効くが確認用で効かなければ「確認用では効かない」", flush=True)

    # 4. 合意の設定の評価
    _print_compare(f"探索用 {len(env.tune)} シーンでの比較（合意の設定）",
                   [("現行", env.baseline), ("合意", cons_tune)], env.label_names, env.label_ids, env.gt_tune)
    summary = {"seeds": seeds, "split_seed": args.split_seed, "per_seed": results, "params": param_stats,
               "consensus": {"params": cons, "tune": cons_tune["objective"]}}
    if has_hold:
        _print_compare(f"確認用 {len(env.holdout)} シーンでの比較（合意の設定）",
                       [("現行", env.baseline_hold), ("合意", cons_hold)],
                       env.label_names, env.hold_ids, env.gt_holdout)
        mean_best = float(np.mean([r["holdout"]["best"] for r in results]))
        _log(f"確認用: 合意の設定 {cons_hold['objective']:.4f} / 各シードの最良の平均 {mean_best:.4f}"
             f" / 現行 {env.baseline_hold['objective']:.4f}")
        summary["consensus"]["holdout"] = cons_hold["objective"]
    _log("合意の設定（config.py の形）:")
    _print_config(cons, env.group_names)
    out = env.cache_dir / f"tune_{_study_name(args, env.n_scenes, f'split{args.split_seed}_seeds{_seed_spec(seeds)}')}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    _log(f"まとめを保存しました: {out}")


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
    random.Random(args.split_seed).shuffle(shuffled)
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
    _log(f"キャッシュ: {cache_dir}（探索 {len(tune)} シーン / 確認 {len(holdout)} シーン、"
         f"分け方のシード {args.split_seed}）を読み込み中")
    tune_frames = _load_frames([cache_dir / f"{t}.json.gz" for t in tune], labels, groups, candidates, args)
    holdout_frames = _load_frames([cache_dir / f"{t}.json.gz" for t in holdout], labels, groups,
                                  candidates, args)
    gt_tune = Counter(int(l) for f in tune_frames for l, ig in zip(f.gt_label, f.gt_ignored) if not ig)
    gt_holdout = Counter(int(l) for f in holdout_frames for l, ig in zip(f.gt_label, f.gt_ignored) if not ig)
    label_ids = sorted(l for l, c in gt_tune.items() if c >= args.min_label_gt and l >= 0)
    if not label_ids:
        sys.exit(f"GT が {args.min_label_gt} 個以上のラベルがありません")
    # 確認用のシーンに GT が無いラベルは、Fβ が 0 になって平均を下げるだけなので外す
    hold_ids = [lab for lab in label_ids if gt_holdout[lab]]
    label_names = list(labels)  # 読み込みで増えたラベルも含める
    _log(f"読み込み完了 {_fmt(time.time() - started)}: 探索 {len(tune_frames)} 枚 GT {sum(gt_tune.values())} 個"
         f" / 確認 {len(holdout_frames)} 枚 GT {sum(gt_holdout.values())} 個")
    _log(f"目的関数: Fβ(β={args.beta}) の平均（{', '.join(label_names[l] for l in label_ids)}）"
         + (f"、全体の precision が {args.min_precision} 未満なら減点" if args.min_precision else ""))

    def score_tune(p):
        return _scores(*_evaluate(tune_frames, p, ctx), label_ids, args.beta, args.min_precision)

    def score_hold(p):
        return _scores(*_evaluate(holdout_frames, p, ctx), hold_ids, args.beta, args.min_precision)

    t0 = time.time()
    baseline = score_tune(current)
    per_trial = time.time() - t0
    _log(f"現行の設定: 目的関数 {baseline['objective']:.4f}"
         f"（P {baseline['total'][0]:.3f} / R {baseline['total'][1]:.3f}）  1 試行 {per_trial:.2f} 秒")
    env = SimpleNamespace(
        args=args, cache_dir=cache_dir, n_scenes=len(scenes), tune=tune, holdout=holdout,
        tune_frames=tune_frames, holdout_frames=holdout_frames, gt_tune=gt_tune, gt_holdout=gt_holdout,
        label_ids=label_ids, hold_ids=hold_ids, label_names=label_names, group_names=group_names,
        current=current, baseline=baseline, baseline_hold=score_hold(current) if holdout_frames else None,
        per_trial=per_trial, score_tune=score_tune, score_hold=score_hold)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if len(args.seeds) > 1:
        _log(f"Optuna のシード {len(args.seeds)} 個（{', '.join(map(str, args.seeds))}）で順に探索する"
             f"（全体の見込み 最大 {_fmt(len(args.seeds) * args.trials * per_trial)}）")
    results = []
    for k, seed in enumerate(args.seeds, 1):
        if len(args.seeds) > 1:
            print()
            _log(f"===== Optuna のシード {seed}（{k}/{len(args.seeds)}）=====")
        result = _run_seed(optuna, env, seed)
        if result is None:
            return
        results.append(result)
    if len(results) > 1:
        _summarize(env, results)
    _log(f"完了（経過 {_fmt(time.time() - started)}）")


main()
