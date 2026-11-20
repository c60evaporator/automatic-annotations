"""Detection2D の検出結果を GT と比べ、誤り（誤検出・見逃し・ラベル違い）を数える.

Detection2D の run の検出（detection_2ds。SigLIP2 の再判定・IoS 結合で論理削除された
ものは除く）を、同じカメラのキーフレーム（run の sample_interval で間引いた、推論した
sample だけ）に投影した GT の 2D ボックスと、ラベルを問わず
IoU で 1 対 1 に対応付ける（IoU の大きい組からの貪欲法、IoU >= --match-iou）。そのうえで:

  - 正解（TP）      : 対応した GT とラベルが同じ
  - ラベル違い      : 対応した GT とラベルが違う（検出ラベルの誤検出、GT ラベルの見逃しにも数える）
  - 誤検出（FP）    : どの GT とも対応しない。次の 3 つに分ける
        どの GT とも重ならない（IoU < 0.1）: 誤検出か、GT の無い物体（nuScenes は LiDAR 点の
                                              無い遠方の物体などには GT が無い）
        部分的に重なる（IoU 0.1〜match-iou）: 位置ずれ、隠れ・はみ出しで 3D GT の投影と大きさが違う等
        重なる GT を別の検出が先に取った    : 同じ物体の重複検出（NMS / IoS 結合で落ちなかった）
  - 見逃し（FN）    : どの検出とも対応しない GT

visibility が --min-visibility 未満の GT（ほとんど隠れている物体）と、検出ラベルに対応しない
カテゴリ（movable_object.debris 等。config の NUSC_CATEGORY_TO_LABEL に無いもの）の GT は
「無視」扱いにする。
その GT に対応した検出は正解にも誤検出にも数えず、対応しなくても見逃しに数えない。
論理削除された検出のうち、残った検出が取らなかった GT と対応するもの（削除しすぎ）も数える。

webapp コンテナの中で動かす。使い方（docker-compose.yml のある webapp フォルダで）:

    docker compose run --rm --no-deps -T webapp python - [オプション] < tools/check_det2d_errors.py

  ``--no-deps`` を付けると推論サーバーを起動しない（DB を読むだけなので不要）。

オプション（いずれも省略可）:
  --runs RUN_ID ...    調べる Detection2D の run（前方一致）。省略すると、成功した run を
                       シーンごとに最新の 1 件ずつ使う
  --max-scenes N       対象シーン数の上限（既定: 無制限）。対象 run を持つシーンが上限を
                       超えるときは --order で選ぶ
  --order ORDER        上限を超えるときの選び方。newest（run が新しいシーンから）/
                       random（ランダム。--seed で再現できる）。既定 newest
  --seed N             --order random の乱数の種（既定 0）
  --match-iou IOU      GT と対応付ける IoU の下限（既定 0.3）
  --min-visibility V   この visibility（1: 0〜40%, 2: 40〜60%, 3: 60〜80%, 4: 80〜100%）未満の
                       GT は無視する（既定 2）

例:
    docker compose run --rm --no-deps -T webapp python - --max-scenes 30 --order random < tools/check_det2d_errors.py
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

import numpy as np
from sqlalchemy import select

from app.db.session import read_only_session
from app.models.ann_intermediate import Detection2D, Detection2DParams
from app.repositories.annotation import AnnotationRepository
from app.repositories.sensor import SensorRepository
from app.services.annotation_2d import project_annotations_to_frame
from app.services.label_service import nusc_category_to_label
from app.services.sample_service import get_skipped_sample_indices

NO_OVERLAP_IOU = 0.1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="check_det2d_errors")
    parser.add_argument("--runs", nargs="*", help="Detection2D の run id（前方一致）")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--order", choices=("newest", "random"), default="newest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--min-visibility", type=int, default=2)
    return parser.parse_args()


def _select_runs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """(run id, scene_token, sample_interval) のリスト. シーンごとに最新の成功 run を、新しい順に."""
    with read_only_session() as session:
        rows = session.execute(
            select(Detection2DParams.id, Detection2DParams.scene_token,
                   Detection2DParams.sample_interval)
            .where(Detection2DParams.status == "succeeded")
            .order_by(Detection2DParams.started_at.desc())
        ).all()
    if args.runs:
        rows = [r for r in rows if any(r[0].startswith(p) for p in args.runs)]
    latest: dict[str, tuple[str, int]] = {}
    for run_id, scene, interval in rows:
        latest.setdefault(scene, (run_id, int(interval or 1)))
    runs = [(run_id, scene, interval) for scene, (run_id, interval) in latest.items()]
    if args.max_scenes is not None and len(runs) > args.max_scenes:
        if args.order == "random":
            runs = random.Random(args.seed).sample(runs, args.max_scenes)
        else:
            runs = runs[:args.max_scenes]
    return runs


def _iou_matrix(dets, gts) -> np.ndarray:
    d = np.array([[r.xmin, r.ymin, r.xmax, r.ymax] for r in dets], dtype=float)
    g = np.array([[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in gts], dtype=float)
    ix = np.clip(np.minimum(d[:, None, 2], g[None, :, 2]) - np.maximum(d[:, None, 0], g[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(d[:, None, 3], g[None, :, 3]) - np.maximum(d[:, None, 1], g[None, :, 1]), 0, None)
    inter = ix * iy
    area_d = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = area_d[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _greedy(m: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """IoU の大きい組から 1 対 1 に対応付ける."""
    pairs, used_d, used_g = [], set(), set()
    for flat in np.argsort(-m, axis=None):
        i, j = np.unravel_index(flat, m.shape)
        if m[i, j] < threshold:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        pairs.append((int(i), int(j)))
    return pairs


def check_run(run_id: str, scene: str, interval: int, args: argparse.Namespace,
              by_label: dict[str, Counter], total: Counter) -> Counter:
    with read_only_session() as session:
        dets = session.execute(select(
            Detection2D.sample_data_token, Detection2D.xmin, Detection2D.ymin,
            Detection2D.xmax, Detection2D.ymax, Detection2D.label,
            Detection2D.is_deleted, Detection2D.deleted_by,
        ).where(Detection2D.detection_2d_params_id == run_id)).all()
        camera_frames = [f for f in SensorRepository(session).list_frames_by_scene(
            scene, keyframe_only=True) if f["modality"] == "camera"]
        annotations = AnnotationRepository(session).list_by_scene(
            scene, source="imported", include_attributes=False)
    # 推論した sample だけを対象にする（sample_interval で間引いたキーフレーム。Detection2D の
    # ページと同じ選び方）。推論していないフレームの GT を見逃しに数えないため
    num_samples = len({f["sample_token"] for f in camera_frames})
    inferred = set(get_skipped_sample_indices(num_samples, interval))
    frames = {f["token"]: f for f in camera_frames if f["sample_idx"] in inferred}
    anns_by_sample = defaultdict(list)
    for a in annotations:
        anns_by_sample[a["sample_token"]].append(a)
    visibility = {a["token"]: int(a["visibility_token"]) if str(a.get("visibility_token") or "").isdigit()
                  else 4 for a in annotations}
    dets_by_frame = defaultdict(list)
    for d in dets:
        if d.sample_data_token in frames:  # GT のあるキーフレームだけ
            dets_by_frame[d.sample_data_token].append(d)

    stats = Counter()
    for token, frame in frames.items():
        frame_dets = dets_by_frame.get(token, [])
        kept = [d for d in frame_dets if not d.is_deleted]
        deleted = [d for d in frame_dets if d.is_deleted]
        gts = project_annotations_to_frame(anns_by_sample[frame["sample_token"]], frame)
        # 隠れている GT と、検出ラベルに対応しないカテゴリ（debris 等）の GT は無視する
        ignored = [visibility.get(g["annotation_token"], 4) < args.min_visibility
                   or nusc_category_to_label(g["category_name"]) is None for g in gts]
        stats["frames"] += 1
        stats["detections"] += len(kept)
        stats["gt"] += sum(not x for x in ignored)
        stats["gt_ignored"] += sum(ignored)
        matched_g: set[int] = set()
        if kept and gts:
            m = _iou_matrix(kept, gts)
            pairs = _greedy(m, args.match_iou)
        else:
            m, pairs = np.zeros((len(kept), len(gts))), []
        matched_d = {i for i, _ in pairs}
        for i, j in pairs:
            matched_g.add(j)
            if ignored[j]:
                stats["det_on_ignored_gt"] += 1
                continue
            d, g = kept[i], gts[j]
            if d.label == g["label"]:
                stats["tp"] += 1
                by_label[d.label]["tp"] += 1
            else:
                stats["label_mismatch"] += 1
                by_label[d.label]["fp_label"] += 1
                by_label[g["label"]]["fn_label"] += 1
        for i, d in enumerate(kept):
            if i in matched_d:
                continue
            best = float(m[i].max()) if gts else 0.0
            kind = ("fp_no_overlap" if best < NO_OVERLAP_IOU else
                    "fp_partial" if best < args.match_iou else "fp_duplicate")
            stats[kind] += 1
            by_label[d.label][kind] += 1
        for j, g in enumerate(gts):
            if j not in matched_g and not ignored[j]:
                stats["fn"] += 1
                by_label[g["label"]]["fn"] += 1
        # 削除しすぎ: 残った検出が取らなかった GT と、削除された検出の対応
        rest = [j for j in range(len(gts)) if j not in matched_g and not ignored[j]]
        if deleted and rest:
            md = _iou_matrix(deleted, [gts[j] for j in rest])
            for i, _ in _greedy(md, args.match_iou):
                stats["deleted_would_match"] += 1
                stats["deleted_would_match_by_" + str(deleted[i].deleted_by)] += 1
    total.update(stats)
    return stats


def _ratio(a: int, b: int) -> str:
    return f"{a / b:.3f}" if b else "-"


def main() -> None:
    args = _parse_args()
    runs = _select_runs(args)
    if not runs:
        print("対象の run がありません")
        return
    by_label: dict[str, Counter] = defaultdict(Counter)
    total: Counter = Counter()
    print(f"## run ごと（{len(runs)} シーン、order={args.order}）")
    for run_id, scene, interval in runs:
        s = check_run(run_id, scene, interval, args, by_label, total)
        fp = s["fp_no_overlap"] + s["fp_partial"] + s["fp_duplicate"] + s["label_mismatch"]
        fn = s["fn"] + s["label_mismatch"]
        print(f"  run {run_id[:8]} scene {scene[:8]}: 検出 {s['detections']} / GT {s['gt']} / "
              f"precision {_ratio(s['tp'], s['tp'] + fp)} / recall {_ratio(s['tp'], s['tp'] + fn)}")

    t = total
    fp = t["fp_no_overlap"] + t["fp_partial"] + t["fp_duplicate"] + t["label_mismatch"]
    fn = t["fn"] + t["label_mismatch"]
    print(f"\n## 全体（IoU >= {args.match_iou}、visibility {args.min_visibility} 未満と"
          "検出ラベルに対応しないカテゴリの GT は無視）")
    print(f"  フレーム {t['frames']} / 検出 {t['detections']} / GT {t['gt']}（無視した GT {t['gt_ignored']}）")
    print(f"  正解 {t['tp']} / precision {_ratio(t['tp'], t['tp'] + fp)} / "
          f"recall {_ratio(t['tp'], t['tp'] + fn)}")
    print("  誤検出の内訳:")
    print(f"    どの GT とも重ならない（IoU < {NO_OVERLAP_IOU}）: {t['fp_no_overlap']}")
    print(f"    部分的に重なる（IoU {NO_OVERLAP_IOU}〜{args.match_iou}）: {t['fp_partial']}")
    print(f"    重なる GT を別の検出が取った（重複検出）: {t['fp_duplicate']}")
    print(f"    ラベル違い: {t['label_mismatch']}")
    print(f"  見逃し（どの検出とも対応しない GT）: {t['fn']}")
    print(f"  無視した GT に対応した検出（数えない）: {t['det_on_ignored_gt']}")
    print(f"  削除しすぎ（論理削除された検出が、残りの検出の取らなかった GT と対応）: "
          f"{t['deleted_would_match']}  "
          + " / ".join(f"{k[len('deleted_would_match_by_'):]}: {v}" for k, v in sorted(t.items())
                       if k.startswith("deleted_would_match_by_")))

    print("\n## ラベルごと")
    print(f"  {'label':20s} {'TP':>6s} {'FP(重ならず)':>12s} {'FP(部分)':>9s} {'FP(重複)':>9s} "
          f"{'FP(ラベル)':>10s} {'FN':>6s} {'FN(ラベル)':>10s} {'precision':>9s} {'recall':>7s}")
    for label, c in sorted(by_label.items(), key=lambda kv: -(kv[1]["tp"] + kv[1]["fn"])):
        fp_l = c["fp_no_overlap"] + c["fp_partial"] + c["fp_duplicate"] + c["fp_label"]
        fn_l = c["fn"] + c["fn_label"]
        print(f"  {label:20s} {c['tp']:6d} {c['fp_no_overlap']:12d} {c['fp_partial']:9d} "
              f"{c['fp_duplicate']:9d} {c['fp_label']:10d} {c['fn']:6d} {c['fn_label']:10d} "
              f"{_ratio(c['tp'], c['tp'] + fp_l):>9s} {_ratio(c['tp'], c['tp'] + fn_l):>7s}")


main()
