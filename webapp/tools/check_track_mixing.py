"""トラックに GT の別物体が混ざる原因を、トラッキング / カメラ間結合に分けて数える.

Depth Boxfitting の run の box_fittings の各行（カメラごとのインスタンス × フレーム）について、
トラッキングの 2D ボックスを、同じカメラ・フレームに投影した GT の 2D ボックスと IoU で
1 対 1 に対応付け（IoU の大きい組からの貪欲法、IoU >= --match-iou）、GT のインスタンスを割り当てる。
そのうえで、box_merged_track（ボックスの規定長と向きに基づく結合の後のトラック。導入前の run では
global_track_id）に 2 番目に多い GT が --min-frames フレーム以上含まれていれば「混入あり」とし、
原因を次の段階に振り分ける（重複あり）:

  - トラッキング      : 1 本のカメラ内トラック（channel, track_id）に GT が 2 つ以上
  - Hull Merge        : 1 つの hull_merged_track に、主な GT が違うカメラ内トラックが同居
  - Box-Orient Merge  : 1 つの box_merged_track に、主な GT が違う hull_merged_track が同居

GT と対応しない行（Detection2D の誤検出など）は GT のインスタンスを持ち込まないので、混入の
原因には数えない。Detection2D の誤りは tools/check_det2d_errors.py で調べる。

注意: 3D ボックスの中心から最も近い GT を探す測り方では、位置がずれたボックスが隣の物体に
対応して「混入」に数えられてしまう。ここでは 2D（トラッキングのボックス）で対応付ける。

webapp コンテナの中で動かす。使い方（docker-compose.yml のある webapp フォルダで）:

    docker compose run --rm --no-deps -T webapp python - [オプション] < tools/check_track_mixing.py

  ``--no-deps`` を付けると推論サーバーを起動しない（DB を読むだけなので不要）。

オプション（いずれも省略可）:
  --runs RUN_ID ...   調べる Depth Boxfitting の run（前方一致）。省略すると、ボックスの規定長と
                      向きに基づく結合（box_merge_diagnostics）を記録している成功 run を、
                      シーンごとに最新の 1 件ずつ使う
  --max-scenes N      対象シーン数の上限（既定: 無制限）。対象 run を持つシーンが上限を
                      超えるときは --order で選ぶ
  --order ORDER       上限を超えるときの選び方。newest（run が新しいシーンから）/
                      random（ランダム。--seed で再現できる）。既定 newest
  --seed N            --order random の乱数の種（既定 0）
  --match-iou IOU     GT と対応付ける IoU の下限（既定 0.3。トラッキングのボックスはマスクの
                      外接矩形なので、3D GT の投影より小さく出やすい）
  --min-frames N      2 番目に多い GT がこのフレーム数以上なら混入とみなす（既定 2）
  --examples N        原因の組み合わせごとに表示する例の数（既定 3）

例:
    docker compose run --rm --no-deps -T webapp python - --runs 4a12a77d < tools/check_track_mixing.py
    docker compose run --rm --no-deps -T webapp python - --max-scenes 30 --order random < tools/check_track_mixing.py
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

import numpy as np
from sqlalchemy import select

from app.db.session import read_only_session
from app.models.ann_intermediate import BoxFitting3D, DepthEstimationParams, InstanceTracking2D
from app.repositories.annotation import AnnotationRepository
from app.repositories.sensor import SensorRepository
from app.services.annotation_2d import project_annotations_to_frame

CAUSES = ("tracking", "hull_merge", "box_orient_merge")
CAUSE_TEXT = {"tracking": "トラッキング", "hull_merge": "Hull Merge",
              "box_orient_merge": "Box-Orient Merge"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="check_track_mixing")
    parser.add_argument("--runs", nargs="*", help="Depth Boxfitting の run id（前方一致）")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--order", choices=("newest", "random"), default="newest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


def _select_runs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """(run id, scene_token) のリスト. シーンごとに最新の成功 run を、新しい順に."""
    with read_only_session() as session:
        rows = session.execute(
            select(DepthEstimationParams.id, DepthEstimationParams.scene_token,
                   DepthEstimationParams.box_merge_diagnostics.is_not(None))
            .where(DepthEstimationParams.status == "succeeded")
            .order_by(DepthEstimationParams.started_at.desc())
        ).all()
    if args.runs:
        rows = [(i, s, True) for i, s, _ in rows if any(i.startswith(p) for p in args.runs)]
    latest: dict[str, str] = {}
    for run_id, scene, has_box_merge in rows:
        if has_box_merge and scene not in latest:
            latest[scene] = run_id
    runs = [(run_id, scene) for scene, run_id in latest.items()]
    if args.max_scenes is not None and len(runs) > args.max_scenes:
        if args.order == "random":
            runs = random.Random(args.seed).sample(runs, args.max_scenes)
        else:
            runs = runs[:args.max_scenes]
    return runs


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dominant(counter: Counter) -> str | None:
    return counter.most_common(1)[0][0] if counter else None


def _match_frame(frame_rows, gts, match_iou) -> dict[int, str]:
    """1 フレームの行と GT を IoU の大きい組から 1 対 1 に対応付ける. {行 id: GT のインスタンス}."""
    matched: dict[int, str] = {}
    if not gts:
        return matched
    m = np.array([[_iou((r.xmin, r.ymin, r.xmax, r.ymax),
                        (g["xmin"], g["ymin"], g["xmax"], g["ymax"])) for g in gts]
                  for r in frame_rows])
    used_r: set[int] = set()
    used_g: set[int] = set()
    for flat in np.argsort(-m, axis=None):
        i, j = np.unravel_index(flat, m.shape)
        if m[i, j] < match_iou:
            break
        if i in used_r or j in used_g:
            continue
        used_r.add(i)
        used_g.add(j)
        matched[frame_rows[i].id] = gts[j]["instance_token"]
    return matched


def check_run(run_id: str, scene: str, args: argparse.Namespace,
              total: Counter, examples: dict) -> Counter:
    def mixed(counter: Counter) -> bool:
        top = counter.most_common(2)
        return len(top) == 2 and top[1][1] >= args.min_frames

    with read_only_session() as session:
        rows = session.execute(select(
            BoxFitting3D.id, BoxFitting3D.sample_data_token, BoxFitting3D.track_id,
            BoxFitting3D.hull_merged_track_id,
            BoxFitting3D.box_merged_track_id, BoxFitting3D.global_track_id,
            InstanceTracking2D.xmin, InstanceTracking2D.ymin,
            InstanceTracking2D.xmax, InstanceTracking2D.ymax,
        ).join(InstanceTracking2D, InstanceTracking2D.id == BoxFitting3D.instance_tracking_2d_id)
            .where(BoxFitting3D.depth_estimation_params_id == run_id)).all()
        frames = {f["token"]: f for f in SensorRepository(session).list_frames_by_scene(scene)}
        annotations = AnnotationRepository(session).list_by_scene(
            scene, source="imported", include_attributes=False)
    anns_by_sample = defaultdict(list)
    for a in annotations:
        anns_by_sample[a["sample_token"]].append(a)

    # 行ごとに GT のインスタンスを割り当てる
    rows_by_frame = defaultdict(list)
    for r in rows:
        rows_by_frame[r.sample_data_token].append(r)
    gt_of: dict[int, str] = {}
    for sample_data, frame_rows in rows_by_frame.items():
        frame = frames.get(sample_data)
        gts = (project_annotations_to_frame(anns_by_sample[frame["sample_token"]], frame)
               if frame else [])
        gt_of.update(_match_frame(frame_rows, gts, args.match_iou))

    stats = Counter()
    stats["rows"] = len(rows)
    stats["rows_unmatched"] = sum(1 for r in rows if r.id not in gt_of)

    # カメラ内トラック → hull_merged_track → box_merged_track
    cam_track: dict[tuple, Counter] = defaultdict(Counter)
    hull_members: dict[str, set] = defaultdict(set)
    box_members: dict[str, set] = defaultdict(set)
    for r in rows:
        key = (frames[r.sample_data_token]["channel"], r.track_id)
        cam_track[key]  # GT と対応しないトラックも数える
        if r.id in gt_of:
            cam_track[key][gt_of[r.id]] += 1
        hull = str(r.hull_merged_track_id or r.global_track_id or f"{key[0]}:{key[1]}")
        box = str(r.box_merged_track_id or r.global_track_id or hull)
        hull_members[hull].add(key)
        box_members[box].add(hull)

    stats["cam_tracks"] = len(cam_track)
    stats["cam_tracks_mixed"] = sum(mixed(c) for c in cam_track.values())
    for box, hulls in box_members.items():
        stats["box_tracks"] += 1
        hull_gt = {h: sum((cam_track[k] for k in hull_members[h]), Counter()) for h in hulls}
        if not mixed(sum(hull_gt.values(), Counter())):
            continue
        stats["box_tracks_mixed"] += 1
        causes = []
        if any(mixed(cam_track[k]) for h in hulls for k in hull_members[h]):
            causes.append("tracking")
        if any(len({_dominant(cam_track[k]) for k in hull_members[h] if cam_track[k]}) > 1
               for h in hulls):
            causes.append("hull_merge")
        if len({_dominant(c) for c in hull_gt.values() if c}) > 1:
            causes.append("box_orient_merge")
        for cause in causes:
            stats["cause_" + cause] += 1
        combo = " + ".join(CAUSE_TEXT[c] for c in causes) or "判定できず"
        stats["combo_" + combo] += 1
        if len(examples[combo]) < args.examples:
            examples[combo].append(
                f"run {run_id[:8]} box_merged {box}: hull_merged {sorted(hulls)} / "
                f"カメラ内トラック {sorted(f'{c}:{t}' for h in hulls for c, t in hull_members[h])}"
                f" / GT {dict(sum(hull_gt.values(), Counter()).most_common())}")
    total.update(stats)
    return stats


def _pct(a: int, b: int) -> str:
    return f"{a} / {b}（{100.0 * a / b:.1f}%）" if b else f"{a} / 0"


def main() -> None:
    args = _parse_args()
    runs = _select_runs(args)
    if not runs:
        print("対象の run がありません")
        return
    total: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    print(f"## run ごと（{len(runs)} シーン、order={args.order}）")
    for run_id, scene in runs:
        s = check_run(run_id, scene, args, total, examples)
        print(f"  run {run_id[:8]} scene {scene[:8]}: 混入 "
              f"{_pct(s['box_tracks_mixed'], s['box_tracks'])}")

    t = total
    print(f"\n## トラックへの別物体の混入（{len(runs)} run、IoU >= {args.match_iou}、"
          f"2 番目の GT が {args.min_frames} フレーム以上）")
    print(f"  box_merged_track       : {_pct(t['box_tracks_mixed'], t['box_tracks'])}")
    print(f"  カメラ内トラック単体   : {_pct(t['cam_tracks_mixed'], t['cam_tracks'])}")
    print("  原因（重複あり）:")
    for cause in CAUSES:
        print(f"    {CAUSE_TEXT[cause]:18s}: {t['cause_' + cause]}")
    print("  原因の組み合わせ:")
    for key in sorted(k for k in t if k.startswith("combo_")):
        print(f"    {key[len('combo_'):]:30s}: {t[key]}")

    print(f"  （GT と対応せず混入の判定に使わなかった行: {_pct(t['rows_unmatched'], t['rows'])}）")

    print("\n## 例")
    for combo, items in examples.items():
        print(f"  [{combo}]")
        for item in items:
            print(f"    {item}")


main()
