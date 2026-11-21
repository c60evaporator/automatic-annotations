"""Detection2D の閾値探索用に、後処理前の検出と GT をシーンごとにキャッシュする.

閾値探索では、試行のたびに GroundingDINO と SigLIP2 を動かし直すと遅すぎる。
そこで、推論は低いスコア閾値・緩い NMS で 1 回だけ行い、後処理（スコア閾値・NMS・
再判定・IoS 結合）をオフラインでやり直せる形で保存する。

キャッシュに入れるもの（シーンごとに 1 ファイル）:
  - 全キーフレーム × 全カメラの検出（スコア --min-score より大きいもの。NMS は
    同一クラス --nms-same-max、クラス間 --nms-cross-max で、探索範囲の上限にしてある）
  - 再判定の対象ラベルの検出には、SigLIP2 の全候補のスコア（reclassification_scores）
  - そのフレームに投影した GT の 2D ボックス（visibility とカテゴリ付き）

キャッシュに入れないもの:
  - IoS 結合（スコアに依存しない幾何の判定なので、探索側でやり直す）
  - 検出 0 件のフレームの救済（Whole-Image Fallback）。後処理の結果で対象のフレームが
    変わるため、探索の対象外にする

sample_interval は使わず、全キーフレームを推論する。GT はキーフレームにしかないので、
間引くと評価に使えるデータが減るだけになる（評価はフレーム単位なので、間引いても結果の
期待値は変わらない）。

## シーンの選び方と追加

GT のあるシーンを --order で並べ（random なら --seed でシャッフル）、先頭から --max-scenes 件を
使う。同じ --seed なら、--max-scenes を 10 から 20 に増やしたときの 20 件は最初の 10 件を
含むので、キャッシュ済みのシーンは飛ばして残りの 10 件だけを推論する。

キャッシュの置き場所は ``DERIVED_ROOT/det2d_cache/<--name>/``（シーンごとの
``<scene_token>.json.gz`` と、推論の設定を記録した ``meta.json``）。設定を変えて作り直すときは
別の --name を指定する（同じ --name で設定が違うと、混ざらないよう止める）。
シーンを選んだ順は meta.json の scene_order に残し、探索（tools/tune_det2d_thresholds.py）の
--max-scenes はこの順で先頭から使う。

## 使い方（docker-compose.yml のある webapp フォルダで）

推論サーバーが必要なので、先に起動しておく:

    docker compose up -d inference

    docker compose run --rm --no-deps -T webapp python - [オプション] < tools/build_det2d_cache.py

オプション（いずれも省略可）:
  --max-scenes N       キャッシュするシーン数（既定 10）
  --order ORDER        シーンの並べ方。random（--seed でシャッフル）/ name（シーン名順）。既定 random
  --seed N             --order random の乱数の種（既定 0）
  --scenes NAME ...    シーンを名前で直接指定する（--max-scenes / --order は使わない）
  --dataset NAME       データセット名（データセットが 1 つなら省略可）
  --name NAME          キャッシュの名前（既定 default）
  --min-score S        キャッシュするスコアの下限（既定 0.10。探索範囲の下限以下にする）
  --nms-same-max IOU   同一クラス NMS の IoU（既定 0.90。探索範囲の上限にする）
  --nms-cross-max IOU  クラス間 NMS の IoU（既定 0.95。探索範囲の上限にする）
  --log-interval SEC   シーン内の進捗を表示する間隔（秒。既定 30）

例: まず 10 シーン、あとで 20 シーンに増やす（増やした分だけ推論する）

    docker compose run --rm --no-deps -T webapp python - --max-scenes 10 < tools/build_det2d_cache.py
    docker compose run --rm --no-deps -T webapp python - --max-scenes 20 < tools/build_det2d_cache.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import read_only_session
from app.models.dataset import Dataset
from app.models.scene import Scene
from app.repositories.annotation import AnnotationRepository
from app.repositories.sensor import SensorRepository
from app.services import inference_client
from app.services.annotation_2d import project_annotations_to_frame
from app.services.label_service import label_groups, sublabel_groups, sublabel_to_label

POLL_SEC = 2.0
# 推論の設定のうち、キャッシュの中身を変えるもの。同じ --name で違えば止める
META_KEYS = ("dataset_id", "model_name", "min_score", "nms_same_max", "nms_cross_max",
             "crop_margin_ratio", "re_classification_candidates", "sublabel_groups")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="build_det2d_cache")
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--order", choices=("random", "name"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", nargs="*", help="シーン名")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--name", default="default")
    parser.add_argument("--min-score", type=float, default=0.10)
    parser.add_argument("--nms-same-max", type=float, default=0.90)
    parser.add_argument("--nms-cross-max", type=float, default=0.95)
    parser.add_argument("--log-interval", type=float, default=30.0)
    return parser.parse_args()


def _fmt(sec: float | None) -> str:
    if sec is None:
        return "--:--"
    sec = int(round(sec))
    h, rest = divmod(sec, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _select_dataset(name: str | None) -> tuple[str, str]:
    with read_only_session() as session:
        rows = session.execute(select(Dataset.id, Dataset.name, Dataset.dataroot)).all()
    if name:
        rows = [r for r in rows if r.name == name]
    if len(rows) != 1:
        names = ", ".join(r.name for r in rows) or "なし"
        sys.exit(f"データセットを 1 つに決められません（候補: {names}）。--dataset で指定してください")
    return rows[0].id, rows[0].dataroot


def _select_scenes(dataset_id: str, args: argparse.Namespace) -> list[tuple[str, str]]:
    """(scene_token, scene_name) のリスト."""
    with read_only_session() as session:
        rows = session.execute(
            select(Scene.token, Scene.name).where(Scene.dataset_id == dataset_id)
            .order_by(Scene.name)).all()
    scenes = [(r.token, r.name) for r in rows]
    if args.scenes:
        by_name = {name: token for token, name in scenes}
        missing = [n for n in args.scenes if n not in by_name]
        if missing:
            sys.exit(f"見つからないシーン: {', '.join(missing)}")
        return [(by_name[n], n) for n in args.scenes]
    if args.order == "random":
        # 全体をシャッフルして先頭から取る。--max-scenes を増やしても先頭は変わらない
        random.Random(args.seed).shuffle(scenes)
    return scenes[:args.max_scenes]


def _check_meta(cache_dir: Path, meta: dict) -> None:
    path = cache_dir / "meta.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = [k for k in META_KEYS if old.get(k) != meta.get(k)]
        if diff:
            sys.exit(f"キャッシュ {cache_dir} は別の設定で作られています（違う項目: {', '.join(diff)}）。"
                     "別の --name を指定してください")
        meta = {**old, **meta, "created_at": old.get("created_at", meta["created_at"])}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def _load_scene_inputs(scene_token: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """カメラのキーフレームと、フレームごとに投影した GT."""
    with read_only_session() as session:
        frames = [f for f in SensorRepository(session).list_frames_by_scene(
            scene_token, keyframe_only=True) if f["modality"] == "camera"]
        annotations = AnnotationRepository(session).list_by_scene(
            scene_token, source="imported", include_attributes=False)
    anns_by_sample = defaultdict(list)
    for a in annotations:
        anns_by_sample[a["sample_token"]].append(a)
    visibility = {a["token"]: int(a["visibility_token"])
                  if str(a.get("visibility_token") or "").isdigit() else None
                  for a in annotations}
    gts: dict[str, list[dict]] = {}
    for f in frames:
        gts[f["token"]] = [
            {"xmin": g["xmin"], "ymin": g["ymin"], "xmax": g["xmax"], "ymax": g["ymax"],
             "label": g["label"], "category_name": g["category_name"],
             "instance_token": g["instance_token"],
             "visibility": visibility.get(g["annotation_token"])}
            for g in project_annotations_to_frame(anns_by_sample[f["sample_token"]], f)
        ]
    return frames, gts


def _build_payload(dataroot: str, frames: list[dict], args: argparse.Namespace,
                   margin: float) -> dict:
    settings = get_settings()
    groups = sublabel_groups()
    return {
        "dataroot": dataroot,
        "frames": [{
            "sample_data_token": f["token"], "sample_token": f["sample_token"],
            "filename": f["filename"], "channel": f["channel"],
            "width": f["width"], "height": f["height"], "sample_idx": f["sample_idx"],
        } for f in frames],
        "label_groups": [{
            "name": name, "labels": groups[name],
            "score_threshold": args.min_score, "nms_same_class_iou": args.nms_same_max,
        } for name in label_groups()],
        "nms_cross_class_iou": args.nms_cross_max,
        "sublabel_to_label": sublabel_to_label(),
        "re_classification_candidates": {
            label: dict(mapping)
            for label, mapping in settings.RE_CLASSIFICATION_CANDIDATES.items()},
        "reclassification_crop_margin_ratio": margin,
        "return_reclassification_scores": True,
        # 救済と IoS 結合は探索側の扱いなので、推論ではしない
        "whole_image_candidates": {},
        "ios_delete_threshold": 0.0,
        "stub_delay_sec": settings.DET2D_STUB_DELAY_SEC,
    }


class Progress:
    """全体の残り時間の見積もり. 終わったシーンの平均時間 × 残りシーン数 + 実行中のシーンの残り."""

    def __init__(self, num_todo: int) -> None:
        self.num_todo = num_todo
        self.durations: list[float] = []
        self.started = time.time()

    def remaining(self, index: int, scene_elapsed: float, fraction: float) -> tuple[float | None, float | None]:
        """(実行中のシーンの残り, 全体の残り). index は今回推論するシーンの 0 始まりの番号."""
        scene_total = scene_elapsed / fraction if fraction >= 0.05 else None
        scene_left = scene_total - scene_elapsed if scene_total is not None else None
        per_scene = (sum(self.durations) / len(self.durations)) if self.durations else scene_total
        if scene_left is None or per_scene is None:
            return scene_left, None
        return scene_left, scene_left + per_scene * (self.num_todo - index - 1)


def _run_scene(payload: dict, label: str, index: int, progress: Progress,
               log_interval: float) -> dict | None:
    """ジョブを投げて終わるまで待つ. 成功なら result を返す."""
    job = inference_client.submit_detection2d(payload)
    job_id = job["job_id"]
    started = time.time()
    last_log = started
    try:
        while True:
            time.sleep(POLL_SEC)
            # 部分結果は使わないので受け取らない（since を大きくする）
            job = inference_client.get_detection2d_job(job_id, since=10**9)
            if job["status"] in inference_client.TERMINAL_STATUSES:
                break
            now = time.time()
            if now - last_log < log_interval:
                continue
            last_log = now
            done, total = job.get("processed", 0), job.get("total", 0) or 1
            fraction = done / total
            if done == 0:
                phase = "モデル準備中"
            elif "再判定" in (job.get("message") or ""):
                phase = "SigLIP2 再判定"
            else:
                phase = "GroundingDINO"
            scene_left, all_left = progress.remaining(index, now - started, fraction)
            overall = (index + fraction) / progress.num_todo
            _log(f"  {label} {phase} {fraction:4.0%} ({done}/{total})"
                 f"  経過 {_fmt(now - started)}  残り約 {_fmt(scene_left)}"
                 f"  | 全体 {overall:4.0%}  残り約 {_fmt(all_left)}")
    except KeyboardInterrupt:
        inference_client.cancel_detection2d_job(job_id)
        _log("中断しました（推論ジョブもキャンセルしました）。キャッシュ済みのシーンは残ります")
        raise SystemExit(1)
    if job["status"] != "succeeded":
        _log(f"  {label} 失敗: status={job['status']} {job.get('error') or ''}")
        return None
    progress.durations.append(time.time() - started)
    return job["result"]


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    if not inference_client.is_available():
        sys.exit("推論サーバーに接続できません。先に `docker compose up -d inference` で起動してください")

    dataset_id, dataroot = _select_dataset(args.dataset)
    scenes = _select_scenes(dataset_id, args)
    cache_dir = settings.DERIVED_ROOT / "det2d_cache" / args.name
    margin = float(settings.DEFAULT_RECLASSIFICATION_CROP_MARGIN_RATIO)
    _check_meta(cache_dir, {
        "dataset_id": dataset_id,
        "model_name": settings.DET2D_MODEL_NAME,
        "min_score": args.min_score,
        "nms_same_max": args.nms_same_max,
        "nms_cross_max": args.nms_cross_max,
        "crop_margin_ratio": margin,
        "re_classification_candidates": settings.RE_CLASSIFICATION_CANDIDATES,
        "sublabel_groups": sublabel_groups(),
        "label_groups": label_groups(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    # シーンを選んだ順を残す。探索側の --max-scenes はこの順で先頭から取る
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    order = meta.get("scene_order", [])
    order += [t for t, _ in scenes if t not in order]
    meta["scene_order"] = order
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    todo = [(t, n) for t, n in scenes if not (cache_dir / f"{t}.json.gz").exists()]
    _log(f"キャッシュ: {cache_dir}")
    _log(f"対象 {len(scenes)} シーン（キャッシュ済み {len(scenes) - len(todo)}、今回推論 {len(todo)}）"
         f"  min_score={args.min_score} nms_same={args.nms_same_max} nms_cross={args.nms_cross_max}"
         f" crop_margin={margin}")
    if not todo:
        _log("すべてキャッシュ済みです")
        return

    progress = Progress(len(todo))
    failed: list[str] = []
    for index, (scene_token, scene_name) in enumerate(todo):
        label = f"[{index + 1}/{len(todo)}] {scene_name}"
        frames, gts = _load_scene_inputs(scene_token)
        if not frames:
            _log(f"{label} カメラのキーフレームがないので飛ばします")
            continue
        num_gt = sum(len(v) for v in gts.values())
        num_samples = len({f["sample_token"] for f in frames})
        _log(f"{label} 開始: {num_samples} sample × カメラ = {len(frames)} 枚、GT {num_gt} 個")
        result = _run_scene(_build_payload(dataroot, frames, args, margin), label, index,
                            progress, args.log_interval)
        if result is None:
            failed.append(scene_name)
            continue

        by_token = {f["token"]: f for f in frames}
        out_frames = []
        num_boxes = num_scored = 0
        for r in result["frames"]:
            f = by_token[r["sample_data_token"]]
            boxes = r["boxes"]
            num_boxes += len(boxes)
            num_scored += sum(1 for b in boxes if b.get("reclassification_scores"))
            out_frames.append({
                "sample_data_token": f["token"], "sample_token": f["sample_token"],
                "sample_idx": f["sample_idx"], "channel": f["channel"],
                "width": f["width"], "height": f["height"], "error": r.get("error"),
                "boxes": boxes, "gts": gts[f["token"]],
            })
        data = {"scene_token": scene_token, "scene_name": scene_name,
                "inference_time": result.get("inference_time"), "frames": out_frames}
        # 書きかけのファイルをキャッシュ済みと見なさないよう、一時ファイルから置き換える
        tmp = cache_dir / f"{scene_token}.json.gz.tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
        tmp.replace(cache_dir / f"{scene_token}.json.gz")
        overall_left = (sum(progress.durations) / len(progress.durations)
                        * (len(todo) - index - 1))
        _log(f"{label} 完了 {_fmt(progress.durations[-1])}: 検出 {num_boxes} 個"
             f"（再判定スコア付き {num_scored}）| 全体 {(index + 1) / len(todo):4.0%}"
             f"  経過 {_fmt(time.time() - progress.started)}  残り約 {_fmt(overall_left)}")

    _log(f"終了: {len(todo) - len(failed)}/{len(todo)} シーンをキャッシュしました"
         f"（経過 {_fmt(time.time() - progress.started)}）")
    if failed:
        _log(f"失敗したシーン（もう一度実行すると推論し直します）: {', '.join(failed)}")


main()
