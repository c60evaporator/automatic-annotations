"""nuScenes 形式（JSON テーブル群）への書き出し.

devkit が `NuScenes(version=..., dataroot=...)` で読み込める形にする。
出力は JSON のみで、画像や LiDAR のファイルはコピーしない
（元データが数十 GB あり、複製する意味がないため）。
利用時は元データセットの `samples/` `sweeps/` と同じ dataroot に置く。

## 気をつけている点

- **参照を閉じる**: シーンを絞って出すので、sample_data → ego_pose →
  calibrated_sensor → sensor と辿って必要な行だけを集める。
  1 つでも欠けると devkit のトークン解決が落ちる。
- **チェーンの張り直し**: アノテーションを種別で絞ると、
  instance の first/last/nbr や prev/next が出力に無い行を指しうる。
  出力対象だけで組み直す。
- **トークンの正規化**: 自動生成した UUID にはハイフンが入る。
  nuScenes のトークンは 32 桁の英数字なので、揃えて出す。
"""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import read_only_session
from app.repositories.nuscenes_export import (
    ANNOTATION_SOURCE_AUTO,
    ANNOTATION_SOURCE_GT,
    NuScenesExportRepository,
)

logger = get_logger(__name__)

# devkit が読み込むテーブル。1 つでも欠けると初期化で落ちる
NUSCENES_TABLES = (
    "category", "attribute", "visibility", "instance", "sensor",
    "calibrated_sensor", "ego_pose", "log", "scene", "sample",
    "sample_data", "sample_annotation", "map",
)

# 自動アノテーションに与える可視性。推定していないので、
# 「見えている」側に倒して学習時に除外されないようにする
DEFAULT_VISIBILITY_LEVEL = "v80-100"


def _token(value: str | None) -> str:
    """トークンを nuScenes の書式（ハイフン無し）へ揃える.

    参照側も同じ関数を通すこと。片方だけ変換すると参照が切れる。
    """
    if not value:
        return ""
    return value.replace("-", "")


def export_dir(name: str) -> Path:
    """エクスポート先のディレクトリ（DERIVED_ROOT 配下）."""
    return get_settings().DERIVED_ROOT / "export" / name


def build_tables(
    dataset_id: str,
    *,
    scene_tokens: list[str] | None = None,
    source: str = ANNOTATION_SOURCE_AUTO,
    depth_estimation_params_id: str | None = None,
    visibility_token: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """nuScenes のテーブル一式を組み立てる（ファイルには書かない）.

    Args:
        scene_tokens: 対象シーン。None ならデータセット全体
        source: 'auto' | 'gt' | 'both'
        depth_estimation_params_id: 特定の run の自動アノテーションだけ出す
        visibility_token: 可視性が未設定の行に入れるトークン。
            None なら「最も見えている」レベルを自動で選ぶ
    """
    with read_only_session() as session:
        repo = NuScenesExportRepository(session)

        dataset = repo.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"データセットが見つかりません: {dataset_id}")

        scenes = repo.list_scenes(dataset_id, scene_tokens)
        if not scenes:
            raise ValueError("対象のシーンがありません")

        samples = repo.list_samples([s["token"] for s in scenes])
        sample_tokens = [s["token"] for s in samples]
        sample_data = repo.list_sample_data(sample_tokens)

        # 参照を辿って、必要な行だけを集める
        ego_poses = repo.list_ego_poses(
            sorted({d["ego_pose_token"] for d in sample_data})
        )
        calibrated = repo.list_calibrated_sensors(
            sorted({d["calibrated_sensor_token"] for d in sample_data})
        )
        sensors = repo.list_sensors(sorted({c["sensor_token"] for c in calibrated}))
        logs = repo.list_logs(sorted({s["log_token"] for s in scenes}))

        categories = repo.list_categories(dataset_id)
        attributes = repo.list_attributes(dataset_id)
        visibilities = repo.list_visibilities(dataset_id)

        annotations = repo.list_annotations(
            sample_tokens, source=source,
            depth_estimation_params_id=depth_estimation_params_id,
        )
        instances = repo.list_instances(
            sorted({a["instance_token"] for a in annotations})
        )

    # 可視性が未設定の自動アノテーション向けの既定トークン
    if visibility_token is None:
        visibility_token = _default_visibility_token(visibilities)

    annotations = _rebuild_annotation_chains(annotations, samples)
    instances = _rebuild_instance_chains(instances, annotations)

    tables: dict[str, list[dict[str, Any]]] = {
        "category": [{
            "token": _token(c["token"]), "name": c["name"],
            "description": c["description"],
        } for c in categories],
        "attribute": [{
            "token": _token(a["token"]), "name": a["name"],
            "description": a["description"],
        } for a in attributes],
        "visibility": [{
            "token": v["token"], "level": v["level"],
            "description": v["description"],
        } for v in visibilities],
        "sensor": [{
            "token": _token(s["token"]), "channel": s["channel"],
            "modality": s["modality"],
        } for s in sensors],
        "calibrated_sensor": [{
            "token": _token(c["token"]),
            "sensor_token": _token(c["sensor_token"]),
            "translation": c["translation"], "rotation": c["rotation"],
            "camera_intrinsic": c["camera_intrinsic"],
        } for c in calibrated],
        "ego_pose": [{
            "token": _token(e["token"]), "timestamp": e["timestamp"],
            "translation": e["translation"], "rotation": e["rotation"],
        } for e in ego_poses],
        "log": [{
            "token": _token(l["token"]), "logfile": l["logfile"] or "",
            "vehicle": l["vehicle"] or "", "date_captured": l["date_captured"] or "",
            "location": l["location"] or "",
        } for l in logs],
        "scene": [{
            "token": _token(s["token"]), "log_token": _token(s["log_token"]),
            "nbr_samples": s["nbr_samples"], "name": s["name"],
            "description": s["description"] or "",
            "first_sample_token": _token(s["first_sample_token"]),
            "last_sample_token": _token(s["last_sample_token"]),
        } for s in scenes],
        "sample": [{
            "token": _token(s["token"]), "timestamp": s["timestamp"],
            "scene_token": _token(s["scene_token"]),
            "prev": _token(s["prev"]), "next": _token(s["next"]),
        } for s in samples],
        "sample_data": [{
            "token": _token(d["token"]),
            "sample_token": _token(d["sample_token"]),
            "ego_pose_token": _token(d["ego_pose_token"]),
            "calibrated_sensor_token": _token(d["calibrated_sensor_token"]),
            "filename": d["filename"], "fileformat": d["fileformat"],
            "width": d["width"] or 0, "height": d["height"] or 0,
            "timestamp": d["timestamp"], "is_key_frame": bool(d["is_key_frame"]),
            "prev": _token(d["prev"]), "next": _token(d["next"]),
        } for d in sample_data],
        "sample_annotation": [{
            "token": _token(a["token"]),
            "sample_token": _token(a["sample_token"]),
            "instance_token": _token(a["instance_token"]),
            "visibility_token": a["visibility_token"] or visibility_token,
            # 自動生成では属性を推定していない。空配列は仕様上そのまま有効
            "attribute_tokens": [],
            "translation": [float(v) for v in a["translation"]],
            "size": [float(v) for v in a["size"]],
            "rotation": [float(v) for v in a["rotation"]],
            "prev": _token(a["prev"]), "next": _token(a["next"]),
            "num_lidar_pts": int(a["num_lidar_pts"] or 0),
            "num_radar_pts": int(a["num_radar_pts"] or 0),
        } for a in annotations],
        "instance": [{
            "token": _token(i["token"]),
            "category_token": _token(i["category_token"]),
            "nbr_annotations": i["nbr_annotations"],
            "first_annotation_token": _token(i["first_annotation_token"]),
            "last_annotation_token": _token(i["last_annotation_token"]),
        } for i in instances],
        # 地図は取り込んでいないが、devkit が読み込むテーブルなので空で出す
        "map": [],
    }
    return tables


def _default_visibility_token(visibilities: list[dict[str, Any]]) -> str:
    """可視性が未設定の行に入れるトークンを選ぶ.

    自動アノテーションは可視性を推定していない。
    低い値を入れると学習時のフィルタで落とされるため、
    最も見えているレベル（v80-100）に倒す。
    """
    for visibility in visibilities:
        if visibility["level"] == DEFAULT_VISIBILITY_LEVEL:
            return visibility["token"]
    return visibilities[-1]["token"] if visibilities else ""


def _rebuild_annotation_chains(
    annotations: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """出力対象のアノテーションだけで prev/next を張り直す.

    種別や run で絞ると、元の prev/next が出力に無い行を指しうる。
    その参照を残すと devkit のトークン解決が落ちる。
    """
    order = {s["token"]: index for index, s in enumerate(samples)}
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        by_instance.setdefault(annotation["instance_token"], []).append(annotation)

    for chain in by_instance.values():
        chain.sort(key=lambda a: order.get(a["sample_token"], 0))
        for index, annotation in enumerate(chain):
            annotation["prev"] = chain[index - 1]["token"] if index > 0 else ""
            annotation["next"] = (
                chain[index + 1]["token"] if index + 1 < len(chain) else ""
            )
    return annotations


def _rebuild_instance_chains(
    instances: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """instance の first/last/nbr を、出力対象のアノテーションから作り直す."""
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        by_instance.setdefault(annotation["instance_token"], []).append(annotation)

    rebuilt: list[dict[str, Any]] = []
    for instance in instances:
        chain = by_instance.get(instance["token"], [])
        if not chain:
            # アノテーションが 1 つも残らない instance は出さない
            continue
        # prev が空のものが先頭。張り直し済みなので順序は信用できる
        first = next((a for a in chain if not a["prev"]), chain[0])
        last = next((a for a in chain if not a["next"]), chain[-1])
        rebuilt.append({
            **instance,
            "nbr_annotations": len(chain),
            "first_annotation_token": first["token"],
            "last_annotation_token": last["token"],
        })
    return rebuilt


def write_tables(
    tables: dict[str, list[dict[str, Any]]],
    output_root: Path,
    version: str,
) -> Path:
    """テーブルを ``<output_root>/<version>/*.json`` へ書き出す.

    Returns:
        書き出したディレクトリ。
    """
    directory = output_root / version
    directory.mkdir(parents=True, exist_ok=True)

    for name in NUSCENES_TABLES:
        rows = tables.get(name, [])
        path = directory / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            # devkit は JSON をそのまま読むだけなので、
            # 読みやすさより出力サイズを優先する
            json.dump(rows, handle, ensure_ascii=False, separators=(",", ":"))
    return directory


def make_zip(directory: Path, zip_path: Path) -> Path:
    """書き出したディレクトリを zip にまとめる（ダウンロード用）."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory.parent))
    return zip_path


def export_nuscenes(
    dataset_id: str,
    *,
    scene_tokens: list[str] | None = None,
    source: str = ANNOTATION_SOURCE_AUTO,
    depth_estimation_params_id: str | None = None,
    version: str = "v1.0-auto",
    name: str | None = None,
) -> dict[str, Any]:
    """テーブルを組み立てて書き出し、zip まで作る.

    Returns:
        ``{"directory", "zip_path", "counts", "version"}``
    """
    tables = build_tables(
        dataset_id, scene_tokens=scene_tokens, source=source,
        depth_estimation_params_id=depth_estimation_params_id,
    )

    name = name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = export_dir(name)
    if root.exists():
        shutil.rmtree(root)
    directory = write_tables(tables, root, version)
    zip_path = make_zip(directory, root / f"{version}.zip")

    counts = {table: len(rows) for table, rows in tables.items()}
    logger.info("exported nuScenes tables to %s: %s", directory, counts)
    return {
        "directory": directory, "zip_path": zip_path,
        "counts": counts, "version": version, "name": name,
    }


def cleanup_export(name: str) -> None:
    """書き出したエクスポートを削除する."""
    root = export_dir(name)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
