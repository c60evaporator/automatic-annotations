"""nuScenes 本体データセット（JSON）を DB に取り込む CLI.

使い方:
    # コンテナ内
    python -m app.json_conversion.to_nusc_db --name mini --version v1.0-mini

    # ホストから
    docker compose run --rm webapp \
        python -m app.json_conversion.to_nusc_db --name mini --version v1.0-mini

想定するディレクトリ構成（--dataroot は settings.DATA_ROOT からの相対パス）:

    DATA_ROOT/<dataroot>/
      ├── v1.0-mini/            ← --version で指定。ここに *.json がある
      │     ├── log.json, scene.json, sample.json, ...
      ├── samples/              ← キーフレームのセンサーデータ
      ├── sweeps/               ← 非キーフレーム
      └── maps/
            ├── <hash>.png            ← basemap
            └── expansion/            ← Map Expansion（任意）
                  └── boston-seaport.json

設計上のポイント:

1. INSERT は FK 依存の位相順に流す。
   dataset → log → map_meta → sensor → calibrated_sensor → ego_pose
          → scene → sample → sample_data
          → category / attribute / visibility → instance → sample_annotation
          → annotation_attributes

2. prev / next の自己参照 FK は「第2パスの UPDATE」で埋める。
   1パスで入れようとすると必ず前方参照になり FK 違反で落ちる。

3. token の衝突を事前に検出して落とす。
   token はグローバルに一意ではないため、同じ DB に trainval と mini を
   読むと primary key が衝突する。運用でカバーする方針だが、
   原因の分かりにくい IntegrityError になるので入口で明示的に弾く。

4. ORM インスタンスは作らず `insert(Model)` の executemany で流す。
   trainval の sample_annotation は 100 万行を超えるため、
   ORM オブジェクト生成のオーバーヘッドが無視できない。
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import ijson  # 大きい JSON のストリーミングパース（C バックエンド yajl2_c）
import orjson
from pydantic import BaseModel, TypeAdapter
from sqlalchemy import bindparam, delete, insert, select, update
from sqlalchemy.orm import Session
from tqdm import tqdm

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import session_scope
from app.json_conversion import schemas_nuscenes as ns
from app.models.annotation import (
    Attribute,
    Category,
    Instance,
    SampleAnnotation,
    Visibility,
    annotation_attribute,
)
from app.models.dataset import Dataset
from app.models.map import MapMeta
from app.models.scene import Log, Sample, Scene
from app.models.sensor import CalibratedSensor, EgoPose, SampleData, Sensor

logger = get_logger(__name__)

M = TypeVar("M", bound=BaseModel)

# executemany の 1 バッチあたり行数
CHUNK_SIZE = 5_000
# `token IN (...)` のバインド変数上限対策（古い SQLite は 999）
IN_CLAUSE_CHUNK = 500
# nuScenes の basemap は 0.1 m / pixel（expansion が無いときの canvas_edge 推定に使う）
BASEMAP_METERS_PER_PIXEL = 0.1


class ImportError_(RuntimeError):
    """インポートを中断すべきエラー."""


# ── JSON 読み込み ──────────────────────────────────────────────────────────────

def _load(meta_dir: Path, filename: str, model: type[M], *, required: bool = True) -> list[M]:
    """nuScenes の JSON を読んで Pydantic モデルのリストにする.

    小さいファイル専用。大きいファイル（sample_annotation / sample_data /
    ego_pose）は `_stream` を使うこと。
    """
    path = meta_dir / filename
    if not path.exists():
        if required:
            raise ImportError_(f"必要な JSON が見つかりません: {path}")
        logger.warning("optional JSON not found, skipped: %s", path)
        return []
    with path.open("rb") as f:
        raw = orjson.loads(f.read())
    records = TypeAdapter(list[model]).validate_python(raw)  # type: ignore[valid-type]
    logger.info("loaded %-24s %7d records", filename, len(records))
    return records


def _stream(meta_dir: Path, filename: str, model: type[M]) -> Iterator[M]:
    """大きい JSON を1レコードずつ流す.

    trainval の sample_annotation.json は数百 MB あり、
    orjson で丸ごと読むと Python オブジェクト展開後に数 GB を占めて
    メモリ不足で落ちる。ijson の逐次パースで常時一定メモリに抑える。

    NOTE: use_float=True は必須。既定では小数が Decimal で返り、
    JSON 列に Decimal が入って書き込み時に落ちる。
    """
    path = meta_dir / filename
    if not path.exists():
        raise ImportError_(f"必要な JSON が見つかりません: {path}")
    with path.open("rb") as f:
        for record in ijson.items(f, "item", use_float=True):
            yield model.model_validate(record)


def _table(target: Any):
    """ORM モデルなら Core の Table を取り出す.

    `session.execute(insert(Model), [dict, ...])` は ORM のバルク処理パスに入り、
    主キーの同期や永続オブジェクトの追従を試みる。ここでは JSON を素通しで
    流し込むだけなので、Core の Table を直接使って executemany に落とす。
    そのほうが速く、ORM 固有の制約にも引っかからない。
    """
    return getattr(target, "__table__", target)


def _bulk_insert(session: Session, target: Any, rows: list[dict], label: str) -> None:
    """executemany で一括 INSERT する（メモリに載るサイズ用）.

    NOTE: 全 dict のキー集合が揃っている必要がある。
    キーを省いた列は server_default が効く（source='imported' 等）。
    """
    if not rows:
        return
    stmt = insert(_table(target))
    for i in tqdm(range(0, len(rows), CHUNK_SIZE),
                  desc=f"  insert {label:<20}", unit="chunk", leave=False):
        session.execute(stmt, rows[i:i + CHUNK_SIZE])
    logger.info("inserted %-24s %7d rows", label, len(rows))


def _bulk_insert_stream(
    session: Session, target: Any, rows: Iterator[dict], label: str
) -> int:
    """イテレータから一定件数ずつ INSERT する（大きいテーブル用）.

    行 dict をリストに溜め込まないので、レコード数によらずメモリが一定に保たれる。
    """
    stmt = insert(_table(target))
    buf: list[dict] = []
    total = 0
    bar = tqdm(desc=f"  insert {label:<20}", unit="row", leave=False)
    for row in rows:
        buf.append(row)
        if len(buf) >= CHUNK_SIZE:
            session.execute(stmt, buf)
            total += len(buf)
            bar.update(len(buf))
            buf.clear()
    if buf:
        session.execute(stmt, buf)
        total += len(buf)
        bar.update(len(buf))
    bar.close()
    logger.info("inserted %-24s %7d rows", label, total)
    return total


# ── token 衝突チェック ─────────────────────────────────────────────────────────

def _find_existing_tokens(session: Session, model: Any, tokens: Sequence[str]) -> list[str]:
    """指定テーブルに既に存在する token を返す."""
    # そのテーブルが空なら衝突しようがないので、走査を丸ごと省く
    if session.scalar(select(model.token).limit(1)) is None:
        return []
    found: list[str] = []
    for i in range(0, len(tokens), IN_CLAUSE_CHUNK):
        chunk = tokens[i:i + IN_CLAUSE_CHUNK]
        found.extend(session.scalars(select(model.token).where(model.token.in_(chunk))).all())
        if found:  # 1件でも見つかれば十分（全件挙げる必要はない）
            break
    return found


def _peek_tokens(meta_dir: Path, filename: str, model: type[M], limit: int = 200) -> list[str]:
    """大きい JSON の先頭から token を limit 件だけ取り出す（衝突判定用）."""
    tokens: list[str] = []
    for rec in _stream(meta_dir, filename, model):
        tokens.append(rec.token)
        if len(tokens) >= limit:
            break
    return tokens


def _check_token_collisions(session: Session, groups: dict[str, tuple[Any, Sequence[str]]]) -> None:
    """既存データとの token 衝突を検査し、あれば中断する.

    CLAUDE.md の「1データセットにつきメタデータは1つ」制約を、
    運用任せにせず入口で機械的に弾くためのガード。
    """
    for label, (model, tokens) in groups.items():
        if not tokens:
            continue
        dup = _find_existing_tokens(session, model, tokens)
        if dup:
            raise ImportError_(
                f"token が既存データと衝突しています（{label}: 例 {dup[:3]}）。\n"
                "nuScenes の token はデータセット間で一意ではないため、"
                "1つの DB に複数のメタデータ（例: trainval と mini）は読み込めません。\n"
                "別データセットとして読む場合は、既存データを削除するか "
                "--replace を指定してください。"
            )


# ── 一括 INSERT / 第2パス UPDATE ──────────────────────────────────────────────

def _link_prev_next_stream(
    session: Session, model: Any, records: Iterator[Any], label: str
) -> int:
    """prev / next を第2パスの UPDATE で埋める（ストリーミング版）.

    第1パスでは prev/next を NULL で入れておく。自己参照 FK なので、
    1パスで入れると参照先がまだ存在せず必ず FK 違反になる。

    リンク用の差分をリストに溜めると trainval では数百 MB になるため、
    JSON をもう一度流し直して一定メモリで処理する。
    JSON の再パースより、巨大リストを抱えて OOM するほうが痛い。

    バインド変数名を b_token / b_prev / b_next と列名からずらしているのは、
    UPDATE の SET 句と WHERE 句で同名パラメータが衝突するのを避けるため。
    """
    tbl = _table(model)
    stmt = (
        update(tbl)
        .where(tbl.c.token == bindparam("b_token"))
        .values(prev=bindparam("b_prev"), next=bindparam("b_next"))
    )
    buf: list[dict] = []
    total = 0
    bar = tqdm(desc=f"  link   {label:<20}", unit="row", leave=False)
    for r in records:
        if r.prev is None and r.next is None:
            continue
        buf.append({"b_token": r.token, "b_prev": r.prev, "b_next": r.next})
        if len(buf) >= CHUNK_SIZE:
            session.execute(stmt, buf)
            total += len(buf); bar.update(len(buf)); buf.clear()
    if buf:
        session.execute(stmt, buf)
        total += len(buf); bar.update(len(buf))
    bar.close()
    logger.info("linked   %-24s %7d rows", label, total)
    return total


def _link_prev_next(session: Session, model: Any, records: Sequence[Any], label: str) -> int:
    """メモリに載っているレコード列に対する prev/next リンク."""
    return _link_prev_next_stream(session, model, iter(records), label)


# ── マップ ────────────────────────────────────────────────────────────────────

def _resolve_canvas_edge(root: Path, location: str, basemap_rel: str) -> tuple[str, list[float]]:
    """Map Expansion から version / canvas_edge を得る.

    Map Expansion パックが無い環境でも動くよう、
    basemap PNG の画素数 × 0.1 m/px から推定するフォールバックを持つ。
    （nuScenes の basemap は 0.1 m/pixel）
    """
    expansion = root / "maps" / "expansion" / f"{location}.json"
    if expansion.exists():
        with expansion.open("rb") as f:
            meta = ns.MapExpansion.model_validate(orjson.loads(f.read()))
        return meta.version, meta.canvas_edge

    basemap = root / basemap_rel
    if basemap.exists():
        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None  # basemap は非常に大きい
            with Image.open(basemap) as im:
                w, h = im.size
            edge = [w * BASEMAP_METERS_PER_PIXEL, h * BASEMAP_METERS_PER_PIXEL]
            logger.warning(
                "map expansion not found for %s; canvas_edge estimated from basemap: %s",
                location, edge,
            )
            return "unknown", edge
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to read basemap %s: %s", basemap, exc)

    logger.warning("cannot determine canvas_edge for %s; falling back to [0, 0]", location)
    return "unknown", [0.0, 0.0]


def _build_map_rows(
    root: Path, dataset_id: str, maps: list[ns.Map], logs: list[ns.Log]
) -> list[dict]:
    """map.json + log.json + expansion JSON を突き合わせて MapMeta 行を作る.

    map.json 単体には location も version も canvas_edge も無い:
      - location   : log_tokens から log.location を引く
      - version    : maps/expansion/<location>.json
      - canvas_edge: 同上
    """
    log_location = {lg.token: lg.location for lg in logs}
    rows: list[dict] = []
    for mp in maps:
        locations = {log_location[t] for t in mp.log_tokens if t in log_location}
        if not locations:
            logger.warning("map %s references no known log; skipped", mp.token)
            continue
        if len(locations) > 1:
            logger.warning("map %s spans multiple locations %s; using the first",
                           mp.token, sorted(locations))
        location = sorted(locations)[0]
        version, canvas_edge = _resolve_canvas_edge(root, location, mp.filename)
        rows.append({
            "token": mp.token,
            "dataset_id": dataset_id,
            "location": location,
            "version": version,
            "canvas_edge": canvas_edge,
            "basemap_path": mp.filename,
        })
    return rows


# ── メイン処理 ────────────────────────────────────────────────────────────────

def _discover_dataroots(data_root: Path, version: str) -> list[str]:
    """DATA_ROOT 配下から、指定 version を含むデータセットルートを探す.

    DATA_ROOT の直下には複数のデータセットルートが並ぶ想定:
        /data/nuscenes/v1.0-mini/...
        /data/dataset2/v1.0-mini/...
    --dataroot の指定ミス時に候補を提示するために使う。
    """
    if not data_root.is_dir():
        return []
    found: list[str] = []
    # DATA_ROOT 直下（--dataroot .）と、その1階層下を探す
    if (data_root / version).is_dir():
        found.append(".")
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and (child / version).is_dir():
            found.append(child.name)
    return found


def import_nuscenes(
    session: Session,
    *,
    name: str,
    version: str,
    dataroot: str,
    description: str | None = None,
    replace: bool = False,
    dry_run: bool = False,
) -> str | None:
    """nuScenes メタデータを DB に取り込み、生成した dataset_id を返す."""
    settings = get_settings()
    root = (settings.DATA_ROOT / dataroot).resolve()
    meta_dir = root / version

    if not meta_dir.is_dir():
        candidates = _discover_dataroots(settings.DATA_ROOT, version)
        hint = (
            "\n  --dataroot に指定できる候補: "
            + ", ".join(repr(c) for c in candidates)
            if candidates else
            f"\n  DATA_ROOT ({settings.DATA_ROOT}) 配下に "
            f"'{version}' を含むディレクトリが見つかりません。"
            "\n  データのマウント先と --version の指定を確認してください。"
        )
        raise ImportError_(
            f"メタデータのディレクトリがありません: {meta_dir}{hint}"
        )

    logger.info("importing from %s", meta_dir)
    started = time.perf_counter()

    # --- 1. JSON 読み込み ------------------------------------------------------
    # 小さいファイルは一括で読む
    logs      = _load(meta_dir, "log.json", ns.Log)
    scenes    = _load(meta_dir, "scene.json", ns.Scene)
    samples   = _load(meta_dir, "sample.json", ns.Sample)
    sensors   = _load(meta_dir, "sensor.json", ns.Sensor)
    calibs    = _load(meta_dir, "calibrated_sensor.json", ns.CalibratedSensor)
    cats      = _load(meta_dir, "category.json", ns.Category)
    attrs     = _load(meta_dir, "attribute.json", ns.Attribute)
    vis       = _load(meta_dir, "visibility.json", ns.Visibility)
    insts     = _load(meta_dir, "instance.json", ns.Instance)
    maps      = _load(meta_dir, "map.json", ns.Map, required=False)

    # 大きいファイル（ego_pose / sample_data / sample_annotation）は
    # ここでは読まず、INSERT 時にストリーミングする。
    # trainval の sample_annotation.json は数百 MB あり、一括で読むと
    # Python オブジェクト展開後に数 GB を占めてメモリ不足で落ちる。
    for fn in ("ego_pose.json", "sample_data.json", "sample_annotation.json"):
        if not (meta_dir / fn).exists():
            raise ImportError_(f"必要な JSON が見つかりません: {meta_dir / fn}")

    if dry_run:
        # dry-run では大きいファイルも一度流してバリデーションだけ通す
        for fn, model in (("ego_pose.json", ns.EgoPose),
                          ("sample_data.json", ns.SampleData),
                          ("sample_annotation.json", ns.SampleAnnotation)):
            n = sum(1 for _ in _stream(meta_dir, fn, model))
            logger.info("validated %-23s %7d records", fn, n)
        logger.info("dry-run: JSON のバリデーションのみ実施しました（DB は変更していません）")
        return None

    # --- 2. 既存データセットの扱い -------------------------------------------
    existing = session.scalar(select(Dataset).where(Dataset.name == name))
    if existing is not None:
        if not replace:
            raise ImportError_(
                f"データセット名 '{name}' は既に存在します（id={existing.id}）。"
                "上書きするなら --replace を指定してください。"
            )
        logger.warning("replacing existing dataset '%s' (id=%s)", name, existing.id)
        # 全テーブルが dataset_id で ON DELETE CASCADE しているので1行消せば足りる
        session.execute(delete(Dataset).where(Dataset.id == existing.id))
        session.flush()

    # --- 3. token 衝突の事前チェック -----------------------------------------
    # 大きい3テーブルは全 token を集めるとそれ自体がメモリを食うので、
    # 先頭の一部だけをサンプリングして判定する。
    # token の衝突は「別データセットを丸ごと読んだ」ときに起きるもので、
    # 一部だけ衝突することは実質ないため、サンプルで十分検出できる。
    sample_tokens = _peek_tokens(meta_dir, "ego_pose.json", ns.EgoPose)
    sd_tokens     = _peek_tokens(meta_dir, "sample_data.json", ns.SampleData)
    ann_tokens    = _peek_tokens(meta_dir, "sample_annotation.json", ns.SampleAnnotation)

    _check_token_collisions(session, {
        "log":               (Log, [r.token for r in logs]),
        "scene":             (Scene, [r.token for r in scenes]),
        "sample":            (Sample, [r.token for r in samples]),
        "sensor":            (Sensor, [r.token for r in sensors]),
        "calibrated_sensor": (CalibratedSensor, [r.token for r in calibs]),
        "ego_pose":          (EgoPose, sample_tokens),
        "sample_data":       (SampleData, sd_tokens),
        "category":          (Category, [r.token for r in cats]),
        "attribute":         (Attribute, [r.token for r in attrs]),
        "visibility":        (Visibility, [r.token for r in vis]),
        "instance":          (Instance, [r.token for r in insts]),
        "sample_annotation": (SampleAnnotation, ann_tokens),
        "map":               (MapMeta, [r.token for r in maps]),
    })

    # --- 4. Dataset 行 --------------------------------------------------------
    dataset_id = str(uuid.uuid4())
    session.execute(insert(Dataset), [{
        "id": dataset_id,
        "name": name,
        "dataset_type": "nuscenes",
        "version": version,
        "dataroot": dataroot,
        "description": description,
    }])
    session.flush()
    logger.info("dataset created: %s (id=%s)", name, dataset_id)

    d = dataset_id  # 以下の行構築で頻出するため短縮

    # --- 5. 位相順に INSERT ---------------------------------------------------
    # prev/next は第1パスでは NULL のまま入れ、第2パスで UPDATE する

    _bulk_insert(session, Log, [{
        "token": r.token, "dataset_id": d, "source_token": None,
        "logfile": r.logfile, "vehicle": r.vehicle,
        "date_captured": r.date_captured, "location": r.location,
    } for r in logs], "logs")

    _bulk_insert(session, MapMeta, _build_map_rows(root, d, maps, logs), "map_meta")

    _bulk_insert(session, Sensor, [{
        "token": r.token, "dataset_id": d,
        "channel": r.channel, "modality": r.modality,
    } for r in sensors], "sensors")

    _bulk_insert(session, CalibratedSensor, [{
        "token": r.token, "dataset_id": d, "sensor_token": r.sensor_token,
        "translation": r.translation, "rotation": r.rotation,
        "camera_intrinsic": r.camera_intrinsic,
    } for r in calibs], "calibrated_sensors")

    _bulk_insert_stream(session, EgoPose, ({
        "token": r.token, "dataset_id": d, "timestamp": r.timestamp,
        "translation": r.translation, "rotation": r.rotation,
    } for r in _stream(meta_dir, "ego_pose.json", ns.EgoPose)), "ego_poses")

    _bulk_insert(session, Scene, [{
        "token": r.token, "dataset_id": d, "log_token": r.log_token,
        "name": r.name, "description": r.description, "nbr_samples": r.nbr_samples,
        "first_sample_token": r.first_sample_token,
        "last_sample_token": r.last_sample_token,
    } for r in scenes], "scenes")

    _bulk_insert(session, Sample, [{
        "token": r.token, "dataset_id": d, "scene_token": r.scene_token,
        "timestamp": r.timestamp, "prev": None, "next": None,
    } for r in samples], "samples")

    _bulk_insert_stream(session, SampleData, ({
        "token": r.token, "dataset_id": d, "sample_token": r.sample_token,
        "calibrated_sensor_token": r.calibrated_sensor_token,
        "ego_pose_token": r.ego_pose_token,
        "filename": r.filename, "fileformat": r.fileformat,
        "timestamp": r.timestamp, "is_key_frame": r.is_key_frame,
        "width": r.width, "height": r.height,
        "prev": None, "next": None,
    } for r in _stream(meta_dir, "sample_data.json", ns.SampleData)), "sample_data")

    _bulk_insert(session, Category, [{
        "token": r.token, "dataset_id": d,
        "name": r.name, "description": r.description,
    } for r in cats], "categories")

    _bulk_insert(session, Attribute, [{
        "token": r.token, "dataset_id": d,
        "name": r.name, "description": r.description,
    } for r in attrs], "attributes")

    _bulk_insert(session, Visibility, [{
        "token": r.token, "dataset_id": d,
        "level": r.level, "description": r.description,
    } for r in vis], "visibilities")

    # source は指定しない → server_default の 'imported' が入る
    _bulk_insert(session, Instance, [{
        "token": r.token, "dataset_id": d, "category_token": r.category_token,
        "nbr_annotations": r.nbr_annotations,
        "first_annotation_token": r.first_annotation_token,
        "last_annotation_token": r.last_annotation_token,
    } for r in insts], "instances")

    # sample_annotation と annotation_attributes は同じ JSON から作れるので、
    # 1回のストリーミングで両方の行を組み立てる（JSON の走査回数を減らす）。
    # attribute の行は 1 annotation あたり 0〜1 件程度で総量が小さいため、
    # こちらだけはリストに溜めて後段でまとめて INSERT する。
    attr_rows: list[dict] = []

    def _ann_rows() -> Iterator[dict]:
        for r in _stream(meta_dir, "sample_annotation.json", ns.SampleAnnotation):
            for at in r.attribute_tokens:
                attr_rows.append({"annotation_token": r.token, "attribute_token": at})
            yield {
                "token": r.token, "dataset_id": d, "sample_token": r.sample_token,
                "instance_token": r.instance_token,
                "translation": r.translation, "rotation": r.rotation, "size": r.size,
                "num_lidar_pts": r.num_lidar_pts, "num_radar_pts": r.num_radar_pts,
                "visibility_token": r.visibility_token,
                "depth_estimation_params_id": None, "score": None,
                "prev": None, "next": None,
            }

    _bulk_insert_stream(session, SampleAnnotation, _ann_rows(), "sample_annotations")
    _bulk_insert(session, annotation_attribute, attr_rows, "annotation_attributes")

    # --- 6. 第2パス: prev / next を埋める ------------------------------------
    _link_prev_next(session, Sample, samples, "samples")
    _link_prev_next_stream(
        session, SampleData,
        _stream(meta_dir, "sample_data.json", ns.SampleData), "sample_data")
    _link_prev_next_stream(
        session, SampleAnnotation,
        _stream(meta_dir, "sample_annotation.json", ns.SampleAnnotation), "sample_annotations")

    elapsed = time.perf_counter() - started
    logger.info("import finished in %.1fs (dataset_id=%s)", elapsed, dataset_id)
    return dataset_id


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="to_nusc_db",
        description="nuScenes メタデータ（JSON）を DB に取り込む",
    )
    p.add_argument("--name", required=True,
                   help="DB 上のデータセット名（一意）。例: mini")
    p.add_argument("--version", required=True,
                   help="メタデータのディレクトリ名。例: v1.0-mini")
    p.add_argument("--dataroot", required=True,
                   help="DATA_ROOT からのデータセットルートの相対パス。例: nuscenes\n"
                        "（DATA_ROOT 直下にメタデータがある場合は '.' を指定）")
    p.add_argument("--description", default=None, help="任意の説明文")
    p.add_argument("--replace", action="store_true",
                   help="同名のデータセットが既にある場合、削除してから取り込む")
    p.add_argument("--dry-run", action="store_true",
                   help="JSON のバリデーションのみ行い、DB は変更しない")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with session_scope() as session:
            import_nuscenes(
                session,
                name=args.name,
                version=args.version,
                dataroot=args.dataroot,
                description=args.description,
                replace=args.replace,
                dry_run=args.dry_run,
            )
    except ImportError_ as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
