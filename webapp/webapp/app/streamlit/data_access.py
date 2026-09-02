"""Streamlit 用のキャッシュ付きデータアクセス.

Repository を直接ページから呼ばず、この層を挟む理由:

1. @st.cache_data の対象を1か所に集約するため。
   キャッシュキーは引数から作られるので、dataset_id / scene_token を
   必ず引数に含める。含め忘れるとデータセットを切り替えても
   古い結果が返り続ける。

2. ORM オブジェクトを UI に漏らさないため。
   @st.cache_data は戻り値を保持するので、Session が閉じた後に
   ORM オブジェクトへ触ると DetachedInstanceError になる。
   Repository が dict を返す設計と対になっている。

NOTE: Engine / sessionmaker は app.db.engine / app.db.session 側で
lru_cache により1つに保たれる。Streamlit はスクリプトを再実行するが
import 済みモジュールは再読み込みされないため、
これらを @st.cache_resource で包む必要はない。
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from app.db.session import read_only_session
from app.repositories.annotation import AnnotationRepository
from app.repositories.dataset import DatasetRepository
from app.repositories.scene import SceneRepository
from app.repositories.sensor import SensorRepository
from app.services.annotation_2d import project_annotations_to_frame
from app.services.detection2d_service import (
    list_runs as _list_runs,
    load_run_boxes as _load_run_boxes,
)
from app.services.instance_tracking_service import (
    list_prompt_runs as _list_prompt_runs,
    list_runs as _list_tracking_runs,
    list_track_ids as _list_track_ids,
    load_run_instances as _load_run_instances,
)

# 推論結果は頻繁に更新されるため、参照系のキャッシュは短めにする
CACHE_TTL_SEC = 300


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner="データセットを読み込み中...")
def list_datasets() -> list[dict[str, Any]]:
    with read_only_session() as session:
        return DatasetRepository(session).list()


@st.cache_data(ttl=CACHE_TTL_SEC)
def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    with read_only_session() as session:
        return DatasetRepository(session).get(dataset_id)


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner="シーンを読み込み中...")
def list_scenes(dataset_id: str) -> list[dict[str, Any]]:
    with read_only_session() as session:
        return SceneRepository(session).list_by_dataset(dataset_id)


@st.cache_data(ttl=CACHE_TTL_SEC)
def get_scene(dataset_id: str, scene_token: str) -> dict[str, Any] | None:
    with read_only_session() as session:
        return SceneRepository(session).get(dataset_id, scene_token)


@st.cache_data(ttl=CACHE_TTL_SEC)
def list_waypoints(dataset_id: str, scene_token: str) -> list[dict[str, Any]]:
    """自車軌跡。dataset_id は使わないが、キャッシュキーに含めるため受け取る."""
    with read_only_session() as session:
        return SceneRepository(session).list_waypoints(scene_token)


@st.cache_data(ttl=CACHE_TTL_SEC)
def get_scene_map(dataset_id: str, scene_token: str) -> dict[str, Any] | None:
    with read_only_session() as session:
        return SceneRepository(session).get_map(dataset_id, scene_token)


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner="アノテーション進捗を集計中...")
def count_annotated_samples(
    dataset_id: str, *, source: str | None = None
) -> dict[str, dict[str, Any]]:
    """データセット内の全シーンのアノテーション進捗を一括集計する.

    戻り値は scene_token をキーにした dict:
        {"nbr_samples": 40, "annotated_samples": 12, "status": "partial"}

    シーンごとにクエリを投げると N+1 になるため、1クエリで全シーン分を返す。
    """
    with read_only_session() as session:
        return SceneRepository(session).count_annotated_samples(
            dataset_id, source=source
        )


# ── sample ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SEC)
def list_samples(dataset_id: str, scene_token: str) -> list[dict[str, Any]]:
    """シーン内の sample をタイムスタンプ順で返す."""
    with read_only_session() as session:
        return SceneRepository(session).list_samples(scene_token)


# ── センサーフレーム（sample_data + calibrated_sensor + ego_pose）─────────────

@st.cache_data(ttl=CACHE_TTL_SEC)
def list_frames_by_sample(
    dataset_id: str,
    sample_token: str,
    *,
    keyframe_only: bool = True,
    sensor_token: str | None = None,
) -> list[dict[str, Any]]:
    """1 sample のセンサーフレームを、投影に必要な情報ごと返す."""
    with read_only_session() as session:
        return SensorRepository(session).list_frames_by_sample(
            sample_token, keyframe_only=keyframe_only, sensor_token=sensor_token
        )


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner="センサーデータを読み込み中...")
def list_frames_by_scene(
    dataset_id: str,
    scene_token: str,
    *,
    keyframe_only: bool = True,
    sensor_token: str | None = None,
) -> list[dict[str, Any]]:
    """シーン内の全センサーフレームを1クエリでまとめて返す（バッチ処理用）."""
    with read_only_session() as session:
        return SensorRepository(session).list_frames_by_scene(
            scene_token, keyframe_only=keyframe_only, sensor_token=sensor_token
        )


@st.cache_data(ttl=CACHE_TTL_SEC)
def list_sensors(dataset_id: str, *, modality: str | None = None) -> list[dict[str, Any]]:
    """センサー（チャンネル）一覧。UI のチャンネル選択に使う.

    modality に 'camera' / 'lidar' / 'radar' を渡すと種類で絞れる。
    """
    with read_only_session() as session:
        return SensorRepository(session).list_sensors(dataset_id, modality=modality)


@st.cache_data(ttl=CACHE_TTL_SEC)
def list_categories(
    dataset_id: str, *, include_counts: bool = False
) -> list[dict[str, Any]]:
    """カテゴリ一覧。ラベルマッピングの設定 UI に使う.

    include_counts=True で instance 数・annotation 数を付与する。
    """
    with read_only_session() as session:
        return AnnotationRepository(session).list_categories(
            dataset_id, include_counts=include_counts
        )


# ── アノテーション ────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SEC)
def list_annotations_by_sample(
    dataset_id: str,
    sample_token: str,
    *,
    source: str | None = None,
    include_attributes: bool = True,
) -> list[dict[str, Any]]:
    """1 sample のアノテーションを instance 情報付きで返す."""
    with read_only_session() as session:
        return AnnotationRepository(session).list_by_sample(
            sample_token, source=source, include_attributes=include_attributes
        )


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner="アノテーションを読み込み中...")
def list_annotations_by_scene(
    dataset_id: str,
    scene_token: str,
    *,
    source: str | None = None,
    include_attributes: bool = True,
) -> list[dict[str, Any]]:
    """シーン内の全アノテーションを1クエリでまとめて返す（バッチ処理用）."""
    with read_only_session() as session:
        return AnnotationRepository(session).list_by_scene(
            scene_token, source=source, include_attributes=include_attributes
        )


@st.cache_data(ttl=CACHE_TTL_SEC)
def list_instances_by_scene(
    dataset_id: str, scene_token: str, *, source: str | None = None
) -> list[dict[str, Any]]:
    """シーンに登場する instance の一覧（トラック単位の表示用）."""
    with read_only_session() as session:
        return AnnotationRepository(session).list_instances_by_scene(
            scene_token, source=source
        )


# ── Ground truth の 2D 投影 ──────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SEC)
def list_gt_boxes_2d(
    dataset_id: str,
    sample_token: str,
    sensor_token: str,
    *,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """指定 sample / カメラに写るアノテーションを 2D BBox にして返す.

    3D → 2D の投影は numpy の計算が入るので、
    再実行のたびに走らないようキャッシュする。
    """
    with read_only_session() as session:
        frames = SensorRepository(session).list_frames_by_sample(
            sample_token, keyframe_only=True, sensor_token=sensor_token
        )
        if not frames:
            return []
        annotations = AnnotationRepository(session).list_by_sample(
            sample_token, source=source, include_attributes=False
        )
    return project_annotations_to_frame(annotations, frames[0])


def clear_caches() -> None:
    """データを書き換えた後に呼ぶ（インポート・削除・推論実行後など）."""
    st.cache_data.clear()


# ── 2D 検出 run ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SEC)
def list_detection_runs(dataset_id: str, scene_token: str) -> list[dict[str, Any]]:
    """シーンの検出 run 一覧（新しい順）."""
    return _list_runs(dataset_id, scene_token)


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner="検出結果を読み込み中...")
def load_detection_run_boxes(params_id: str) -> dict[str, list[dict[str, Any]]]:
    """run の検出結果を {sample_data_token: [box, ...]} で返す."""
    return _load_run_boxes(params_id)


# ── Instance Tracking run ────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SEC)
def list_tracking_runs(dataset_id: str, scene_token: str) -> list[dict[str, Any]]:
    """シーンのトラッキング run 一覧（新しい順）."""
    return _list_tracking_runs(dataset_id, scene_token)


@st.cache_data(ttl=CACHE_TTL_SEC)
def list_prompt_detection_runs(
    dataset_id: str, scene_token: str
) -> list[dict[str, Any]]:
    """Box Prompt に選べる Detection2D run（成功したもののみ）."""
    return _list_prompt_runs(dataset_id, scene_token)


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner="トラッキング結果を読み込み中...")
def load_tracking_run_instances(
    params_id: str, *, include_mask: bool = True
) -> dict[str, list[dict[str, Any]]]:
    """run の結果を {sample_data_token: [instance, ...]} で返す.

    include_mask=False にすると mask_rle を読まない。
    マスクは 1 件で数千要素になるので、外接矩形だけで足りる表示では効く。
    """
    return _load_run_instances(params_id, include_mask=include_mask)


@st.cache_data(ttl=CACHE_TTL_SEC)
def list_tracking_track_ids(params_id: str) -> list[dict[str, Any]]:
    """run に含まれる track の一覧（凡例・色分け用）."""
    return _list_track_ids(params_id)
