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
from app.repositories.dataset import DatasetRepository
from app.repositories.scene import SceneRepository

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


def clear_caches() -> None:
    """データを書き換えた後に呼ぶ（インポート・削除・推論実行後など）."""
    st.cache_data.clear()
