"""シーンに対応する basemap を解決する共通コンポーネント.

キャッシュを2段に分けているのが要点:

  1. get_scene_map()      … @st.cache_data / キー = (dataset_id, scene_token)
                            DB から map_meta（basemap_path, canvas_edge）を引く
  2. load_basemap_image() … @st.cache_resource / キー = (dataset_id, basemap_path)
                            PIL の画像そのものを保持する

basemap はロケーション単位なので、1枚の画像を多数のシーンが共有する。
scene_token でキャッシュすると同じ画像がシーン数だけメモリに載るため、
画像のキャッシュキーは basemap_path にする。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import streamlit as st
from PIL import Image

from app.services.basemap_service import DEFAULT_SCALE, load_basemap
from app.streamlit.data_access import get_scene_map


@dataclass(frozen=True)
class SceneBasemap:
    """シーンに紐づく basemap 一式.

    画像が無くても軌跡だけは描けるので、失敗を例外にせず
    image=None と message で表現する。
    """
    image: Image.Image | None
    canvas_edge: Sequence[float] | None
    map_meta: dict[str, Any] | None
    message: str | None = None

    @property
    def available(self) -> bool:
        return self.image is not None and self.canvas_edge is not None


@st.cache_resource(show_spinner="basemap を準備中...")
def load_basemap_image(
    dataset_id: str, dataroot: str, basemap_path: str, scale: float = DEFAULT_SCALE
) -> Image.Image | None:
    """縮小済み basemap を返す（プロセス内で共有）.

    PIL の Image は @st.cache_data のシリアライズに向かないため
    cache_resource を使う。初回はリサイズしてディスクにキャッシュを作る。
    """
    return load_basemap(dataset_id, dataroot, basemap_path, scale=scale)


def get_scene_basemap(
    dataset_id: str, dataroot: str, scene_token: str, scale: float = DEFAULT_SCALE
) -> SceneBasemap:
    """シーンに対応する basemap を解決する."""
    map_meta = get_scene_map(dataset_id, scene_token)
    if not map_meta or not map_meta.get("basemap_path"):
        return SceneBasemap(
            None, None, map_meta,
            "このシーンに対応するマップが登録されていません。",
        )

    image = load_basemap_image(dataset_id, dataroot, map_meta["basemap_path"], scale)
    if image is None:
        return SceneBasemap(
            None, map_meta["canvas_edge"], map_meta,
            "basemap 画像が見つからないため、軌跡のみ表示しています。",
        )
    return SceneBasemap(image, map_meta["canvas_edge"], map_meta)


def render_basemap_notice(basemap: SceneBasemap) -> None:
    """basemap を用意できなかった理由をキャプションで示す."""
    if basemap.message:
        st.caption(basemap.message)
