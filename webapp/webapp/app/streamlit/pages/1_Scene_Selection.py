"""シーン選択ページ."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.services.basemap_service import load_basemap
from app.streamlit import state as S
from app.streamlit.components.waypoint_viewer import (
    format_timestamp,
    render_scene_waypoint,
)
from app.streamlit.data_access import (
    get_dataset,
    get_scene_map,
    list_scenes,
    list_waypoints,
)

# データセット未選択ならここで描画を打ち切る（メッセージとリンクは state 側が出す）
dataset_id = S.require_dataset()
dataset = get_dataset(dataset_id)

st.title("🎬 シーン選択")
st.caption(f"データセット: **{dataset['name']}** ({dataset['version']})")

scenes = list_scenes(dataset_id)
if not scenes:
    st.warning("このデータセットにはシーンがありません。")
    st.stop()


@st.cache_resource(show_spinner="basemap を準備中...")
def _basemap(dataset_id: str, dataroot: str, basemap_path: str):
    """basemap 画像は PIL オブジェクトなので cache_resource で保持する.

    初回はリサイズしてディスクにキャッシュを作るため数秒かかる。
    """
    return load_basemap(dataset_id, dataroot, basemap_path)


# --- シーン一覧 ---------------------------------------------------------------
df = pd.DataFrame(scenes)
df_view = pd.DataFrame({
    "name": df["name"],
    "location": df["location"],
    "nbr_samples": df["nbr_samples"],
    "start_time": [format_timestamp(t, with_date=True) if t else "" for t in df["start_time"]],
    "description": df["description"],
})

event = st.dataframe(
    df_view,
    hide_index=True,
    width="stretch",
    selection_mode="single-row",
    on_select="rerun",
    key="_w_scene_table",
    height=320,
    column_config={
        "name": st.column_config.TextColumn("シーン"),
        "location": st.column_config.TextColumn("ロケーション"),
        "nbr_samples": st.column_config.NumberColumn("サンプル数", format="%d"),
        "start_time": st.column_config.TextColumn("開始時刻"),
        "description": st.column_config.TextColumn("説明"),
    },
)

selected_rows = event.selection.rows if event and event.selection else []

# 未選択なら、確定済みのシーンがあればそれをプレビューする
if selected_rows:
    scene = scenes[selected_rows[0]]
else:
    committed = S.get(S.SCENE_TOKEN)
    scene = next((s for s in scenes if s["token"] == committed), None)

if scene is None:
    st.info("シーンを選択すると、自車軌跡と詳細が表示されます。")
    S.render_selection_sidebar(dataset_name=dataset["name"])
    st.stop()

# --- 詳細と軌跡 ---------------------------------------------------------------
info_col, map_col = st.columns([1, 3])

with info_col:
    st.subheader(scene["name"])
    st.markdown(
        f"**ロケーション:** {scene['location']}  \n"
        f"**車両:** {scene['vehicle']}  \n"
        f"**撮影日:** {scene['date_captured']}  \n"
        f"**サンプル数:** {scene['nbr_samples']}  \n"
        f"**開始:** {format_timestamp(scene['start_time'])}  \n"
        f"**終了:** {format_timestamp(scene['end_time'])}"
    )
    if scene["description"]:
        st.markdown(f"**説明:** {scene['description']}")

    st.divider()

    next_page = Path(__file__).parent / "2_Detection2D.py"
    if st.button("このシーンでアノテーションを開始", type="primary"):
        S.set_selection(S.SCENE_TOKEN, scene["token"])
        if next_page.exists():
            st.switch_page("pages/2_Detection2D.py")
        else:
            st.success(f"シーン **{scene['name']}** を選択しました。")
            st.info("2D Object Detection ページは未実装です。")

with map_col:
    map_meta = get_scene_map(dataset_id, scene["token"])
    basemap_img = None
    canvas_edge = None
    if map_meta and map_meta["basemap_path"]:
        basemap_img = _basemap(dataset_id, dataset["dataroot"], map_meta["basemap_path"])
        canvas_edge = map_meta["canvas_edge"]
        if basemap_img is None:
            st.caption("basemap 画像が見つからないため、軌跡のみ表示しています。")
    else:
        st.caption("このシーンに対応するマップが登録されていません。")

    waypoints = list_waypoints(dataset_id, scene["token"])
    render_scene_waypoint(
        waypoints,
        title=f"{scene['name']} ({len(waypoints)} samples)",
        basemap_img=basemap_img,
        canvas_edge=canvas_edge,
    )

S.render_selection_sidebar(
    dataset_name=dataset["name"],
    scene_name=next((s["name"] for s in scenes
                     if s["token"] == S.get(S.SCENE_TOKEN)), None),
)
