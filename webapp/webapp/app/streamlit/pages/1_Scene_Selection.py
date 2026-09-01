"""シーン選択ページ."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.streamlit import state as S
from app.streamlit.components.waypoint_viewer import (
    format_timestamp,
    render_scene_waypoint_view,
)
from app.streamlit.data_access import (
    count_annotated_samples,
    get_dataset,
    list_scenes,
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

# アノテーション進捗（全シーン分を1クエリで取得）
progress = count_annotated_samples(dataset_id)

STATUS_LABEL = {
    "complete": "✅ 完了",
    "partial": "🟡 途中",
    "none": "⬜ 未着手",
    "empty": "— サンプル無し",
}


# --- シーン一覧 ---------------------------------------------------------------
df = pd.DataFrame(scenes)
df_view = pd.DataFrame({
    "name": df["name"],
    "status": [STATUS_LABEL.get(progress.get(t, {}).get("status", "empty"), "")
               for t in df["token"]],
    "progress": [
        (progress.get(t, {}).get("annotated_samples", 0)
         / max(1, progress.get(t, {}).get("nbr_samples", 0)))
        for t in df["token"]
    ],
    "annotated": [
        f"{progress.get(t, {}).get('annotated_samples', 0)}"
        f" / {progress.get(t, {}).get('nbr_samples', 0)}"
        for t in df["token"]
    ],
    "location": df["location"],
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
        "status": st.column_config.TextColumn("状態"),
        "progress": st.column_config.ProgressColumn(
            "進捗", min_value=0.0, max_value=1.0, format="%.0f%%"
        ),
        "annotated": st.column_config.TextColumn("アノテーション済"),
        "location": st.column_config.TextColumn("ロケーション"),
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
    render_scene_waypoint_view(
        dataset_id,
        dataset["dataroot"],
        scene["token"],
        title=scene["name"],
    )

S.render_selection_sidebar(
    dataset_name=dataset["name"],
    scene_name=next((s["name"] for s in scenes
                     if s["token"] == S.get(S.SCENE_TOKEN)), None),
)
