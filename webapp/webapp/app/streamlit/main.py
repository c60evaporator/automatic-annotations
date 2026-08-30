"""メインページ: データセット選択."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.streamlit import state as S
from app.streamlit.data_access import clear_caches, get_dataset, list_datasets

# set_page_config はアプリ内で最初の Streamlit 呼び出しである必要があり、
# メインスクリプトでのみ設定する
st.set_page_config(
    page_title="Automatic Annotation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚗 Automatic Annotation")
st.caption("nuScenes データセットの自動アノテーション（3D bounding box）デモ")

datasets = list_datasets()

if not datasets:
    st.warning("データセットがまだ登録されていません。")
    settings = get_settings()
    st.markdown(
        "先にインポート CLI を実行してください。\n\n"
        "```bash\n"
        "docker compose run --rm webapp \\\n"
        "  python -m app.json_conversion.to_nusc_db \\\n"
        "    --name mini --version v1.0-mini --dataroot nuscenes\n"
        "```\n\n"
        f"データは `{settings.DATA_ROOT}` にマウントされています。"
        "`--dataroot` にはその直下のデータセットルート名を指定します。"
    )
    st.stop()

# --- 一覧表示 -----------------------------------------------------------------
df = pd.DataFrame(datasets)[[
    "name", "version", "dataset_type", "dataroot",
    "nbr_scenes", "nbr_samples", "nbr_annotations", "description", "created_at",
]]

st.subheader("データセットを選択")
event = st.dataframe(
    df,
    hide_index=True,
    width="stretch",
    selection_mode="single-row",
    on_select="rerun",
    key="_w_dataset_table",
    column_config={
        "name": st.column_config.TextColumn("名前"),
        "version": st.column_config.TextColumn("バージョン"),
        "dataset_type": st.column_config.TextColumn("種別"),
        "dataroot": st.column_config.TextColumn("データルート"),
        "nbr_scenes": st.column_config.NumberColumn("シーン数", format="%d"),
        "nbr_samples": st.column_config.NumberColumn("サンプル数", format="%d"),
        "nbr_annotations": st.column_config.NumberColumn("アノテーション数", format="%d"),
        "description": st.column_config.TextColumn("説明"),
        "created_at": st.column_config.DatetimeColumn("登録日時", format="YYYY-MM-DD HH:mm"),
    },
)

selected_rows = event.selection.rows if event and event.selection else []
selected = datasets[selected_rows[0]] if selected_rows else None

# --- 選択の確定 ---------------------------------------------------------------
# テーブルの選択状態はウィジェットに紐づくため、ページを移ると失われる。
# ボタンで明示的に正規キーへ確定させる（state.set_selection）。
col1, col2, col3 = st.columns([2, 2, 6])

with col1:
    if st.button("このデータセットを使う", type="primary", disabled=selected is None):
        S.set_selection(S.DATASET_ID, selected["id"])
        st.switch_page("pages/1_Scene_Selection.py")

with col2:
    if st.button("再読み込み", help="DB の変更をキャッシュに反映します"):
        clear_caches()
        st.rerun()

# --- 現在の選択 ---------------------------------------------------------------
current_id = S.get(S.DATASET_ID)
current = get_dataset(current_id) if current_id else None

if current:
    st.success(f"選択中: **{current['name']}** ({current['version']})")
    if st.button("シーン選択へ進む"):
        st.switch_page("pages/1_Scene_Selection.py")
elif selected:
    st.info(f"**{selected['name']}** を選択中です。"
            "「このデータセットを使う」で確定してください。")

S.render_selection_sidebar(dataset_name=current["name"] if current else None)
