import streamlit as st

from components.waypoint_viewer import render_scene_waypoint
from utils.page_guard import require_dataset_loaded

require_dataset_loaded()

if st.session_state.scene_token is not None:
    st.session_state.setdefault("scene_browser_revision", 0)

    # TODO: Get the scene dataframe

    # Scene selection table
    event = st.dataframe(
        scene_df,
        key=f"scene_browser_{st.session_state.scene_browser_revision}",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config={"map_token": None, "end_time": None},
        height=st.session_state.config["scene_table"]["height"],
    )

    if event.selection.rows:
        col1, col2 = st.columns([1, 3])

        with col2:
            idx = event.selection.rows[0]
            scene = (
                scene_df.iloc[idx]
            )
            # TODO: src.common.nuscenes_utils.get_scene_contentsで取得したリソースに置き換える（sample_annotation, instances以外。これら2つは良く更新されるので逐次クエリ取得する）
            st.session_state.selected_scene_contents = {
                "token": scene["token"],
                "name": scene["name"],
                "nbr_samples": scene["nbr_samples"],
                "description": scene["description"],
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
            }

            render_scene_waypoint(
                st.session_state.nusc,
                st.session_state.basemap_cache,
                st.session_state.config["basemap"]["canvas_edge"],
                scene["token"],
            )

        with col1:
            st.markdown(
                f"**Selected Scene:** {scene['name']}  \n"
                f"**Number of Samples:** {scene['nbr_samples']}  \n"
                f"**Start Time:** {scene['start_time']}  \n"
                f"**End Time:** {scene['end_time']}"
            )
            if st.button("Cut This Scene", type="primary"):
                st.switch_page(
                    "pages/2_Scene_Cutter.py"
                )
