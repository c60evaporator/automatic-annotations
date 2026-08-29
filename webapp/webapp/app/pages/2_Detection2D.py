import os
import fnmatch
from pathlib import Path

import streamlit as st

from services.cut_scene_manager import init_cut_scene_session_state, get_cut_scenes_list, load_cut_scene
from services.dataset_creator import create_new_dataset
from components.waypoint_viewer import render_scene_waypoint
from components.sensor_viewer import render_selected_sample_sensors
from utils.page_guard import require_dataset_loaded
from utils.scene_utils import get_samples_in_scene

# Guard
require_dataset_loaded()

scene_info = st.session_state.get("selected_scene_contents")
if scene_info is None:
    st.warning("Please select a scene first.")
    st.switch_page("pages/1_Scene_Selection.py")
    st.stop()

else:
    # ------------------------------------------------------------------
    # Bounding box predictions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Predicted bounding boxes view
    # ------------------------------------------------------------------
    # Load the selected cut scene
    matched = cut_scene_df.index[cut_scene_df["token"] == st.session_state.selected_scene_token]
    if len(matched) == 0:
        st.warning("Please select a Details button for a cut scene")
        st.session_state.selected_scene_token = None
        st.stop()
    idx = matched[0]
    scene_name = cut_scene_df.iloc[idx]["name"]
    load_ph = st.empty()
    load_ph.write(f"Loading cut scene: {scene_name}")
    cut_nusc = load_cut_scene(
        Path(st.session_state.nusc.dataroot),
        Path(cut_root_dir) / cut_scene_df.iloc[idx]["src_version"] / scene_name,
    )
    samples_df = get_samples_in_scene(cut_nusc, cut_nusc.scene[0]["token"])
    load_ph.empty()

    # Columun layout
    left_col, right_col = st.columns([1, 3])

    # Display Sample Selection and Map Preview
    with left_col:
        # Sample Selection
        num_samples = cut_scene_df.iloc[idx]["nbr_samples"]
        selected_sample_idx = st.slider(
            "Select Sample",
            min_value=0,
            max_value=max(num_samples - 1, 0),
            value=max(num_samples - 1, 0)//2,
        )
        # Display the waypoint of the selected cut scene
        render_scene_waypoint(
            cut_nusc,
            st.session_state.basemap_cache,
            st.session_state.config["basemap"]["canvas_edge"],
            cut_nusc.scene[0]["token"],
            highlight_index=selected_sample_idx,
        )
    
    # 全カメラをBounding Box付きで表示
    with right_col:
        # Columun layout
        cam_left_col, cam_right_col = st.columns([1, 1])

