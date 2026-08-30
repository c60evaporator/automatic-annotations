import os
import json
import yaml

import streamlit as st
import pandas as pd

from utils.session import init_session
from utils.map_utils import build_basemap_cache
from services.nuscenes_loader import build_datasets_dataframe, load_nuscenes
from services.scene_manager import build_scene_dataframe

init_session()

# Set the page configuration
st.set_page_config(
    page_title="NuScenes Cutter",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the configuration
with open("config/settings.yml", "r") as f:
    config = yaml.safe_load(f)
st.session_state.config = config

# Header placeholder
header_ph = st.empty()
header_ph.subheader("Loading the datasets")

# Get the list of dataset folders in the input directory
INPUT_BASE = "/data/input"
datasets_df = build_datasets_dataframe(INPUT_BASE)

header_ph.subheader("Please select a dataset to load")

# Show the datasets in a selectable table
event = st.dataframe(
    datasets_df,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
)
selected_rows = event.selection.rows
selected_folder = datasets_df.iloc[selected_rows[0]]["dataset_name"] if selected_rows else None
selected_version = datasets_df.iloc[selected_rows[0]]["version"] if selected_rows else None

# Select the dataset folder and load the dataset
if st.button("Load Dataset", disabled=not selected_rows):
    header_ph.write(f"Loading dataset: {selected_folder}, version: {selected_version}")
    dataset_root = os.path.join(INPUT_BASE, selected_folder)
    # Store the dataset root in session state
    st.session_state.dataset_root = dataset_root
    nusc = load_nuscenes(dataset_root, selected_version)
    st.session_state.nusc = nusc
    st.session_state.scene_df = build_scene_dataframe(nusc)
    st.session_state.basemap_cache = build_basemap_cache(nusc, scale=st.session_state.config["basemap"]["cache_scale"])
    st.session_state.selected_scene_info = None
    st.session_state.scene_browser_revision += 1
    header_ph.empty()
    st.markdown(f"**Loaded dataset: {selected_folder}, version: {selected_version}**")

if st.session_state.nusc is not None:
    if st.button("Open Dataset Browser", type="primary"):
        st.switch_page("pages/1_Dataset_Browser.py")
