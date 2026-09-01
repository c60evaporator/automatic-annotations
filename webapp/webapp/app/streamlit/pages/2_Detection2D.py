"""シーン選択ページ."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.streamlit import state as S
from app.streamlit.components.waypoint_viewer import (
    render_scene_waypoint_view,
)
from app.streamlit.data_access import (
    get_dataset,
    get_scene,
    list_samples,
    list_sensors,
    list_annotations_by_sample,
    list_instances_by_scene
)
from app.services.sample_service import get_skipped_sample_indices
from app.services.frame_image import get_keyframe_image

# シーン未選択ならここで描画を打ち切る（メッセージとリンクは state 側が出す）
dataset_id, scene_token = S.require_scene()
dataset = get_dataset(dataset_id)
scene = get_scene(dataset_id, scene_token)
samples = list_samples(dataset_id, scene_token)

st.subheader("2D Object Detection by Grounding DINO")

param_col, map_col = st.columns([2, 1])

# Set the parameters
with param_col:
    with st.container(border=True) as inference_container:
        st.markdown(
            """
            <div style="
                font-size: 1.4rem;
                font-weight: 600;
                margin-top: 0;
                margin-bottom: 4px;
                line-height: 1.0;
            ">
            Inference Parameters
            </div>
            """,
            unsafe_allow_html=True,
        )
        settings = get_settings()
        sample_interval_col, score_threshold_col, nms_threshold_col = st.columns([1, 1, 1])

        # Sample Interval
        with sample_interval_col:
            st.markdown("**Sample Interval**")
            sample_interval = st.number_input(
                "Sample Interval",
                min_value=1,
                max_value=10,
                value=settings.DET2D_DEFAULT_SAMPLE_INTERVAL,
                step=1,
            )
            sample_indices = get_skipped_sample_indices(len(samples), sample_interval)
            st.text("Indices: " + str(sample_indices))

        # Score Threshold
        with score_threshold_col:
            st.markdown("**Score Threshold**")
            max_default_score_threshold = max(settings.DET2D_DEFAULT_SCORE_THRESHOLDS.values())
            score_threshold_ratio = st.slider(
                "Ratio",
                key="score_threshold_ratio",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
            )
            st.table({k: v * score_threshold_ratio
                    for k, v in settings.DET2D_DEFAULT_SCORE_THRESHOLDS.items()},
                    border="horizontal",
                    width="content")

        # NMS Threshold
        with nms_threshold_col:
            st.markdown("**NMS Threshold**")
            nms_threshold_ratio = st.slider(
                "Ratio",
                key="nms_threshold_ratio",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
            )
            same_nms_col, cross_nms_col = st.columns([2, 1])
            with same_nms_col:
                st.table({k: v * nms_threshold_ratio
                        for k, v in settings.DET2D_NMS_SAME_CLASS_IOUS.items()},
                        border="horizontal",
                        width="content")
            with cross_nms_col:
                st.text("Cross-class NMS IOU: " + str(settings.DET2D_NMS_CROSS_CLASS_IOU * nms_threshold_ratio))


# Display Sample Selection and Map Preview
with map_col:
    # Sample Selection
    num_samples = len(samples)
    selected_sample_idx = st.slider(
        "Select Sample",
        min_value=0,
        max_value=max(num_samples - 1, 0),
        value=sample_indices[len(sample_indices) // 2] if sample_indices else 0,
        step=sample_interval,
    )
    # Display the waypoint
    render_scene_waypoint_view(
        dataset_id,
        dataset["dataroot"],
        scene["token"],
        title=scene["name"],
        highlight_index=selected_sample_idx,
        height=320,
        show_sample_info=False
    )

# ------------------------------------------------------------------
# Bounding box predictions
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Predicted bounding boxes view
# ------------------------------------------------------------------



# 全カメラをBounding Box付きで表示
selected_sample = samples[selected_sample_idx]
cam_sensors = list_sensors(dataset_id, modality="camera")
# Get the predicted bounding boxes from the database

# Check button for showing ground truth bounding boxes
# Get the ground truth bounding boxes from the database
sample_annotations = list_annotations_by_sample(dataset_id, selected_sample["token"])
instances = list_instances_by_scene(dataset_id, scene["token"])
# Manual annotation button
# Columun layout
cam_left_col, cam_right_col = st.columns([1, 1])

with cam_left_col:
    for i in range(len(cam_sensors)//2):
        sensor = cam_sensors[i * 2]
        image, frame = get_keyframe_image(
            dataset_id,
            dataset["dataroot"],
            selected_sample["token"],
            sensor["token"],
        )
        st.image(image, caption=f"{sensor['channel']} ({sensor['modality']})")

with cam_right_col:
    for i in range(len(cam_sensors)//2):
        sensor = cam_sensors[i * 2 + 1]
        image, frame = get_keyframe_image(
            dataset_id,
            dataset["dataroot"],
            selected_sample["token"],
            sensor["token"],
        )
        st.image(image, caption=f"{sensor['channel']} ({sensor['modality']})")
