"""2D Object Detection ページ.

## 推論の起動方式

「Run Inference」でジョブを登録し、job_id だけを session_state に持つ。
推論はサーバー側で走り続けるので、スライダーを動かして再実行が入っても
推論は止まらない。

進捗表示は st.fragment(run_every=...) を使う。
フラグメントの中だけが定期的に再実行されるので、
パラメータ欄や画像グリッドを巻き込んで再描画せずに済む。
（ページ全体を st.rerun でループさせると、そのたびに
  画像の再デコードとテーブルの再構築が走って重い）
"""
from __future__ import annotations

import streamlit as st

from app.core.config import get_settings
from app.services.inference_client import (
    InferenceServerError,
    cancel_detection2d_job,
    get_detection2d_job,
    is_available,
    submit_detection2d,
)
from app.services.label_service import (
    label_groups,
    scaled_nms_same_class_ious,
    scaled_score_thresholds,
    validate_label_config,
)
from app.services.sample_service import get_skipped_sample_indices
from app.services.frame_image import get_keyframe_image
from app.streamlit import state as S
from app.streamlit.components.det2d_viewer import render_camera_grid
from app.streamlit.components.waypoint_viewer import render_scene_waypoint_view
from app.streamlit.data_access import (
    get_dataset,
    get_scene,
    list_frames_by_scene,
    list_samples,
    list_sensors,
)

# シーン未選択ならここで描画を打ち切る
dataset_id, scene_token = S.require_scene()
dataset = get_dataset(dataset_id)
scene = get_scene(dataset_id, scene_token)
samples = list_samples(dataset_id, scene_token)
cam_sensors = list_sensors(dataset_id, modality="camera")
n_cameras = len(cam_sensors)
settings = get_settings()

st.subheader("2D Object Detection by Grounding DINO")

# ラベルを増やしたときの設定漏れ（閾値やカテゴリ変換の未定義）は
# 実行時まで気付きにくいので、ページ表示のタイミングで洗い出す
config_problems = validate_label_config()
if config_problems:
    with st.expander(f"⚠️ ラベル設定に {len(config_problems)} 件の問題があります"):
        for msg in config_problems:
            st.write("-", msg)

# ジョブ状態は session_state に置く。
# job_id はシーンに紐づくので、シーンを変えたら state 側でクリアされる
JOB_ID = S.DET2D_JOB_ID
RESULTS = "det2d_results"        # {sample_data_token: [box, ...]}
PARTIAL_SINCE = "det2d_since"    # 受け取り済みの部分結果の件数

st.session_state.setdefault(RESULTS, {})
st.session_state.setdefault(PARTIAL_SINCE, 0)


param_col, map_col = st.columns([2, 1])

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
with param_col:
    with st.container(border=True):
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
        sample_interval_col, score_threshold_col, nms_threshold_col = st.columns(3)

        with sample_interval_col:
            st.markdown("**Sample Interval**")
            sample_interval = st.number_input(
                "Sample Interval", min_value=1, max_value=10,
                value=settings.DET2D_DEFAULT_SAMPLE_INTERVAL, step=1,
                label_visibility="collapsed",
            )
            sample_indices = get_skipped_sample_indices(len(samples), sample_interval)
            st.caption(f"{len(sample_indices)} samples: {sample_indices}")
            n_groups = len(label_groups())
            st.caption(
                f"推論回数 = {len(sample_indices)} sample "
                f"× {n_cameras} camera × {n_groups} group "
                f"= {len(sample_indices) * n_cameras * n_groups}"
            )

        with score_threshold_col:
            st.markdown("**Score Threshold**")
            score_threshold_ratio = st.slider(
                "Ratio", key="score_threshold_ratio",
                min_value=0.0, max_value=2.0, value=1.0, step=0.05,
            )
            # 閾値はカテゴリグループ単位（config の
            # DET2D_DEFAULT_SCORE_THRESHOLDS がグループ名をキーに持つ）
            score_thresholds = scaled_score_thresholds(score_threshold_ratio)
            st.table(score_thresholds)

        with nms_threshold_col:
            st.markdown("**NMS Threshold**")
            nms_threshold_ratio = st.slider(
                "Ratio", key="nms_threshold_ratio",
                min_value=0.0, max_value=2.0, value=1.0, step=0.05,
            )
            nms_same = scaled_nms_same_class_ious(nms_threshold_ratio)
            same_nms_col, cross_nms_col = st.columns([2, 1])
            with same_nms_col:
                st.table(nms_same, border="horizontal", width="content")
            with cross_nms_col:
                st.text("Cross-class NMS IOU: " + str(settings.DET2D_NMS_CROSS_CLASS_IOU * nms_threshold_ratio))

# ------------------------------------------------------------------
# Sample selection & map
# ------------------------------------------------------------------
with map_col:
    num_samples = len(samples)
    default_idx = sample_indices[len(sample_indices) // 2] if sample_indices else 0
    selected_sample_idx = st.select_slider(
        "Select Sample",
        options=sample_indices or [0],
        value=default_idx,
    )
    render_scene_waypoint_view(
        dataset_id, dataset["dataroot"], scene["token"],
        title=scene["name"],
        highlight_index=selected_sample_idx,
        height=320,
        show_sample_info=False,
    )

# ------------------------------------------------------------------
# Run / progress
# ------------------------------------------------------------------
def _build_payload() -> dict:
    """推論リクエストを組み立てる.

    対象フレームは webapp 側で決めて渡す。推論サーバーは DB を持たないので、
    「どの画像を処理するか」を知っているのはこちら側だけ。
    """
    selected_tokens = {samples[i]["token"]: i for i in sample_indices}
    cam_frames = list_frames_by_scene(dataset_id, scene_token)
    frames = [
        {
            "sample_data_token": f["token"],
            "sample_token": f["sample_token"],
            "filename": f["filename"],
            "channel": f["channel"],
            "width": f["width"],
            "height": f["height"],
            "sample_idx": selected_tokens[f["sample_token"]],
        }
        for f in cam_frames
        if f["modality"] == "camera" and f["sample_token"] in selected_tokens
    ]
    # 閾値はグループ単位でまとめて渡す。
    # 並列の dict を複数渡す形だと、キーの取りこぼしが実行時まで分からない
    groups = [
        {
            "name": name,
            "labels": labels,
            "score_threshold": score_thresholds.get(name, 0.3),
            "nms_same_class_iou": nms_same.get(name, 0.6),
        }
        for name, labels in label_groups().items()
    ]
    return {
        "dataroot": dataset["dataroot"],
        "frames": frames,
        "label_groups": groups,
        "nms_cross_class_iou": min(
            settings.DET2D_NMS_CROSS_CLASS_IOU * nms_threshold_ratio, 1.0
        ),
        "stub_delay_sec": settings.DET2D_STUB_DELAY_SEC,
    }


def _merge_partial(items: list[dict]) -> None:
    """部分結果を sample_data_token 単位で取り込む."""
    store = st.session_state[RESULTS]
    for item in items:
        store[item["sample_data_token"]] = item.get("boxes", [])


with param_col:
    with st.container(border=True):
        run_col, cancel_col, status_col = st.columns([1, 1, 3])

        with run_col:
            disabled = bool(st.session_state.get(JOB_ID))
            if st.button("Run Inference", type="primary", disabled=disabled):
                if not is_available():
                    st.error("推論サーバーに接続できません。")
                else:
                    try:
                        job = submit_detection2d(_build_payload())
                        S.set_selection(JOB_ID, job["job_id"])
                        st.session_state[RESULTS] = {}
                        st.session_state[PARTIAL_SINCE] = 0
                        st.rerun()
                    except InferenceServerError as exc:
                        st.error(str(exc))

        with cancel_col:
            if st.session_state.get(JOB_ID):
                if st.button("Cancel"):
                    try:
                        cancel_detection2d_job(st.session_state[JOB_ID])
                    except InferenceServerError as exc:
                        st.warning(str(exc))

        @st.fragment(run_every=1.0)
        def progress_area() -> None:
            """進捗表示。ここだけが1秒ごとに再実行される."""
            job_id = st.session_state.get(JOB_ID)
            if not job_id:
                done = len(st.session_state[RESULTS])
                if done:
                    st.caption(f"検出結果: {done} フレーム分を保持中")
                return

            try:
                job = get_detection2d_job(job_id, since=st.session_state[PARTIAL_SINCE])
            except InferenceServerError as exc:
                st.error(str(exc))
                return

            new = job.get("partial") or []
            if new:
                _merge_partial(new)
                st.session_state[PARTIAL_SINCE] += len(new)

            st.progress(
                min(job["progress"], 1.0),
                text=f"{job['processed']}/{job['total']}  {job['message']}",
            )
            st.caption(
                f"status={job['status']} / {job['elapsed_sec']:.1f}s / "
                f"完了フレーム {st.session_state[PARTIAL_SINCE]}"
            )

            if job["status"] in ("succeeded", "failed", "cancelled"):
                st.session_state.pop(JOB_ID, None)
                if job["status"] == "failed":
                    st.error(job.get("error") or "推論に失敗しました")
                # フラグメント外（画像グリッド）も更新するため全体を再実行する
                st.rerun()

        with status_col:
            progress_area()

# ------------------------------------------------------------------
# Predicted bounding boxes view
# ------------------------------------------------------------------
st.divider()

view_col, opt_col = st.columns([9, 1])
with opt_col:
    show_boxes = st.checkbox("Show boxes", value=True)
    min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05)

selected_sample = samples[selected_sample_idx]
results = st.session_state[RESULTS]

items = []
for sensor in cam_sensors:
    image, frame = get_keyframe_image(
        dataset_id, dataset["dataroot"], selected_sample["token"], sensor["token"]
    )
    token = frame["token"] if frame else None
    items.append({
        "channel": sensor["channel"],
        "image": image,
        "boxes": results.get(token, []),
        "pending": token not in results,
    })

with view_col:
    render_camera_grid(items, columns=2, min_score=min_score, show_boxes=show_boxes)

S.render_selection_sidebar(dataset_name=dataset["name"], scene_name=scene["name"])
