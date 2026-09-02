"""Instance Tracking ページ.

構成は 2_Detection2D.py を踏襲している（パラメータ → 実行 → 保存済み run →
Sample スライダー → 地図 → 画像グリッド）。

Detection2D との違い:
  - プロンプトに使う Detection2D run を選ぶ
  - Sample スライダーは全 sample を選べる
    （トラッキングは interval の間の sample にも結果が出るため）
  - 表示はマスク中心。色分けを「ラベル」と「Track ID」で切り替えられる
"""
from __future__ import annotations

import streamlit as st

from app.core.config import get_settings
from app.models.ann_intermediate import IOU_LABEL_MATCHES, IOU_METHODS
from app.services.inference_client import (
    InferenceServerError,
    cancel_instance_tracking_job,
    get_instance_tracking_job,
    is_available,
    submit_instance_tracking,
)
from app.services.frame_image import get_keyframe_image
from app.services.instance_tracking_service import (
    build_tracking_payload,
    delete_run,
    resolve_display_run,
    save_tracking_run,
)
from app.services.sweep_service import MAX_SWEEPS_PER_SAMPLE
from app.streamlit import state as S
from app.streamlit.components.det2d_viewer import render_label_legend
from app.streamlit.components.instance_tracking_viewer import (
    BOX_MODE_INSTANCE,
    BOX_MODE_PROMPT,
    BOX_MODES,
    COLOR_MODE_LABEL,
    COLOR_MODE_TRACK,
    COLOR_MODES,
    INSTANCE_TEXT_MODES,
    TEXT_MODE_TRACK,
    color_for_track,
    legend_entries,
    render_instance_grid,
)
from app.streamlit.components.det2d_viewer import color_for as color_for_label
from app.streamlit.components.waypoint_viewer import render_scene_waypoint_view
from app.streamlit.data_access import (
    clear_caches,
    get_dataset,
    get_scene,
    list_prompt_detection_runs,
    list_samples,
    list_sensors,
    list_tracking_runs,
    load_detection_run_boxes,
    load_tracking_run_instances,
)

dataset_id, scene_token = S.require_scene()
dataset = get_dataset(dataset_id)
scene = get_scene(dataset_id, scene_token)
samples = list_samples(dataset_id, scene_token)
cam_sensors = list_sensors(dataset_id, modality="camera")
n_cameras = len(cam_sensors)
settings = get_settings()

st.subheader("Instance Tracking by SAM2")

JOB_ID = S.TRACKING_JOB_ID
RESULTS = S.TRACKING_RESULTS
PARTIAL_SINCE = S.TRACKING_PARTIAL_SINCE
SAVED_JOB_ID = S.TRACKING_SAVED_JOB_ID
VIEW_RUN_ID = S.TRACKING_VIEW_RUN_ID

st.session_state.setdefault(PARTIAL_SINCE, 0)

# 表示オプションは正規キーに退避する。
# 「Run Inference」の st.rerun() で、その下のウィジェットは生成されず
# 状態が破棄されるため
OPT_BOX, W_BOX = "tracking_box_mode", "_w_tracking_box_mode"
OPT_COLOR, W_COLOR = "tracking_color_mode", "_w_tracking_color_mode"
OPT_TEXT, W_TEXT = "tracking_text_mode", "_w_tracking_text_mode"

# --- 初期表示: DB に保存済みの run から読み込む -------------------------------
if RESULTS not in st.session_state:
    run_id, _reason = resolve_display_run(dataset_id, scene_token)
    if run_id:
        st.session_state[RESULTS] = load_tracking_run_instances(run_id)
        st.session_state[VIEW_RUN_ID] = run_id
    else:
        st.session_state[RESULTS] = {}
        st.session_state[VIEW_RUN_ID] = None

# プロンプトに使える Detection2D run
prompt_runs = list_prompt_detection_runs(dataset_id, scene_token)
if not prompt_runs:
    st.warning(
        "プロンプトに使える 2D 検出結果がありません。"
        "先に 2D Object Detection を実行してください。"
    )
    # st.page_link はページ未登録時に KeyError を投げるため、state 側の
    # ラッパーを通す（単体テストや構成変更で落とさないため）
    S.safe_page_link("pages/2_Detection2D.py", "2D Object Detection へ", "🔍")
    st.stop()

param_col, map_col = st.columns([2, 1])

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
with param_col:
    with st.expander("Inference Parameters", expanded=False):
        st.markdown("**Box Prompt**")

        def _prompt_label(run: dict) -> str:
            return (f"{run['started_at'].strftime('%m-%d %H:%M:%S')}  "
                    f"{run['nbr_boxes']} boxes  interval={run['sample_interval']}")

        prompt_ids = [r["id"] for r in prompt_runs]
        prompt_labels = {r["id"]: _prompt_label(r) for r in prompt_runs}
        # 既定は started_at が最新のもの（list は新しい順で返る）
        selected_prompt_id = st.radio(
            "Box Prompt", prompt_ids, index=0,
            format_func=lambda rid: prompt_labels[rid],
            label_visibility="collapsed",
        )
        selected_prompt = next(r for r in prompt_runs if r["id"] == selected_prompt_id)

        sweeps_col, iou_col, match_col = st.columns(3)

        with sweeps_col:
            st.markdown("**Sweeps per Sample**")
            num_sweeps = st.number_input(
                "Sweeps per Sample",
                min_value=1,
                max_value=min(settings.SWEEPS_PER_SAMPLE, MAX_SWEEPS_PER_SAMPLE),
                value=settings.DEFAULT_TRACKING_NUM_SWEEPS,
                step=1, label_visibility="collapsed",
            )
            st.caption("1 ならキーフレームのみ。非キーフレームは伝播にのみ使う")

        with iou_col:
            st.markdown("**IoU Threshold**")
            iou_threshold = st.slider(
                "IoU Threshold", 0.0, 1.0,
                value=settings.DEFAULT_TRACKING_IOU_THRESHOLD, step=0.05,
                label_visibility="collapsed",
            )
            st.markdown("**IoU Method**")
            iou_method = st.selectbox(
                "IoU Method", IOU_METHODS,
                index=IOU_METHODS.index(settings.DEFAULT_TRACKING_IOU_METHOD),
                label_visibility="collapsed",
            )

        with match_col:
            st.markdown("**IoU Label Match**")
            iou_label_match = st.selectbox(
                "IoU Label Match", IOU_LABEL_MATCHES,
                index=IOU_LABEL_MATCHES.index(
                    settings.DEFAULT_TRACKING_IOU_LABEL_MATCH
                ),
                label_visibility="collapsed",
            )
            st.caption(
                "区間の境界で、伝播したインスタンスと"
                "新しいプロンプトのインスタンスを照合する条件"
            )

# ------------------------------------------------------------------
# Run / progress
# ------------------------------------------------------------------


def _merge_partial(items: list[dict]) -> None:
    store = st.session_state[RESULTS]
    for item in items:
        store[item["sample_data_token"]] = item.get("instances", [])


def _save_completed_job(job: dict) -> None:
    """完了したジョブを 1 run として DB に保存する."""
    try:
        params_id = save_tracking_run(
            dataset_id, scene_token,
            job=job,
            instances_by_frame=st.session_state[RESULTS],
            detection_run_id=selected_prompt_id,
            sample_interval=selected_prompt["sample_interval"],
            num_sweeps=int(num_sweeps),
            iou_threshold=float(iou_threshold),
            iou_method=iou_method,
            iou_label_match=iou_label_match,
            mask_score_threshold=settings.DEFAULT_TRACKING_MASK_SCORE_THRESHOLD,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"トラッキング結果の保存に失敗しました: {exc}")
        return

    st.session_state[VIEW_RUN_ID] = params_id
    clear_caches()
    st.session_state[RESULTS] = load_tracking_run_instances(params_id)


with param_col:
    with st.container(border=True):
        st.caption(
            f"prompt={selected_prompt['nbr_boxes']} boxes / "
            f"interval={selected_prompt['sample_interval']} / "
            f"sweeps={num_sweeps} / iou={iou_threshold:g} {iou_method} "
            f"{iou_label_match}"
        )
        run_col, cancel_col, status_col = st.columns([1, 1, 3])

        with run_col:
            if st.button("Run Inference", type="primary",
                         disabled=bool(st.session_state.get(JOB_ID))):
                if not is_available():
                    st.error("推論サーバーに接続できません。")
                else:
                    try:
                        payload = build_tracking_payload(
                            dataset_id, scene_token, dataset["dataroot"],
                            detection_run_id=selected_prompt_id,
                            num_sweeps=int(num_sweeps),
                            iou_threshold=float(iou_threshold),
                            iou_method=iou_method,
                            iou_label_match=iou_label_match,
                            mask_score_threshold=(
                                settings.DEFAULT_TRACKING_MASK_SCORE_THRESHOLD
                            ),
                            stub_delay_sec=settings.TRACKING_STUB_DELAY_SEC,
                        )
                        job = submit_instance_tracking(payload)
                        S.set_selection(JOB_ID, job["job_id"])
                        st.session_state[RESULTS] = {}
                        st.session_state[PARTIAL_SINCE] = 0
                        st.rerun()
                    except (InferenceServerError, ValueError) as exc:
                        st.error(str(exc))

        with cancel_col:
            if st.session_state.get(JOB_ID):
                if st.button("Cancel"):
                    try:
                        cancel_instance_tracking_job(st.session_state[JOB_ID])
                    except InferenceServerError as exc:
                        st.warning(str(exc))

        @st.fragment(run_every=1.0)
        def progress_area() -> None:
            job_id = st.session_state.get(JOB_ID)
            if not job_id:
                done = len(st.session_state[RESULTS])
                if done:
                    st.caption(f"トラッキング結果: {done} フレーム分を保持中")
                return

            try:
                job = get_instance_tracking_job(
                    job_id, since=st.session_state[PARTIAL_SINCE]
                )
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
                # 二重保存の防止（ポーリングは1秒ごとに走る）
                if st.session_state.get(SAVED_JOB_ID) != job_id:
                    st.session_state[SAVED_JOB_ID] = job_id
                    _save_completed_job(job)
                st.session_state.pop(JOB_ID, None)
                if job["status"] == "failed":
                    st.error(job.get("error") or "推論に失敗しました")
                st.rerun()

        with status_col:
            progress_area()

# ------------------------------------------------------------------
# Saved runs
# ------------------------------------------------------------------
with param_col:
    runs = list_tracking_runs(dataset_id, scene_token)
    if runs:
        with st.expander(f"保存済みの推論結果（{len(runs)} 件）", expanded=False):
            current = st.session_state.get(VIEW_RUN_ID)

            def _run_label(run: dict) -> str:
                mark = "★ " if run["id"] == current else ""
                started = run["started_at"].strftime("%m-%d %H:%M:%S")
                refs = (f" / depth参照 {run['nbr_depth_runs']}"
                        if run["nbr_depth_runs"] else "")
                return (f"{mark}{started}  [{run['status']}]  "
                        f"{run['nbr_instances']} inst / {run['num_tracks']} tracks  "
                        f"sweeps={run['num_sweeps']}{refs}")

            options = [r["id"] for r in runs]
            labels = {r["id"]: _run_label(r) for r in runs}
            selected_run = st.radio(
                "表示する run", options,
                index=options.index(current) if current in options else 0,
                format_func=lambda rid: labels[rid],
                key="_w_tracking_run_select",
            )
            show_col, delete_col = st.columns(2)
            with show_col:
                if st.button("この run を表示", width="content",
                             disabled=selected_run == current):
                    st.session_state[VIEW_RUN_ID] = selected_run
                    st.session_state[RESULTS] = load_tracking_run_instances(
                        selected_run
                    )
                    st.rerun()
            with delete_col:
                target = next(r for r in runs if r["id"] == selected_run)
                if target["nbr_depth_runs"]:
                    st.warning(
                        f"この run は Depth Estimation {target['nbr_depth_runs']} 件から"
                        "参照されています。削除すると連鎖して消えます。"
                    )
                if st.button("削除", width="content"):
                    delete_run(selected_run)
                    if current == selected_run:
                        st.session_state.pop(RESULTS, None)
                        st.session_state.pop(VIEW_RUN_ID, None)
                    clear_caches()
                    st.rerun()

# ------------------------------------------------------------------
# Sample selection (param_col の一番下)
# ------------------------------------------------------------------
with param_col:
    # Detection2D と違い、全 sample を選べる。
    # トラッキングは interval の間の sample にも結果が出るため
    selected_sample_idx = st.select_slider(
        "Select Sample",
        options=list(range(len(samples))) or [0],
        value=len(samples) // 2 if samples else 0,
    )

# ------------------------------------------------------------------
# Scene map (map_col)
# ------------------------------------------------------------------
with map_col:
    render_scene_waypoint_view(
        dataset_id, dataset["dataroot"], scene["token"],
        title=scene["name"],
        highlight_index=selected_sample_idx,
        height=320,
        show_sample_info=False,
    )

# ------------------------------------------------------------------
# Instance view
# ------------------------------------------------------------------
st.divider()

view_col, opt_col = st.columns([9, 1])

selected_sample = samples[selected_sample_idx]
results = st.session_state[RESULTS]

# 表示中の run が使ったプロンプト（Prompt 表示モード用）
view_run_id = st.session_state.get(VIEW_RUN_ID)
prompt_boxes_by_frame: dict[str, list[dict]] = {}
if view_run_id:
    run_info = next((r for r in runs if r["id"] == view_run_id), None)
    if run_info and run_info.get("detection_2d_params_id"):
        prompt_boxes_by_frame = load_detection_run_boxes(
            run_info["detection_2d_params_id"]
        )

items = []
for sensor in cam_sensors:
    image, frame = get_keyframe_image(
        dataset_id, dataset["dataroot"], selected_sample["token"], sensor["token"]
    )
    token = frame["token"] if frame else None
    items.append({
        "channel": sensor["channel"],
        "image": image,
        "instances": results.get(token, []),
        "prompt_boxes": prompt_boxes_by_frame.get(token, []),
        "pending": token not in results,
    })

with opt_col:
    S.init_sticky(W_BOX, OPT_BOX, BOX_MODE_PROMPT)
    box_mode = st.radio(
        "Show boxes", BOX_MODES, key=W_BOX,
        on_change=S.sync_sticky, args=(W_BOX, OPT_BOX),
    )
    S.init_sticky(W_COLOR, OPT_COLOR, COLOR_MODE_LABEL)
    color_mode = st.radio(
        "Color", COLOR_MODES, key=W_COLOR,
        on_change=S.sync_sticky, args=(W_COLOR, OPT_COLOR),
    )
    S.init_sticky(W_TEXT, OPT_TEXT, TEXT_MODE_TRACK)
    text_mode = st.radio(
        "Instance text", INSTANCE_TEXT_MODES, key=W_TEXT,
        on_change=S.sync_sticky, args=(W_TEXT, OPT_TEXT),
    )

    # 凡例は Color の選択に合わせて、ラベル別か Track ID 別かが切り替わる。
    # キーの体系が変わるのでチェック状態も別管理にする（key_prefix を分ける）
    keys, counts = legend_entries(items, color_mode)
    if keys:
        enabled_keys = render_label_legend(
            keys,
            counts=counts,
            key_prefix=(
                "tracking_legend_track" if color_mode == COLOR_MODE_TRACK
                else "tracking_legend_label"
            ),
            color_fn=(
                color_for_track if color_mode == COLOR_MODE_TRACK
                else color_for_label
            ),
            header=("Track IDs" if color_mode == COLOR_MODE_TRACK else "Labels"),
        )
    else:
        enabled_keys = None
        st.caption("このサンプルには表示できるインスタンスがありません")

with view_col:
    render_instance_grid(
        items,
        columns=2,
        color_mode=color_mode,
        box_mode=box_mode,
        text_mode=text_mode,
        enabled_keys=enabled_keys,
    )

S.render_selection_sidebar(dataset_name=dataset["name"], scene_name=scene["name"])
