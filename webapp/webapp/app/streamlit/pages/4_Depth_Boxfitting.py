"""Depth Estimation & Box Fitting ページ.

構成は他ステップを踏襲しつつ、設定と表示をタブで切り替える:

  Inference Parameters … General / Depth Estimation / LiDAR Pointcloud / Box Fitting
  表示                  … Depth Estimation / PointCloud / Box Fitting

推論の前に run を作成して id を確定させる点が他ステップと違う。
深度マップと LiDAR 点群は推論サーバーが DERIVED_ROOT へ直接書くため、
書き込み先ディレクトリ名になる params_id を先に決める必要がある。
"""
from __future__ import annotations

import json

import numpy as np
import streamlit as st

from app.core.config import get_settings
from app.services.camera_service import order_cameras
from app.services.depth_boxfitting_service import (
    build_boxfitting_payload,
    create_pending_run,
    delete_run,
    finalize_run,
    resolve_display_run,
)
from app.services.depth_pointcloud import depth_to_ego_points
from app.services.gt_bev import gt_boxes_in_ego
from app.services.derived_file_service import load_depth_map, load_lidar_pointcloud
from app.services.frame_image import get_keyframe_image  # noqa: F401  (将来の重ね表示用)
from app.services.inference_client import (
    InferenceServerError,
    cancel_boxfitting_job,
    get_boxfitting_job,
    is_available,
    submit_boxfitting,
)
from app.streamlit import state as S
from app.streamlit.components.depth_viewer import colorize_depth, render_depth_grid
from app.streamlit.components.det2d_viewer import render_label_legend
from app.streamlit.components.det2d_viewer import color_for as color_for_label
from app.streamlit.components.instance_tracking_viewer import (
    COLOR_MODE_LABEL,
    COLOR_MODE_TRACK,
    COLOR_MODES,
    INSTANCE_TEXT_MODES,
    TEXT_MODE_NONE,
    color_for_track,
    draw_instances,
    preferred_instances,
)
from app.streamlit.components.bev_viewer import (
    build_bev_figure,
    combined_axis_range,
    render_bev,
)
from app.streamlit.components.pointcloud_viewer import (
    CAMERA_PRESETS,
    VIEWS,
    VIEW_GLOBAL,
    build_pointcloud_figure,
    global_camera,
    group_instance_points,
    render_pointcloud,
)
from app.streamlit.components.waypoint_viewer import render_scene_waypoint_view
from app.streamlit.data_access import (
    clear_caches,
    list_gt_annotations,
    refilter_instance_points,
    get_dataset,
    get_scene,
    list_boxfitting_runs,
    list_frames_by_scene,
    list_input_tracking_runs,
    list_samples,
    list_sensors,
    load_box_fittings,
    load_depth_estimations,
    load_lidar_pointclouds,
    load_tracking_run_instances,
)
from common.mask_rle import rle_bbox, resize_rle

dataset_id, scene_token = S.require_scene()
dataset = get_dataset(dataset_id)
scene = get_scene(dataset_id, scene_token)
samples = list_samples(dataset_id, scene_token)
cam_sensors = order_cameras(list_sensors(dataset_id, modality="camera"))
settings = get_settings()

st.subheader("Depth Estimation & Box Fitting by Depth-Anything-3")

JOB_ID = S.BOXFIT_JOB_ID
PENDING_RUN = S.BOXFIT_PENDING_RUN_ID
RESULTS = S.BOXFIT_RESULTS
PARTIAL_SINCE = S.BOXFIT_PARTIAL_SINCE
SAVED_JOB_ID = S.BOXFIT_SAVED_JOB_ID
VIEW_RUN_ID = S.BOXFIT_VIEW_RUN_ID

st.session_state.setdefault(PARTIAL_SINCE, 0)
st.session_state.setdefault(RESULTS, {"depth": [], "lidar": [], "box": []})

# 表示オプション（Run Inference の st.rerun をまたいで保持する）
OPT_SHOW_MASKS, W_SHOW_MASKS = "boxfit_show_masks", "_w_boxfit_show_masks"
OPT_MASK_COLOR, W_MASK_COLOR = "boxfit_mask_color", "_w_boxfit_mask_color"
OPT_INST_TEXT, W_INST_TEXT = "boxfit_inst_text", "_w_boxfit_inst_text"
OPT_PC_COLOR, W_PC_COLOR = "boxfit_pc_color", "_w_boxfit_pc_color"
OPT_PC_POINTS, W_PC_POINTS = "boxfit_pc_points", "_w_boxfit_pc_points"
# 点群ビューの視点。ボタンで切り替え、押すたびに版番号を進めて
# Plotly 側の uirevision を変える（同じ値のままだと視点が更新されない）
PC_VIEW = "boxfit_pc_view"
PC_VIEW_REVISION = "boxfit_pc_view_rev"
# Apply で確定した再フィルタのパラメータ。None なら run 保存時の点群を使う
PC_REFILTER_PARAMS = "boxfit_refilter_params"
# Box Fitting タブの表示オプション
OPT_BF_COLOR, W_BF_COLOR = "boxfit_bf_color", "_w_boxfit_bf_color"
W_SAMPLE = "_w_boxfit_sample"

MASK_MODES = ("None", "Original", "Closed")

# インスタンス点群を、外れ値除去の前後どちらで見るか。
#
# NOTE: フィルタ前の点群は DB に保存していない（深度マップと
# クロージング後マスクから再生成できるため）。再フィルタ用の
# エンドポイントを実装するまで Raw は選べない。
POINTS_MODE_RAW = "Raw"
POINTS_MODE_FILTERED = "ROR/DBSCAN"
POINTS_MODES = (POINTS_MODE_RAW, POINTS_MODE_FILTERED)
# 表示モード → BoxFitting3D のカラム名
POINTS_COLUMNS = {
    POINTS_MODE_FILTERED: ("points_depth_ego", "points_lidar_ego"),
}

# --- 初期表示: 保存済みの run --------------------------------------------------
if VIEW_RUN_ID not in st.session_state:
    run_id, _reason = resolve_display_run(dataset_id, scene_token)
    st.session_state[VIEW_RUN_ID] = run_id

tracking_runs = list_input_tracking_runs(dataset_id, scene_token)
if not tracking_runs:
    st.warning(
        "入力に使える Instance Tracking の結果がありません。"
        "先に Instance Tracking を実行してください。"
    )
    S.safe_page_link("pages/3_Instance_Tracking.py", "Instance Tracking へ", "🎯")
    st.stop()

param_col, map_col = st.columns([2, 1])

# ------------------------------------------------------------------
# Inference Parameters
# ------------------------------------------------------------------
with param_col:
    with st.expander("Inference Parameters", expanded=False):
        # Instance Tracking の選択だけはタブの外に出す。
        # どのマスクを使うかは全タブに影響する前提条件のため
        st.markdown("**Instance Tracking**")

        def _tracking_label(run: dict) -> str:
            return (f"{run['started_at'].strftime('%m-%d %H:%M:%S')}  "
                    f"{run['nbr_instances']} inst / {run['num_tracks']} tracks  "
                    f"sweeps={run['num_sweeps']}")

        tracking_ids = [r["id"] for r in tracking_runs]
        tracking_labels = {r["id"]: _tracking_label(r) for r in tracking_runs}
        selected_tracking_id = st.radio(
            "Instance Tracking", tracking_ids, index=0,
            format_func=lambda rid: tracking_labels[rid],
            label_visibility="collapsed",
        )

        general_tab, depth_tab, lidar_tab, fitting_tab = st.tabs(
            ["General", "Depth Estimation", "LiDAR Pointcloud", "Box Fitting"]
        )

        with general_tab:
            # LiDAR の統合・地面除去・混合は未実装。
            # 実装するまでチェックできないようにしておく
            use_lidar = st.checkbox(
                "Use LiDAR", value=False, disabled=True,
                help="LiDAR を使った点群の混合は未実装です（深度推定のみで動作します）",
            )
            st.markdown("**Mask Closing**")
            mask_dilation = st.slider(
                "Dilation", 1, settings.MASK_DILATION_MAX,
                value=settings.MASK_DILATION_DEFAULT, step=1,
            )
            mask_erosion = st.slider(
                "Erosion", 1, settings.MASK_EROSION_MAX,
                value=settings.MASK_EROSION_DEFAULT, step=1,
            )

        with depth_tab:
            st.markdown("**ROR**")
            depth_ror_nb = st.slider(
                "nb_points", 1, settings.DEPTH_ROR_NB_POINTS_MAX,
                value=settings.DEPTH_ROR_NB_POINTS_DEFAULT, step=1,
                key="_w_depth_ror_nb",
            )
            depth_ror_radius = st.slider(
                "Radius", 0.1, settings.DEPTH_ROR_RADIUS_MAX,
                value=settings.DEPTH_ROR_RADIUS_DEFAULT, step=0.1,
                key="_w_depth_ror_radius",
            )
            st.markdown("**DBSCAN**")
            depth_dbscan_eps = st.slider(
                "eps", 0.1, settings.DEPTH_DBSCAN_EPS_MAX,
                value=settings.DEPTH_DBSCAN_EPS_DEFAULT, step=0.1,
                key="_w_depth_dbscan_eps",
            )
            depth_dbscan_min = st.slider(
                "min_samples", 2, settings.DEPTH_DBSCAN_MIN_SAMPLES_MAX,
                value=settings.DEPTH_DBSCAN_MIN_SAMPLES_DEFAULT, step=1,
                key="_w_depth_dbscan_min",
            )

        with lidar_tab:
            num_lidar_sweeps = st.number_input(
                "LiDAR Sweeps", min_value=1, max_value=settings.SWEEPS_PER_SAMPLE,
                value=settings.LIDAR_NUM_SWEEPS_DEFAULT, step=1,
            )
            min_lidar_points = st.slider(
                "Min LiDAR Points", 1, settings.LIDAR_MIN_POINTS_MAX,
                value=settings.LIDAR_MIN_POINTS_DEFAULT, step=1,
                help="これ未満のインスタンスは深度点群だけでフィッティングする",
            )
            st.markdown("**ROR**")
            lidar_ror_nb = st.slider(
                "nb_points", 1, settings.LIDAR_ROR_NB_POINTS_MAX,
                value=settings.LIDAR_ROR_NB_POINTS_DEFAULT, step=1,
                key="_w_lidar_ror_nb",
            )
            lidar_ror_radius = st.slider(
                "Radius", 0.1, settings.LIDAR_ROR_RADIUS_MAX,
                value=settings.LIDAR_ROR_RADIUS_DEFAULT, step=0.1,
                key="_w_lidar_ror_radius",
            )
            st.markdown("**DBSCAN**")
            lidar_dbscan_eps = st.slider(
                "eps", 0.1, settings.LIDAR_DBSCAN_EPS_MAX,
                value=settings.LIDAR_DBSCAN_EPS_DEFAULT, step=0.1,
                key="_w_lidar_dbscan_eps",
            )
            lidar_dbscan_min = st.slider(
                "min_samples", 2, settings.LIDAR_DBSCAN_MIN_SAMPLES_MAX,
                value=settings.LIDAR_DBSCAN_MIN_SAMPLES_DEFAULT, step=1,
                key="_w_lidar_dbscan_min",
            )

        with fitting_tab:
            boxfit_method = st.selectbox(
                "Method", settings.BOXFIT_METHODS,
                index=settings.BOXFIT_METHODS.index(settings.BOXFIT_METHOD_DEFAULT)
                if settings.BOXFIT_METHOD_DEFAULT in settings.BOXFIT_METHODS else 0,
                help=("convex_hull_moa: BEV の凸包に対し、センサーから見た"
                      "オクルージョン面積が最小になる向きを探す"),
            )
            angle_step_deg = st.slider(
                "angle_step_deg",
                settings.BOXFIT_ANGLE_STEP_DEG_MIN,
                settings.BOXFIT_ANGLE_STEP_DEG_MAX,
                value=settings.BOXFIT_ANGLE_STEP_DEG_DEFAULT, step=0.1,
                help="向きの探索刻み。細かいほど遅くなる",
            )
            z_percentiles = st.slider(
                "z_percentiles", 0.0, 100.0,
                value=(settings.BOXFIT_Z_PERCENTILE_LOW_DEFAULT,
                       settings.BOXFIT_Z_PERCENTILE_HIGH_DEFAULT),
                step=0.5,
                help=("高さを決めるパーセンタイル。0/100 にすると"
                      "はみ出した 1 点で箱が縦に伸びる"),
            )

mask_params = {"dilation": int(mask_dilation), "erosion": int(mask_erosion)}
box_fitting_params = {
    "method": boxfit_method,
    "angle_step_deg": float(angle_step_deg),
    "z_percentiles": [float(z_percentiles[0]), float(z_percentiles[1])],
}
depth_params = {
    "ror_nb_points": int(depth_ror_nb), "ror_radius": float(depth_ror_radius),
    "dbscan_eps": float(depth_dbscan_eps), "dbscan_min_samples": int(depth_dbscan_min),
}
lidar_params = {
    "min_points": int(min_lidar_points),
    "ror_nb_points": int(lidar_ror_nb), "ror_radius": float(lidar_ror_radius),
    "dbscan_eps": float(lidar_dbscan_eps), "dbscan_min_samples": int(lidar_dbscan_min),
}

# ------------------------------------------------------------------
# Run / progress
# ------------------------------------------------------------------


def _merge_partial(items: list[dict]) -> None:
    """部分結果を種別ごとに積む."""
    store = st.session_state[RESULTS]
    for item in items:
        store.setdefault(item["kind"], []).append(item["data"])


def _save_completed_job(job: dict) -> None:
    """完了したジョブを、作成済みの run へ書き込む."""
    params_id = st.session_state.get(PENDING_RUN)
    if not params_id:
        st.error("保存先の run が見つかりません。")
        return
    store = st.session_state[RESULTS]
    try:
        created = finalize_run(
            dataset_id, scene_token, params_id, job=job,
            depth_estimations=store.get("depth", []),
            lidar_pointclouds=store.get("lidar", []),
            box_fittings=store.get("box", []),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"結果の保存に失敗しました: {exc}")
        return

    st.session_state[VIEW_RUN_ID] = params_id
    st.session_state.pop(PENDING_RUN, None)
    clear_caches()
    if created:
        st.toast(f"3D ボックスを {created} 件生成しました")


with param_col:
    with st.container(border=True):
        st.caption(
            f"tracking={next(r['nbr_instances'] for r in tracking_runs if r['id']==selected_tracking_id)} inst / "
            f"use_lidar={use_lidar} / sweeps={num_lidar_sweeps} / "
            f"closing={mask_dilation}-{mask_erosion}"
        )
        run_col, cancel_col, status_col = st.columns([1, 1, 3])

        with run_col:
            if st.button("Run Inference", type="primary",
                         disabled=bool(st.session_state.get(JOB_ID))):
                if not is_available():
                    st.error("推論サーバーに接続できません。")
                else:
                    try:
                        # 深度マップと LiDAR の書き込み先を決めるため、
                        # 推論を投げる前に run を作って id を確定させる
                        params_id = create_pending_run(
                            dataset_id, scene_token,
                            tracking_run_id=selected_tracking_id,
                            use_lidar=bool(use_lidar),
                            num_lidar_sweeps=int(num_lidar_sweeps),
                            mask_params=mask_params, depth_params=depth_params,
                            lidar_params=lidar_params,
                            box_fitting_params=box_fitting_params,
                        )
                        payload = build_boxfitting_payload(
                            dataset_id, scene_token, dataset["dataroot"],
                            params_id=params_id,
                            tracking_run_id=selected_tracking_id,
                            use_lidar=bool(use_lidar),
                            num_lidar_sweeps=int(num_lidar_sweeps),
                            mask_params=mask_params, depth_params=depth_params,
                            lidar_params=lidar_params,
                            box_fitting_params=box_fitting_params,
                            stub_delay_sec=settings.BOXFIT_STUB_DELAY_SEC,
                        )
                        job = submit_boxfitting(payload)
                        S.set_selection(JOB_ID, job["job_id"])
                        st.session_state[PENDING_RUN] = params_id
                        st.session_state[RESULTS] = {"depth": [], "lidar": [], "box": []}
                        st.session_state[PARTIAL_SINCE] = 0
                        st.rerun()
                    except (InferenceServerError, ValueError) as exc:
                        st.error(str(exc))

        with cancel_col:
            if st.session_state.get(JOB_ID):
                if st.button("Cancel"):
                    try:
                        cancel_boxfitting_job(st.session_state[JOB_ID])
                    except InferenceServerError as exc:
                        st.warning(str(exc))

        @st.fragment(run_every=1.0)
        def progress_area() -> None:
            job_id = st.session_state.get(JOB_ID)
            if not job_id:
                store = st.session_state[RESULTS]
                if any(store.values()):
                    st.caption(
                        f"depth {len(store['depth'])} / lidar {len(store['lidar'])} "
                        f"/ box {len(store['box'])}"
                    )
                return
            try:
                job = get_boxfitting_job(
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
            st.caption(f"status={job['status']} / {job['elapsed_sec']:.1f}s")

            if job["status"] in ("succeeded", "failed", "cancelled"):
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
    runs = list_boxfitting_runs(dataset_id, scene_token)
    if runs:
        with st.expander(f"保存済みの推論結果（{len(runs)} 件）", expanded=False):
            current = st.session_state.get(VIEW_RUN_ID)

            def _run_label(run: dict) -> str:
                mark = "★ " if run["id"] == current else ""
                return (f"{mark}{run['started_at'].strftime('%m-%d %H:%M:%S')}  "
                        f"[{run['status']}]  {run['nbr_fitted']}/{run['nbr_box_fittings']} fitted  "
                        f"ann {run['nbr_annotations']}  lidar={run['use_lidar']}")

            options = [r["id"] for r in runs]
            labels = {r["id"]: _run_label(r) for r in runs}
            selected_run = st.radio(
                "表示する run", options,
                index=options.index(current) if current in options else 0,
                format_func=lambda rid: labels[rid], key="_w_boxfit_run_select",
            )
            show_col, delete_col = st.columns(2)
            with show_col:
                if st.button("この run を表示", width="content",
                             disabled=selected_run == current):
                    st.session_state[VIEW_RUN_ID] = selected_run
                    st.rerun()
            with delete_col:
                target = next(r for r in runs if r["id"] == selected_run)
                if target["nbr_annotations"]:
                    st.warning(
                        f"この run は 3D アノテーションを {target['nbr_annotations']} 件"
                        "生成しています。削除すると一緒に消えます。"
                    )
                if st.button("削除", width="content"):
                    delete_run(selected_run)
                    if current == selected_run:
                        st.session_state.pop(VIEW_RUN_ID, None)
                    clear_caches()
                    st.rerun()

# ------------------------------------------------------------------
# Sample selection / map
# ------------------------------------------------------------------
with param_col:
    sample_options = list(range(len(samples))) or [0]
    if (W_SAMPLE not in st.session_state
            or st.session_state[W_SAMPLE] not in sample_options):
        st.session_state[W_SAMPLE] = sample_options[len(sample_options) // 2]
    selected_sample_idx = st.select_slider(
        "Select Sample", options=sample_options, key=W_SAMPLE,
    )

with map_col:
    render_scene_waypoint_view(
        dataset_id, dataset["dataroot"], scene["token"],
        title=scene["name"], highlight_index=selected_sample_idx,
        height=320, show_sample_info=False,
    )

# ------------------------------------------------------------------
# 表示用データの準備
# ------------------------------------------------------------------
st.divider()

view_run_id = st.session_state.get(VIEW_RUN_ID)
selected_sample = samples[selected_sample_idx]

depth_info = load_depth_estimations(view_run_id) if view_run_id else {}
lidar_info = load_lidar_pointclouds(view_run_id) if view_run_id else {}

# このサンプルのカメラフレーム
frames = [
    f for f in list_frames_by_scene(dataset_id, scene_token)
    if f["modality"] == "camera" and f["sample_token"] == selected_sample["token"]
]
frames_by_channel = {f["channel"]: f for f in frames}
frame_tokens = tuple(f["token"] for f in frames)

depth_tab_view, pointcloud_tab_view, fitting_tab_view = st.tabs(
    ["Depth Estimation", "PointCloud", "Box Fitting"]
)

# ------------------------------------------------------------------
# Depth Estimation タブ
# ------------------------------------------------------------------
with depth_tab_view:
    view_col, opt_col = st.columns([8, 1])

    with opt_col:
        S.init_sticky(W_SHOW_MASKS, OPT_SHOW_MASKS, MASK_MODES[0])
        show_masks = st.radio(
            "Show masks", MASK_MODES, key=W_SHOW_MASKS,
            on_change=S.sync_sticky, args=(W_SHOW_MASKS, OPT_SHOW_MASKS),
        )
        S.init_sticky(W_MASK_COLOR, OPT_MASK_COLOR, COLOR_MODE_LABEL)
        mask_color_mode = st.radio(
            "Mask Color", COLOR_MODES, key=W_MASK_COLOR,
            on_change=S.sync_sticky, args=(W_MASK_COLOR, OPT_MASK_COLOR),
        )
        S.init_sticky(W_INST_TEXT, OPT_INST_TEXT, TEXT_MODE_NONE)
        inst_text_mode = st.radio(
            "Instance text", INSTANCE_TEXT_MODES, key=W_INST_TEXT,
            on_change=S.sync_sticky, args=(W_INST_TEXT, OPT_INST_TEXT),
        )

    # マスクの供給元。Original はトラッキング結果、Closed は Box Fitting の保存値。
    #
    # NOTE: トラッキングのマスクは**元画像の解像度**（1600x900）だが、
    # 深度画像は DA3 の出力解像度（例 800x450）。
    # そのまま重ねると draw_instances が解像度不一致でスキップし、
    # 何も表示されない。深度側の解像度へ合わせてから渡すこと。
    depth_size_by_frame = {
        token: (info["depth_height"], info["depth_width"])
        for token, info in depth_info.items()
        if info.get("depth_height") and info.get("depth_width")
    }

    def _fit_to_depth(token: str, instances: list[dict]) -> list[dict]:
        """マスクと外接矩形を深度画像の解像度へ合わせる."""
        size = depth_size_by_frame.get(token)
        if size is None:
            return []
        height, width = size
        resolved = []
        for inst in instances:
            rle = inst.get("mask_rle")
            if not rle:
                continue
            resized = resize_rle(rle, height, width)
            xmin, ymin, xmax, ymax = rle_bbox(resized)
            resolved.append({
                **inst, "mask_rle": resized,
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            })
        return resolved

    instances_by_frame: dict[str, list[dict]] = {}
    if show_masks == "Original":
        run = next((r for r in runs if r["id"] == view_run_id), None) if view_run_id else None
        tracking_id = run["instance_tracking_2d_params_id"] if run else None
        if tracking_id:
            raw = load_tracking_run_instances(tracking_id)
            instances_by_frame = {
                token: _fit_to_depth(token, preferred_instances(insts))
                for token in frame_tokens
                if (insts := raw.get(token))
            }
    elif show_masks == "Closed" and view_run_id:
        fittings = load_box_fittings(
            view_run_id, frame_tokens, include_mask=True
        )
        for token, items in fittings.items():
            # クロージング後のマスクは深度解像度で保存されているが、
            # 保存倍率を変えた過去 run もありうるので同じ経路で合わせる
            instances_by_frame[token] = _fit_to_depth(token, [
                {**fit, "mask_rle": fit.get("mask_rle_closed")} for fit in items
            ])

    items = []
    for sensor in cam_sensors:
        frame = frames_by_channel.get(sensor["channel"])
        info = depth_info.get(frame["token"]) if frame else None
        image = None
        if info and info.get("depth_path"):
            depth = load_depth_map(info["depth_path"])
            if depth is not None:
                image = colorize_depth(depth)
        items.append({
            "channel": sensor["channel"],
            "image": image,
            "instances": (
                instances_by_frame.get(frame["token"], []) if frame else []
            ),
        })

    with opt_col:
        keys = sorted({
            (str(i["track_id"]) if mask_color_mode == COLOR_MODE_TRACK
             else str(i["label"]))
            for item in items for i in item["instances"]
        })
        if keys and show_masks != "None":
            enabled_keys = render_label_legend(
                keys,
                key_prefix=("boxfit_legend_track"
                            if mask_color_mode == COLOR_MODE_TRACK
                            else "boxfit_legend_label"),
                color_fn=(color_for_track
                          if mask_color_mode == COLOR_MODE_TRACK
                          else color_for_label),
                header=("Track IDs" if mask_color_mode == COLOR_MODE_TRACK
                        else "Labels"),
            )
        else:
            enabled_keys = None

    def _draw(image, instances):
        shown = [
            i for i in instances
            if enabled_keys is None or (
                (str(i["track_id"]) if mask_color_mode == COLOR_MODE_TRACK
                 else str(i["label"])) in enabled_keys
            )
        ]
        return draw_instances(
            image, shown, color_mode=mask_color_mode,
            box_mode="None", text_mode=inst_text_mode,
        )

    with view_col:
        if not view_run_id:
            st.info("保存済みの推論結果がありません。Run Inference を実行してください。")
        else:
            render_depth_grid(
                items, columns=2,
                draw_fn=_draw if show_masks != "None" else None,
            )

# ------------------------------------------------------------------
# PointCloud タブ
# ------------------------------------------------------------------
with pointcloud_tab_view:
    view_col, opt_col = st.columns([8, 1])

    with opt_col:
        show_raw_lidar = st.checkbox(
            "Raw LiDAR", value=settings.SHOW_RAW_LIDAR,
            key="_w_pc_raw_lidar",
        )
        show_ground = st.checkbox(
            "LiDAR Ground", value=settings.SHOW_RAW_LIDAR_GROUND,
            key="_w_pc_ground", disabled=not show_raw_lidar,
        )
        run_info = next((r for r in runs if r["id"] == view_run_id), None) if view_run_id else None
        # use_lidar=False の run にはインスタンスごとの LiDAR 点群が無い
        has_lidar_instances = bool(run_info and run_info["use_lidar"])
        show_lidar_instances = st.checkbox(
            "LiDAR Instances", value=has_lidar_instances,
            key="_w_pc_lidar_inst", disabled=not has_lidar_instances,
            help=None if has_lidar_instances else "この run は Use LiDAR=False で実行されています",
        )
        show_raw_depth = st.checkbox(
            "Raw Depth", value=settings.SHOW_RAW_DEPTH_POINTCLOUD,
            key="_w_pc_raw_depth",
        )
        show_depth_instances = st.checkbox(
            "Depth Instances", value=True, key="_w_pc_depth_inst",
        )

        # 保存済みの点群はフィルタ後のみ。Raw を見るには再フィルタが要る
        has_refilter = st.session_state.get(PC_REFILTER_PARAMS) is not None
        points_mode = st.radio(
            "Instance Points", POINTS_MODES, key=W_PC_POINTS,
            on_change=S.sync_sticky, args=(W_PC_POINTS, OPT_PC_POINTS),
            disabled=not has_refilter,
            help=("外れ値除去（ROR / DBSCAN）の適用前後を切り替える。"
                  "下の Filter Params で Apply すると選べるようになる"),
        )

        show_fitted_boxes = st.checkbox(
            "Fitted Boxes", value=False, key="_w_pc_fitted_boxes",
            help=("Box Fitting で推定した 3D ボックスを点群に重ねる。"
                  "色は Instance Color の指定に従う"),
        )

        S.init_sticky(W_PC_COLOR, OPT_PC_COLOR, COLOR_MODE_LABEL)
        pc_color_mode = st.radio(
            "Instance Color", COLOR_MODES, key=W_PC_COLOR,
            on_change=S.sync_sticky, args=(W_PC_COLOR, OPT_PC_COLOR),
        )

        st.markdown("**Cameras**")
        enabled_channels = {
            sensor["channel"] for sensor in cam_sensors
            if st.checkbox(sensor["channel"], value=True,
                           key=f"_w_pc_cam_{sensor['channel']}")
        }

        S.init_sticky(W_PC_POINTS, OPT_PC_POINTS, POINTS_MODE_FILTERED)

    raw_lidar_points = ground_points = raw_depth_points = None
    instance_groups: list[dict] = []
    fitted_boxes: list[dict] = []
    refilter_summary: tuple[int, int, int] | None = None

    if view_run_id:
        info = lidar_info.get(selected_sample["token"])
        if (show_raw_lidar or show_ground) and info:
            loaded = load_lidar_pointcloud(info["pointcloud_path"])
            if loaded is not None:
                points, ground_mask = loaded
                if show_raw_lidar:
                    raw_lidar_points = points[~ground_mask]
                if show_ground:
                    ground_points = points[ground_mask]

        if show_raw_depth:
            chunks = []
            for sensor in cam_sensors:
                if sensor["channel"] not in enabled_channels:
                    continue
                frame = frames_by_channel.get(sensor["channel"])
                meta = depth_info.get(frame["token"]) if frame else None
                if not meta or not meta.get("depth_path"):
                    continue
                depth = load_depth_map(meta["depth_path"])
                if depth is None:
                    continue
                chunks.append(depth_to_ego_points(depth, frame, stride=4))
            if chunks:
                raw_depth_points = np.vstack(chunks)

        if show_depth_instances or show_lidar_instances or show_fitted_boxes:
            # 対象カメラのフレームだけ読む。全フレームぶん点群を読むと重い
            tokens = tuple(
                f["token"] for f in frames if f["channel"] in enabled_channels
            )
            refilter_params = st.session_state.get(PC_REFILTER_PARAMS)

            if refilter_params and tokens:
                # 調整したパラメータで作り直した点群を使う。
                # 同じパラメータならキャッシュが効くので再計算されない
                refiltered = refilter_instance_points(
                    dataset_id, scene_token, view_run_id,
                    selected_sample["token"],
                    json.dumps(refilter_params, sort_keys=True),
                    tuple(sorted(enabled_channels)),
                )
                flat = list(refiltered.values())
                # 再フィルタは点群だけを作り直す。ボックスは run 保存時の
                # ものなので、DB 側から別に読む
                box_source = load_box_fittings(view_run_id, tokens) if (
                    show_fitted_boxes and tokens) else {}
                # 再フィルタの結果はカラム名が違う（DB のものと混ぜない）
                depth_key = (
                    "points_raw_ego" if points_mode == POINTS_MODE_RAW
                    else "points_filtered_ego"
                )
                lidar_key = None
                refilter_summary = (
                    sum(i["num_points_raw"] for i in flat),
                    sum(i["num_points_kept"] for i in flat),
                    len(flat),
                )
            else:
                fittings = load_box_fittings(
                    view_run_id, tokens,
                    include_points=(show_depth_instances or show_lidar_instances),
                ) if tokens else {}
                flat = [fit for items_ in fittings.values() for fit in items_]
                depth_key, lidar_key = POINTS_COLUMNS[POINTS_MODE_FILTERED]
                refilter_summary = None
                box_source = fittings

            if show_depth_instances:
                instance_groups += group_instance_points(
                    flat, color_mode=pc_color_mode, points_key=depth_key
                )
            if show_lidar_instances and lidar_key:
                instance_groups += group_instance_points(
                    flat, color_mode=pc_color_mode, points_key=lidar_key
                )

            if show_fitted_boxes:
                for fit in (f for items_ in box_source.values() for f in items_):
                    if not (fit.get("center_ego") and fit.get("size_wlh")):
                        continue
                    key = (str(fit["track_id"]) if pc_color_mode == COLOR_MODE_TRACK
                           else str(fit["label"]))
                    if enabled_keys is not None and key not in enabled_keys:
                        continue
                    fitted_boxes.append({
                        "key": f"{fit['label']}#{fit['track_id']}",
                        # 点群と同じ色にして、どの点群に当てた箱かを分かるようにする
                        "color": (color_for_track(key)
                                  if pc_color_mode == COLOR_MODE_TRACK
                                  else color_for_label(key)),
                        "center": fit["center_ego"],
                        "size_wlh": fit["size_wlh"],
                        "yaw": fit["yaw_ego"],
                    })

    with view_col:
        # 視点ボタン。既定は Global View（global 座標に対して向きが固定）
        st.session_state.setdefault(PC_VIEW, VIEW_GLOBAL)
        st.session_state.setdefault(PC_VIEW_REVISION, 0)
        view_cols = st.columns(len(VIEWS) + 3)
        for index, view_name in enumerate(VIEWS):
            with view_cols[index]:
                if st.button(view_name, width="stretch",
                             key=f"_w_pc_view_{view_name}"):
                    st.session_state[PC_VIEW] = view_name
                    # 同じ視点を押し直したときも戻せるよう、毎回進める
                    st.session_state[PC_VIEW_REVISION] += 1

        current_view = st.session_state[PC_VIEW]
        if current_view == VIEW_GLOBAL:
            # ego_pose はどのカメラフレームでも同じ sample のものを使う
            ego_pose = next(
                (f["ego_pose"] for f in frames if f.get("ego_pose")), None
            )
            camera = global_camera(
                ego_pose["rotation"],
                settings.DEFAULT_GLOBAL_POINTCLOUD_EYE,
                settings.DEFAULT_GLOBAL_POINTCLOUD_UP,
            ) if ego_pose else None
        else:
            camera = CAMERA_PRESETS.get(current_view)

        # --- 外れ値除去パラメータの調整 -----------------------------------
        # パイプラインを回し直さず、保存済みの深度マップから点群を
        # 作り直して即座に効果を確認する。
        # Plotly の真上に置くのは、Apply の前後で点群の変化を
        # 目線を動かさずに見比べられるようにするため
        with st.expander("Filter Params", expanded=False):
            run_depth = (run_info or {}).get("depth_params") or {}
            run_lidar = (run_info or {}).get("lidar_params") or {}
            # run を切り替えたら既定値も切り替わるよう、key に run を含める
            prefix = f"_w_refilter_{view_run_id}"

            def _slider(label, key, low, high, default, step):
                return st.slider(
                    label, low, high, value=default, step=step,
                    key=f"{prefix}_{key}",
                )

            st.markdown("**LiDAR**")
            lidar_cols = st.columns(4)
            with lidar_cols[0]:
                lidar_ror_nb_new = _slider(
                    "ROR nb_points", "l_ror_nb", 1,
                    settings.LIDAR_ROR_NB_POINTS_MAX,
                    int(run_lidar.get("ror_nb_points",
                                      settings.LIDAR_ROR_NB_POINTS_DEFAULT)), 1)
            with lidar_cols[1]:
                lidar_ror_radius_new = _slider(
                    "ROR Radius", "l_ror_r", 0.1,
                    settings.LIDAR_ROR_RADIUS_MAX,
                    float(run_lidar.get("ror_radius",
                                        settings.LIDAR_ROR_RADIUS_DEFAULT)), 0.1)
            with lidar_cols[2]:
                lidar_eps_new = _slider(
                    "DBSCAN eps", "l_eps", 0.1,
                    settings.LIDAR_DBSCAN_EPS_MAX,
                    float(run_lidar.get("dbscan_eps",
                                        settings.LIDAR_DBSCAN_EPS_DEFAULT)), 0.1)
            with lidar_cols[3]:
                lidar_min_new = _slider(
                    "DBSCAN min_samples", "l_min", 2,
                    settings.LIDAR_DBSCAN_MIN_SAMPLES_MAX,
                    int(run_lidar.get("dbscan_min_samples",
                                      settings.LIDAR_DBSCAN_MIN_SAMPLES_DEFAULT)), 1)

            st.markdown("**Depth**")
            depth_cols = st.columns(4)
            with depth_cols[0]:
                depth_ror_nb_new = _slider(
                    "ROR nb_points", "d_ror_nb", 1,
                    settings.DEPTH_ROR_NB_POINTS_MAX,
                    int(run_depth.get("ror_nb_points",
                                      settings.DEPTH_ROR_NB_POINTS_DEFAULT)), 1)
            with depth_cols[1]:
                depth_ror_radius_new = _slider(
                    "ROR Radius", "d_ror_r", 0.1,
                    settings.DEPTH_ROR_RADIUS_MAX,
                    float(run_depth.get("ror_radius",
                                        settings.DEPTH_ROR_RADIUS_DEFAULT)), 0.1)
            with depth_cols[2]:
                depth_eps_new = _slider(
                    "DBSCAN eps", "d_eps", 0.1,
                    settings.DEPTH_DBSCAN_EPS_MAX,
                    float(run_depth.get("dbscan_eps",
                                        settings.DEPTH_DBSCAN_EPS_DEFAULT)), 0.1)
            with depth_cols[3]:
                depth_min_new = _slider(
                    "DBSCAN min_samples", "d_min", 2,
                    settings.DEPTH_DBSCAN_MIN_SAMPLES_MAX,
                    int(run_depth.get("dbscan_min_samples",
                                      settings.DEPTH_DBSCAN_MIN_SAMPLES_DEFAULT)), 1)
            st.caption("LiDAR 側は点群の混合を実装したあとに効きます")

            apply_col, reset_col, _spacer = st.columns([1, 1, 6])
            with apply_col:
                if st.button("Apply", type="primary", width="stretch",
                             disabled=not view_run_id):
                    st.session_state[PC_REFILTER_PARAMS] = {
                        "depth": {
                            "ror_nb_points": int(depth_ror_nb_new),
                            "ror_radius": float(depth_ror_radius_new),
                            "dbscan_eps": float(depth_eps_new),
                            "dbscan_min_samples": int(depth_min_new),
                        },
                        "lidar": {
                            "ror_nb_points": int(lidar_ror_nb_new),
                            "ror_radius": float(lidar_ror_radius_new),
                            "dbscan_eps": float(lidar_eps_new),
                            "dbscan_min_samples": int(lidar_min_new),
                        },
                    }
                    st.rerun()
            with reset_col:
                if st.button("Reset", width="stretch",
                             disabled=not st.session_state.get(PC_REFILTER_PARAMS)):
                    # run 保存時の点群表示に戻す
                    st.session_state.pop(PC_REFILTER_PARAMS, None)
                    st.rerun()

        if not view_run_id:
            st.info("保存済みの推論結果がありません。Run Inference を実行してください。")
        else:
            fig, counts = build_pointcloud_figure(
                raw_lidar=raw_lidar_points,
                ground=ground_points,
                raw_depth=raw_depth_points,
                instance_groups=instance_groups,
                boxes=fitted_boxes,
                max_points_per_trace=settings.POINTCLOUD_DISPLAY_MAX_POINTS,
                axis_length=settings.POINTCLOUD_AXIS_LENGTH_M,
                camera=camera,
                # Global View は sample ごとに ego の向きが変わるため、
                # sample も版番号に含めて視点を追従させる
                view_revision=(
                    f"{current_view}:{st.session_state[PC_VIEW_REVISION]}"
                    f":{selected_sample_idx if current_view == VIEW_GLOBAL else ''}"
                ),
            )
            render_pointcloud(fig, counts)
            if refilter_summary is not None:
                raw_total, kept_total, n_inst = refilter_summary
                ratio = kept_total / raw_total * 100 if raw_total else 0.0
                st.caption(
                    f"再フィルタ適用中: {n_inst} インスタンス / "
                    f"{raw_total:,} → {kept_total:,} 点 ({ratio:.0f}% 残存)。"
                    "この結果は保存されません（run の記録は実行時のまま）"
                )

# ------------------------------------------------------------------
# Box Fitting タブ
# ------------------------------------------------------------------
with fitting_tab_view:
    bf_view_col, bf_opt_col = st.columns([8, 1])

    with bf_opt_col:
        compare_gt = st.checkbox(
            "Compare with GT", value=False, key="_w_bf_compare_gt",
            help="GT ボックスの BEV を左に並べて表示する",
        )
        show_bf_lidar = st.checkbox(
            "LiDAR Instances", value=False, key="_w_bf_lidar",
            disabled=not (run_info and run_info["use_lidar"]),
        )
        show_bf_depth = st.checkbox(
            "Depth Instances", value=True, key="_w_bf_depth")
        show_hull = st.checkbox(
            "Convex-Hull", value=False, key="_w_bf_hull",
            help="当てはめに使った凸包を重ねる")
        S.init_sticky(W_BF_COLOR, OPT_BF_COLOR, COLOR_MODE_LABEL)
        bf_color_mode = st.radio(
            "Instance Color", COLOR_MODES, key=W_BF_COLOR,
            on_change=S.sync_sticky, args=(W_BF_COLOR, OPT_BF_COLOR),
        )

    if not view_run_id:
        with bf_view_col:
            st.info("保存済みの推論結果がありません。")
    else:
        bf_fittings = load_box_fittings(
            view_run_id, frame_tokens,
            include_points=(show_bf_depth or show_bf_lidar),
            include_hull=show_hull,
        )
        bf_flat = [fit for items_ in bf_fittings.values() for fit in items_]

        def _bf_color(item: dict) -> str:
            key = (str(item.get("track_id")) if bf_color_mode == COLOR_MODE_TRACK
                   else str(item.get("label")))
            return (color_for_track(key) if bf_color_mode == COLOR_MODE_TRACK
                    else color_for_label(key))

        # 推定ボックス（当てはめできたものだけ）
        est_boxes = [
            {
                "key": f"{fit['label']}#{fit['track_id']}",
                "color": _bf_color(fit),
                "center_xy": fit["center_ego"][:2],
                "width": fit["size_wlh"][0],
                "length": fit["size_wlh"][1],
                "yaw": fit["yaw_ego"] or 0.0,
            }
            for fit in bf_flat if fit.get("center_ego") and fit.get("size_wlh")
        ]
        hulls = [
            {"key": f"{fit['label']}#{fit['track_id']}", "color": _bf_color(fit),
             "points": fit["hull_xy"]["points"]}
            for fit in bf_flat
            if show_hull and fit.get("hull_xy") and fit["hull_xy"].get("points")
        ]

        bf_groups = []
        if show_bf_depth:
            bf_groups += group_instance_points(
                bf_flat, color_mode=bf_color_mode, points_key="points_depth_ego")
        if show_bf_lidar:
            bf_groups += group_instance_points(
                bf_flat, color_mode=bf_color_mode, points_key="points_lidar_ego")

        # GT は sample 単位。ego_pose はどのカメラフレームでも同じ
        gt_boxes = []
        if compare_gt:
            ego_pose = next((f["ego_pose"] for f in frames if f.get("ego_pose")), None)
            if ego_pose:
                gt_boxes = [
                    {
                        "key": f"{box['label']}(GT)",
                        # GT も推定と同じラベル色にする。左右に並べたとき、
                        # 同じ色の箱が対応するかどうかで見比べられる。
                        # Track ID 色分けのときも GT はラベル色のまま
                        # （GT に track_id は無く、色を対応づけられない）
                        "color": color_for_label(box["label"]),
                        "center_xy": box["center_ego"][:2],
                        "width": box["size_wlh"][0],
                        "length": box["size_wlh"][1],
                        "yaw": box["yaw_ego"],
                    }
                    for box in gt_boxes_in_ego(
                        list_gt_annotations(dataset_id, selected_sample["token"]),
                        ego_pose,
                    )
                ]

        # 左右で同じ表示範囲にする。揃えないと同じ物体が別位置に見える
        range_sources = [np.asarray(g["points"]) for g in bf_groups]
        centers = [b["center_xy"] for b in est_boxes + gt_boxes]
        if centers:
            range_sources.append(np.asarray(centers, dtype=float))
        axis_range = combined_axis_range(range_sources)

        with bf_view_col:
            st.caption(
                f"インスタンス {len(bf_flat)} 件 / ボックス生成 {len(est_boxes)} 件"
                + (f" / GT {len(gt_boxes)} 件" if compare_gt else "")
            )
            common = dict(
                max_points_per_trace=settings.POINTCLOUD_DISPLAY_MAX_POINTS,
                axis_range=axis_range,
            )
            if compare_gt:
                gt_col, est_col = st.columns(2)
                with gt_col:
                    render_bev(build_bev_figure(
                        boxes=gt_boxes, title="Ground truth", **common))
                with est_col:
                    render_bev(build_bev_figure(
                        instance_groups=bf_groups, boxes=est_boxes, hulls=hulls,
                        title="Fitted", **common))
            else:
                render_bev(build_bev_figure(
                    instance_groups=bf_groups, boxes=est_boxes, hulls=hulls,
                    **common))

S.render_selection_sidebar(dataset_name=dataset["name"], scene_name=scene["name"])
