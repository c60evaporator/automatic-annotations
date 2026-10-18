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
from app.models.ann_intermediate import IOU_LABEL_MATCHES
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
from app.services.label_service import all_labels
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
from app.streamlit.components.sample_token import render_sample_token
from app.streamlit.components.depth_viewer import colorize_depth, render_depth_grid
from app.streamlit.components.det2d_viewer import render_label_legend
from app.streamlit.components.det2d_viewer import color_for as color_for_label
from app.streamlit.components.instance_tracking_viewer import (
    COLOR_MODE_LABEL,
    COLOR_MODE_TRACK,
    COLOR_MODES,
    INSTANCE_TEXT_MODES,
    TEXT_MODE_NONE,
    TEXT_MODE_TRACK,
    color_for_track,
    draw_instances,
    preferred_instances,
)
from app.streamlit.components.box3d_viewer import (
    BOX3D_TEXT_MODES,
    box_center_in_fov,
    render_box3d_comparison_grid,
    render_box3d_grid,
)
from app.streamlit.components.bev_viewer import (
    build_bev_figure,
    combined_axis_range,
    render_bev,
)
from app.streamlit.components.pointcloud_viewer import (
    MARKER_SIZE_LIDAR,
    SYMBOL_LIDAR,
    CAMERA_PRESETS,
    VIEWS,
    VIEW_GLOBAL,
    build_pointcloud_figure,
    global_camera,
    group_instance_points,
    lshape_overlays,
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
# L 字フィルタの設定キー（深度・LiDAR ごと）
LSHAPE_DEPTH_KEYS = (
    "lshape_bin_deg", "lshape_quantile", "lshape_margin_min", "lshape_margin_ratio",
)
LSHAPE_LIDAR_KEYS = ("lshape_bin_deg", "lshape_quantile", "lshape_margin")


def run_refilter_params(run_info: dict) -> dict:
    """run の設定そのままの再フィルタ条件.

    Filter Params の初期値と同じ値になるようにしている。Instance Points を
    途中段階（Raw・ROR/DBSCAN）に切り替えたときの自動の再計算に使う。
    スライダーを触らずに Apply した場合と同じ条件になるので、
    キャッシュも共有される。
    """
    run_depth = run_info.get("depth_params") or {}
    run_lidar = run_info.get("lidar_params") or {}
    run_lshape = (run_info.get("box_fitting_params") or {}).get("lshape") or {}
    run_distance = (
        (run_info.get("box_fitting_params") or {}).get("distance_filter") or {}
    )
    return {
        "depth": {
            "ror_nb_points": int(run_depth.get(
                "ror_nb_points", settings.DEPTH_ROR_NB_POINTS_DEFAULT)),
            "ror_radius": float(run_depth.get(
                "ror_radius", settings.DEPTH_ROR_RADIUS_DEFAULT)),
            "dbscan_eps": float(run_depth.get(
                "dbscan_eps", settings.DEPTH_DBSCAN_EPS_DEFAULT)),
            "dbscan_min_samples": int(run_depth.get(
                "dbscan_min_samples", settings.DEPTH_DBSCAN_MIN_SAMPLES_DEFAULT)),
            "small_mask_nb_points_ratio": float(run_depth.get(
                "small_mask_nb_points_ratio", 1.0)),
            **{k: float(run_depth[k]) for k in LSHAPE_DEPTH_KEYS if k in run_depth},
        },
        "lidar": {
            "ror_nb_points": int(run_lidar.get(
                "ror_nb_points", settings.LIDAR_ROR_NB_POINTS_DEFAULT)),
            "ror_radius": float(run_lidar.get(
                "ror_radius", settings.LIDAR_ROR_RADIUS_DEFAULT)),
            "dbscan_eps": float(run_lidar.get(
                "dbscan_eps", settings.LIDAR_DBSCAN_EPS_DEFAULT)),
            "dbscan_min_samples": int(run_lidar.get(
                "dbscan_min_samples", settings.LIDAR_DBSCAN_MIN_SAMPLES_DEFAULT)),
            "min_points": int(run_lidar.get(
                "min_points", settings.LIDAR_MIN_POINTS_DEFAULT)),
            "max_points_for_depth": int(run_lidar.get(
                "max_points_for_depth", settings.LIDAR_MAX_POINTS_FOR_DEPTH_DEFAULT)),
            **{k: float(run_lidar[k]) for k in LSHAPE_LIDAR_KEYS if k in run_lidar},
        },
        # L 字の設定が無い run（L 字フィルタ導入前）は無効のまま
        "lshape": dict(run_lshape),
        # 距離フィルタも同じ（導入前の run は対象ラベルが無いので適用されない）
        "distance": dict(run_distance),
    }


def lshape_params_differ(params: dict, run_info: dict) -> bool:
    """再フィルタの L 字の設定が、run（推論時）と違うか."""
    base = run_refilter_params(run_info)
    applied = params.get("lshape") or {}
    if not applied.get("enabled") and not base["lshape"].get("enabled"):
        # どちらも無効なら、他の値が違っても結果は同じ
        return False
    if applied != base["lshape"]:
        return True
    return any(
        (params.get(kind) or {}).get(k) != base[kind].get(k)
        for kind, keys in (("depth", LSHAPE_DEPTH_KEYS), ("lidar", LSHAPE_LIDAR_KEYS))
        for k in keys
    )


DISTANCE_PARAM_KEYS = (
    "methods", "labels", "quantile", "margin", "dbscan_eps", "dbscan_min_samples",
)


def distance_params_differ(params: dict, run_info: dict) -> bool:
    """再フィルタの距離フィルタの設定が、run（推論時）と違うか."""
    base = run_refilter_params(run_info)["distance"]
    applied = params.get("distance") or {}
    return any(applied.get(k) != base.get(k) for k in DISTANCE_PARAM_KEYS)


def distance_methods_text(methods: dict | None) -> str:
    """距離フィルタの方式の表示（{label: method} を方式ごとにまとめる）."""
    by_method: dict[str, list[str]] = {}
    for label, method in (methods or {}).items():
        if method:
            by_method.setdefault(str(method), []).append(label)
    if not by_method:
        return "なし"
    return " / ".join(f"{m}: {', '.join(ls)}" for m, ls in by_method.items())
# 深度点群を補正前・補正後のどちらで表示するか
OPT_PC_DEPTH_INST, W_PC_DEPTH_INST = "boxfit_pc_depth_inst", "_w_boxfit_pc_depth_inst"
DEPTH_INSTANCE_RAW = "raw"
DEPTH_INSTANCE_CORRECTED = "LiDAR corrected"
DEPTH_INSTANCE_MODES = (DEPTH_INSTANCE_RAW, DEPTH_INSTANCE_CORRECTED)
# Box Fitting タブの表示オプション
OPT_BF_COLOR, W_BF_COLOR = "boxfit_bf_color", "_w_boxfit_bf_color"
# Box Fitting タブの深度点群の表示（補正前 / 補正後）。
# 既定は補正後。当てはめに使っているのが補正後の点群なので、
# raw のままだとボックスと点群がずれて見える
OPT_BF_DEPTH_INST, W_BF_DEPTH_INST = (
    "boxfit_bf_depth_inst", "_w_boxfit_bf_depth_inst"
)

# カメラ跨ぎの結合を目で確かめるための色分け。
# global_track_id で塗ると、結合されたインスタンスが別カメラでも同色になる
COLOR_MODE_GLOBAL = "Global Track"
BOXFIT_COLOR_MODES = (*COLOR_MODES, COLOR_MODE_GLOBAL)


def instance_color(item: dict, color_mode: str) -> str:
    """インスタンスの表示色を決める（3D ボックスと点群で共通）."""
    if color_mode == COLOR_MODE_GLOBAL:
        # 結合していない run では global_track_id が無いので track_id へ落とす
        key = str(item.get("global_track_id") or item.get("track_id", ""))
        return color_for_track(key)
    if color_mode == COLOR_MODE_TRACK:
        return color_for_track(str(item.get("track_id", "")))
    return color_for_label(str(item.get("label", "")))


def instance_legend_key(item: dict, color_mode: str) -> str:
    """色分けの単位となるキー（凡例フィルタと揃える）."""
    if color_mode == COLOR_MODE_GLOBAL:
        return str(item.get("global_track_id") or item.get("track_id", ""))
    if color_mode == COLOR_MODE_TRACK:
        return str(item.get("track_id", ""))
    return str(item.get("label", ""))


def render_merge_diagnostics(run_info: dict | None) -> None:
    """カメラ間結合の診断をエクスパンダーで出す.

    「閾値を緩めても結合されない」ときの原因は閾値ではなく、
    1 対 1 の割り当てやカメラ重複の保護であることが多い。
    どの対がどの理由で落ちたかを見られるようにしている。
    """
    diagnostics = (run_info or {}).get("merge_diagnostics")
    if not diagnostics:
        return

    groups = diagnostics.get("groups") or []
    edges = diagnostics.get("edges") or []
    merged = [g for g in groups if len(g.get("members") or []) > 1]
    too_few = diagnostics.get("too_few_points") or {}
    too_few_tracks = too_few.get("tracks") or []

    title = f"Merge Diagnostics（{len(groups)} グループ / 結合 {len(merged)} 件"
    if too_few.get("num_frames"):
        title += f" / 点数不足 {too_few['num_frames']} フレーム"
    with st.expander(title + "）", expanded=False):
        st.caption(
            "結合の条件: "
            + " / ".join(f"{k}={v}" for k, v in (diagnostics.get("params") or {}).items())
        )
        st.markdown("**グループ**")
        st.dataframe(
            [
                {
                    "global_track_id": g["global_track_id"],
                    "members": ", ".join(g.get("members") or []),
                    "label": g.get("label"),
                    "frames": g.get("num_frames"),
                }
                for g in sorted(groups, key=lambda g: -len(g.get("members") or []))
            ],
            width="stretch", hide_index=True,
        )

        if too_few_tracks:
            st.markdown("**点数不足で当てはめなかったグループ**")
            st.caption(
                f"結合後の点数が Min Points to fit Box"
                f"（{too_few.get('min_points')}）に届かなかったもの。"
                f"{too_few.get('num_tracks')} グローバルトラック / "
                f"{too_few.get('num_frames')} フレーム。"
                "点群は残っているので、閾値を下げれば拾える"
            )
            if too_few.get("num_tracks", 0) > len(too_few_tracks):
                st.caption(
                    f"フレーム数の多い順に {len(too_few_tracks)} / "
                    f"{too_few['num_tracks']} 件を表示"
                )
            st.dataframe(
                [
                    {
                        "global_track_id": r["global_track_id"],
                        "label": r.get("label"),
                        # 「全フレームで落ちた」のか「遠方だけ落ちた」のかが分かる
                        "skipped / total": (
                            f"{r['num_frames_skipped']} / {r['num_frames_total']}"
                        ),
                        "min points": r.get("min_num_points"),
                        "median points": r.get("median_num_points"),
                    }
                    for r in too_few_tracks
                ],
                width="stretch", hide_index=True,
            )

        st.markdown("**トラック対の判定**")
        gated = [e for e in edges if e.get("status") == "gated"]
        if gated:
            st.caption(
                f"status=gated は足切り（凸包の隙間・ラベル）で、重なりの判定まで"
                f"進まなかった組（表示中 {len(gated)} 件）"
            )
        total = diagnostics.get("num_edges_total", len(edges))
        if total > len(edges):
            st.caption(f"重なりの大きい順に {len(edges)} / {total} 件を表示")
        st.dataframe(
            [
                {
                    "track A": e["pair"][0], "track B": e["pair"][1],
                    "max overlap": e["max_overlap"],
                    # 古い run の診断には無い
                    "min gap [m]": e.get("min_gap"),
                    "status": e["status"], "reason": e.get("reason", ""),
                }
                for e in edges
            ],
            width="stretch", hide_index=True,
        )


def has_box(item: dict) -> bool:
    """3D ボックスを描く対象か.

    結合したグループはボックスを **代表 1 行にだけ**持つ
    （非代表行は status="merged" で center/size が無い）。
    is_primary も併せて見て、意図を明示しておく
    """
    return bool(
        item.get("center_ego") and item.get("size_wlh")
        and item.get("is_primary", True)
    )
# Box on Cam タブの表示オプション
OPT_CAM_COLOR, W_CAM_COLOR = "boxfit_cam_color", "_w_boxfit_cam_color"
OPT_CAM_TEXT, W_CAM_TEXT = "boxfit_cam_text", "_w_boxfit_cam_text"
W_SAMPLE = "_w_boxfit_sample"

MASK_MODES = ("None", "Original", "Closed")

# インスタンス点群を、外れ値除去の前後どちらで見るか。
#
# NOTE: フィルタ前の点群は DB に保存していない（深度マップと
# クロージング後マスクから再生成できるため）。再フィルタ用の
# エンドポイントを実装するまで Raw は選べない。
POINTS_MODE_RAW = "Raw"
POINTS_MODE_FILTERED = "ROR/DBSCAN"
# ROR/DBSCAN の後に L 字フィルタをかけたもの。対象外のラベルは ROR/DBSCAN と同じ
POINTS_MODE_LSHAPE = "L-shape Filter"
# L 字フィルタの後に距離フィルタをかけたもの。保存済みの点群（＝当てはめに
# 使った点群）はこれに当たる。対象外のラベルは L-shape Filter と同じ
POINTS_MODE_DISTANCE = "Distance Filter"
POINTS_MODES = (
    POINTS_MODE_RAW, POINTS_MODE_FILTERED, POINTS_MODE_LSHAPE, POINTS_MODE_DISTANCE,
)
# 保存済みの点群に当たる（最終段階の）表示モード
POINTS_MODE_FINAL = POINTS_MODE_DISTANCE
# 表示モード → BoxFitting3D のカラム名
POINTS_COLUMNS = {
    POINTS_MODE_FINAL: ("points_depth_ego", "points_lidar_ego"),
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

        # 並びは実際の適用順。点群を作る → カメラ間で結合する →
        # 結合した点群にボックスを当てはめる
        (general_tab, depth_tab, lidar_tab, merge_tab,
         fitting_tab) = st.tabs(
            ["General", "Depth Estimation", "LiDAR Pointcloud",
             "Inter-cam Merge", "Box Fitting"]
        )

        with general_tab:
            # LiDAR の統合・地面除去・混合は未実装。
            # 実装するまでチェックできないようにしておく
            use_lidar = st.checkbox(
                "Use LiDAR", value=settings.BOXFIT_USE_LIDAR_DEFAULT,
                help=("LiDAR 点をインスタンスマスクへ投影して、"
                      "インスタンスごとの LiDAR 点群を作る。"
                      "深度点群との混合（座標補正）は未実装で、"
                      "現在は表示と外れ値除去までが動く"),
            )
            st.markdown("**Mask Closing**")
            # カーネルサイズは奇数のみ。偶数だとカーネルの中心を画素の真ん中に
            # 置けず、クロージング後のマスクが右下へ 1 画素ずれる
            mask_dilation = st.slider(
                "Dilation", 1, settings.MASK_DILATION_MAX,
                value=settings.MASK_DILATION_DEFAULT, step=2,
                help="膨張のカーネルサイズ（奇数のみ）",
            )
            mask_erosion = st.slider(
                "Erosion", 1, settings.MASK_EROSION_MAX,
                value=settings.MASK_EROSION_DEFAULT, step=2,
                help="収縮のカーネルサイズ（奇数のみ）",
            )
            st.markdown("**Small Mask**")
            small_mask_short_side = st.slider(
                "Small Mask Short Side [px]", 0,
                settings.SMALL_MASK_SHORT_SIDE_MAX,
                value=settings.SMALL_MASK_SHORT_SIDE_DEFAULT, step=5,
                help=("マスクの外接矩形の短辺（元画像の px）がこの値以下なら"
                      "小物体として扱い、収縮と深度点群の ROR / DBSCAN を緩める。"
                      "面積ではなく短辺で判定するのは、収縮で失われる割合が"
                      "マスクの太さで決まるため。0 で無効"),
            )
            small_mask_erosion_ratio = st.slider(
                "Small Mask Erosion Ratio", 0.0, 1.0,
                value=settings.SMALL_MASK_EROSION_RATIO_DEFAULT, step=0.05,
                help=("小物体の収縮を弱める。片側で削る画素数 (Erosion − Dilation) / 2 に"
                      "倍率をかけて四捨五入し、Dilation + 2 × (片側の画素数) を"
                      "小物体の Erosion にする（奇数のまま保つため）。"
                      "Erosion ≤ Dilation なら適用しない"),
            )
            if mask_erosion > mask_dilation:
                # 推論側（depth_ops.small_mask_erosion）と同じ式
                per_side = int(
                    (mask_erosion - mask_dilation) / 2 * small_mask_erosion_ratio
                    + 0.5
                )
                st.caption(
                    f"小物体の Erosion: {mask_erosion} → "
                    f"{mask_dilation + 2 * per_side}"
                )

            st.markdown("**L-shape Filter**")
            lshape_enabled = st.checkbox(
                "Enable L-shape Filter",
                value=settings.LSHAPE_FILTER_ENABLED_DEFAULT,
                key="_w_lshape_enabled",
                help=("車両の手前 2 面（BEV で L 字）を検出し、それより手前へ"
                      "突き出た点を落とす（ROR / DBSCAN の後、深度と LiDAR に"
                      "別々に適用）。MOA は手前の輪郭で向きと大きさを決めるため、"
                      "小さな突出でも箱が引きずられる"),
            )
            st.caption(
                "対象ラベル: " + ", ".join(settings.LSHAPE_FILTER_LABELS)
                + "（config の LSHAPE_FILTER_LABELS で指定）"
            )
            lshape_support_band = st.slider(
                "L-shape Support Band [m]", 0.05,
                settings.LSHAPE_SUPPORT_BAND_MAX,
                value=settings.LSHAPE_SUPPORT_BAND_DEFAULT, step=0.05,
                disabled=not lshape_enabled, key="_w_lshape_support_band",
                help=("輪郭点が面に「乗っている」とみなす幅（深度・LiDAR 共通）。"
                      "輪郭点の重み・L 字の当てはめ（面に乗る点／手前に残る点の判定）・"
                      "除去の条件（面から垂直方向にこれを超えて手前の点だけ消す）の"
                      "4 つに効く。大きくすると L 字は見つけやすくなるが、"
                      "面に近い突出は消えにくくなる"),
            )
            lshape_violation_weight = st.slider(
                "L-shape Violation Weight", 0.0,
                settings.LSHAPE_VIOLATION_WEIGHT_MAX,
                value=settings.LSHAPE_VIOLATION_WEIGHT_DEFAULT, step=0.25,
                disabled=not lshape_enabled, key="_w_lshape_violation_weight",
                help=("面より手前（センサー側）に残る輪郭点 1 つあたりの減点"
                      "（面に乗る点 1 つ = 1）。大きいほど L 字が外側に寄る。"
                      "小さいと短辺がボンネット等の内側の点に引きずられてめり込み、"
                      "大きすぎると手前のノイズの塊に面が乗ってノイズが残る"),
            )
            st.caption(
                "角度の刻み・調整幅は config の LSHAPE_ANGLE_STEP_DEG / "
                "LSHAPE_ADJUST_RANGE_DEG / LSHAPE_ADJUST_STEP_DEG で指定"
            )

            st.markdown("**Distance Filter**")
            st.caption(
                "方式: " + distance_methods_text(settings.DISTANCE_FILTER_METHODS)
                + "（config の DISTANCE_FILTER_METHODS で指定。long_side の長辺は "
                "DEFAULT_BOX_SIZES）"
            )
            distance_cols = st.columns(4)
            with distance_cols[0]:
                distance_quantile = st.slider(
                    "Distance Quantile [%]", 0.0,
                    settings.DISTANCE_FILTER_QUANTILE_MAX,
                    value=settings.DISTANCE_FILTER_QUANTILE_DEFAULT, step=1.0,
                    key="_w_distance_quantile",
                    help=("L 字フィルタの後、深度と LiDAR に別々に適用する（BEV のみ）。"
                          "センサーからの距離の分位点。long_side ではこれを基準距離 ref "
                          "とし、ref − Margin 〜 ref + 長辺 + Margin の点だけを残す。"
                          "dbscan では、点数上位 2 クラスタのどちらが手前かを"
                          "この分位点で比べる"),
                )
            with distance_cols[1]:
                distance_margin = st.slider(
                    "Distance Margin [m]", 0.0,
                    settings.DISTANCE_FILTER_MARGIN_MAX,
                    value=settings.DISTANCE_FILTER_MARGIN_DEFAULT, step=0.05,
                    key="_w_distance_margin",
                    help=("long_side の余裕。手前（ref − Margin）と奥（ref + 長辺 + "
                          "Margin）の両側に取る。深度推定の誤差や想定サイズとの差を吸収する"),
                )
            with distance_cols[2]:
                distance_dbscan_eps = st.slider(
                    "Distance DBSCAN eps [m]", 0.05,
                    settings.DISTANCE_FILTER_DBSCAN_EPS_MAX,
                    value=settings.DISTANCE_FILTER_DBSCAN_EPS_DEFAULT, step=0.05,
                    key="_w_distance_dbscan_eps",
                    help=("dbscan 方式の DBSCAN（BEV、深度・LiDAR 共通）。"
                          "点数上位 2 クラスタのうち手前の方だけを残す。"
                          "物体と背景の隙間より小さく、物体内の点の間隔より大きくする"),
                )
            with distance_cols[3]:
                distance_dbscan_min = st.slider(
                    "Distance DBSCAN min_samples", 1,
                    settings.DISTANCE_FILTER_DBSCAN_MIN_SAMPLES_MAX,
                    value=settings.DISTANCE_FILTER_DBSCAN_MIN_SAMPLES_DEFAULT, step=1,
                    key="_w_distance_dbscan_min",
                    help=("dbscan 方式の DBSCAN の min_samples（深度・LiDAR 共通）。"
                          "LiDAR は点が疎なので小さめにする"),
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
            st.markdown("**Small Mask**")
            small_mask_nb_ratio = st.slider(
                "Small Mask nb_points / min_samples Ratio", 0.1, 1.0,
                value=settings.SMALL_MASK_NB_POINTS_RATIO_DEFAULT, step=0.05,
                key="_w_depth_small_mask_ratio",
                help=("小物体（General タブの Small Mask Short Side 以下）の"
                      "ROR nb_points と DBSCAN min_samples にかける倍率。"
                      "nb_points はラベル別の倍率（NB_POINTS_RATIO）をかけた後に"
                      "さらにかける。下限は nb_points=1 / min_samples=2"),
            )
            st.markdown("**L-shape Filter**")
            depth_lshape_bin = st.slider(
                "Azimuth Bin [deg]", 0.1, 5.0,
                value=settings.DEPTH_LSHAPE_BIN_DEG_DEFAULT, step=0.1,
                key="_w_depth_lshape_bin", disabled=not lshape_enabled,
                help=("手前輪郭を作るときの方位ビンの幅。各ビンで距離の分位点を取り、"
                      "センサーに見えている輪郭を取り出す"),
            )
            depth_lshape_quantile = st.slider(
                "Near Quantile [%]", 0.0, 50.0,
                value=settings.DEPTH_LSHAPE_QUANTILE_DEFAULT, step=1.0,
                key="_w_depth_lshape_quantile", disabled=not lshape_enabled,
                help=("各ビンで取る距離の分位点。0 だと孤立した 1 点に輪郭が"
                      "引っ張られ、大きいとボンネットなど奥の面に輪郭が下がる"),
            )
            depth_lshape_margin_min = st.slider(
                "Margin Min [m]", 0.05, 2.0,
                value=settings.DEPTH_LSHAPE_MARGIN_MIN_DEFAULT, step=0.05,
                key="_w_depth_lshape_margin_min", disabled=not lshape_enabled,
                help=("L 字の面より視線方向にこれ以上手前にある点を落とす。"
                      "実際の閾値は max(Margin Min, Margin Ratio × 距離)"),
            )
            depth_lshape_margin_ratio = st.slider(
                "Margin Ratio", 0.0, 0.2,
                value=settings.DEPTH_LSHAPE_MARGIN_RATIO_DEFAULT, step=0.01,
                key="_w_depth_lshape_margin_ratio", disabled=not lshape_enabled,
                help="単眼深度の誤差は距離とともに大きくなるので、閾値を距離に比例させる",
            )

        with lidar_tab:
            depth_correction_method = st.selectbox(
                "Depth Correction", settings.DEPTH_CORRECTION_METHODS,
                index=settings.DEPTH_CORRECTION_METHODS.index(
                    settings.DEFAULT_DEPTH_CORRECTION_METHOD
                ),
                help=("LiDAR を基準に深度点群を補正する方式（Use LiDAR のときだけ効く）。"
                      "scale: 視線方向の伸縮（既定）/ shift: 距離のずれ / "
                      "affine: 伸縮＋ずれ（点数が少ないと scale へ落ちる）。"
                      "インスタンス単位では深度の範囲が狭いため、1 パラメータで足りる"),
            )
            num_lidar_sweeps = st.number_input(
                "LiDAR Sweeps", min_value=1, max_value=settings.SWEEPS_PER_SAMPLE,
                value=settings.LIDAR_NUM_SWEEPS_DEFAULT, step=1,
            )
            min_lidar_points = st.slider(
                "Min LiDAR Points to use LiDAR", 1, settings.LIDAR_MIN_POINTS_MAX,
                value=settings.LIDAR_MIN_POINTS_DEFAULT, step=1,
                help=("これ未満のインスタンスは LiDAR 点群を持たせない。"
                      "深度点群だけでフィッティングし、表示もされない"),
            )
            max_lidar_for_depth = st.slider(
                "Max LiDAR Points to use Depth", 0,
                settings.LIDAR_MAX_POINTS_FOR_DEPTH_MAX,
                value=settings.LIDAR_MAX_POINTS_FOR_DEPTH_DEFAULT, step=1,
                help=("LiDAR 点がこの数以上のインスタンスは、**深度点群を使わず**"
                      "LiDAR だけで当てはめ・カメラ間結合を行う。"
                      "LiDAR で形が取れている場合、深度推定の外れ値が混ざるほうが"
                      "害になるため。0 で無効（常に深度点群も使う）"),
            )
            min_correspondences = st.slider(
                "Min Correspondences for Pointwise Correction", 1,
                settings.LIDAR_MIN_CORRESPONDENCES_MAX,
                value=settings.LIDAR_MIN_CORRESPONDENCES_DEFAULT, step=1,
                help=("深度補正で、点ごとの対応（LiDAR 点 ↔ その周囲の"
                      "フィルタ後に残った深度点）から z_lidar / z_pred の中央値で"
                      "係数を求めるのに必要な対応点の数。届かなければ、"
                      "残った LiDAR 点と残った深度点それぞれの深度の中央値の比で求める"
                      "（遠方の小さな物体では画素の対応がほとんど取れないため）"),
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
            st.markdown("**L-shape Filter**")
            lidar_lshape_bin = st.slider(
                "Azimuth Bin [deg]", 0.1, 5.0,
                value=settings.LIDAR_LSHAPE_BIN_DEG_DEFAULT, step=0.1,
                key="_w_lidar_lshape_bin", disabled=not lshape_enabled,
                help="手前輪郭を作るときの方位ビンの幅（LiDAR は点が疎なので深度より広く）",
            )
            lidar_lshape_quantile = st.slider(
                "Near Quantile [%]", 0.0, 50.0,
                value=settings.LIDAR_LSHAPE_QUANTILE_DEFAULT, step=1.0,
                key="_w_lidar_lshape_quantile", disabled=not lshape_enabled,
                help="各ビンで取る距離の分位点",
            )
            lidar_lshape_margin = st.slider(
                "Margin [m]", 0.05, 2.0,
                value=settings.LIDAR_LSHAPE_MARGIN_DEFAULT, step=0.05,
                key="_w_lidar_lshape_margin", disabled=not lshape_enabled,
                help="L 字の面より視線方向にこれ以上手前にある点を落とす（LiDAR は固定値）",
            )

        with merge_tab:
            st.caption(
                "同じ物体が複数カメラに写ると別トラックになるため、"
                "点群化してから結合する。結合したグループには共通の "
                "global_track_id が付き、3D ボックスは結合後の点群に当てる"
            )
            enable_merge = st.checkbox(
                "カメラ間の結合を行う", value=True, key="_w_boxfit_enable_merge",
                help="オフにすると、従来どおりカメラごとに当てはめる",
            )
            merge_method = st.selectbox(
                "Match Method", settings.MERGE_METHODS,
                index=settings.MERGE_METHODS.index(settings.DEFAULT_MERGE_METHOD)
                if settings.DEFAULT_MERGE_METHOD in settings.MERGE_METHODS else 0,
                disabled=not enable_merge,
                help=("BEV convex-hull: BEV に落とした凸包の重なり率で判定する。"
                      "カメラごとに物体の違う面しか見えないため、和集合ではなく"
                      "小さい方の面積で割った値（IoS）を使う"),
            )
            merge_label_match = st.radio(
                "Label Match", IOU_LABEL_MATCHES,
                index=IOU_LABEL_MATCHES.index(settings.DEFAULT_MERGE_LABEL_MATCH)
                if settings.DEFAULT_MERGE_LABEL_MATCH in IOU_LABEL_MATCHES else 0,
                horizontal=True, disabled=not enable_merge,
                help=("結合してよいラベルの条件。判定にはトラック内で"
                      "多数決したラベルを使う"),
            )
            merge_cols = st.columns(4)
            with merge_cols[0]:
                merge_overlap = st.slider(
                    "Overlap Threshold", 0.0,
                    settings.MERGE_OVERLAP_THRESHOLD_MAX,
                    value=settings.DEFAULT_MERGE_OVERLAP_THRESHOLD, step=0.05,
                    disabled=not enable_merge,
                    help="凸包の重なり率（IoS）の下限",
                )
            with merge_cols[1]:
                merge_hull_gap = st.slider(
                    "Max Hull Gap [m]", 0.0,
                    settings.MERGE_MAX_HULL_GAP_MAX,
                    value=settings.DEFAULT_MERGE_MAX_HULL_GAP, step=0.1,
                    disabled=not enable_merge,
                    help=("BEV 凸包どうしの隙間がこれを超える組は、重なりを計算せず"
                          "落とす（足切り）。重なっていれば隙間は 0 なので、長い車両が"
                          "カメラの境界で分かれても落ちない。重なり率は凸包が接して"
                          "いないと 0 なので結合の結果には影響せず、"
                          "「重なりは無いが近い組」を診断に残す範囲だけが変わる"),
                )
            with merge_cols[3]:
                merge_same_cam_gap = st.slider(
                    "Max Same-Camera Gap", 0,
                    settings.MERGE_MAX_SAME_CAMERA_GAP_MAX,
                    value=settings.DEFAULT_MERGE_MAX_SAME_CAMERA_GAP, step=1,
                    disabled=not enable_merge,
                    help=("同一カメラの 2 トラックを同じグループへ入れてよい"
                          "時間的な隔たり（キーフレーム数）。カメラ A → B → A と"
                          "写り込む物体は A 側で 2 トラックに分断されるため、"
                          "同時に存在しなければ同居を許す。0 なら一切許さない"),
                )
            with merge_cols[2]:
                merge_min_frames = st.slider(
                    "Min Match Frames", 1,
                    settings.MERGE_MIN_MATCH_FRAMES_MAX,
                    value=settings.DEFAULT_MERGE_MIN_MATCH_FRAMES, step=1,
                    disabled=not enable_merge,
                    help=("何フレームでマッチしたら結合するか。同じ物体が"
                          "複数カメラに写るキーフレームは高々 1〜2 なので、"
                          "割合ではなくフレーム数で判定する"),
                )

        with fitting_tab:
            def _method_labels(method: str) -> str:
                labels = [k for k, v in settings.BOXFIT_METHODS.items() if v == method]
                return ", ".join(labels) or "なし"

            st.caption(
                "手法はラベルごとに config の BOXFIT_METHODS で指定する"
                "（指定の無いラベルは convex_hull_moa）"
            )
            st.markdown("**Common**")
            boxfit_min_points = st.slider(
                "Min Points to fit Box", 1, settings.BOXFIT_MIN_POINTS_MAX,
                value=settings.BOXFIT_MIN_POINTS_DEFAULT, step=1,
                help=("当てはめに必要な点数の下限。届かないインスタンスは"
                      "ボックスを作らない（status=too_few_points）。"
                      "点群は残るので、フィルタ条件の見直しはできる。"
                      "**カメラ間結合の後、グローバルトラック単位で"
                      "フレームごとに数える**（深度点＋LiDAR 点の合計）"),
            )

            st.markdown("**convex_hull_moa**")
            st.caption("対象ラベル: " + _method_labels("convex_hull_moa"))
            angle_step_deg = st.slider(
                "Angle Step [deg]",
                settings.BOXFIT_ANGLE_STEP_DEG_MIN,
                settings.BOXFIT_ANGLE_STEP_DEG_MAX,
                value=settings.BOXFIT_ANGLE_STEP_DEG_DEFAULT, step=0.1,
                key="_w_boxfit_moa_angle_step",
                help=("BEV の凸包に対し、センサーから見たオクルージョン面積が"
                      "最小になる向きを探す。その向きの探索刻み（細かいほど遅い）"),
            )

            st.markdown("**Bottom Estimation**")
            by_bottom: dict[str, list[str]] = {}
            for _label, _method in settings.BOX_BOTTOM_ESTIMATIONS.items():
                by_bottom.setdefault(_method, []).append(_label)
            st.caption(
                "方式: " + " / ".join(f"{m}: {', '.join(ls)}" for m, ls in by_bottom.items())
                + "（config の BOX_BOTTOM_ESTIMATIONS で指定。指定の無いラベルは "
                "percentile）"
            )
            bottom_cols = st.columns(2)
            with bottom_cols[0]:
                z_percentile_low = st.slider(
                    "z_percentile", 0.0, 50.0,
                    value=settings.BOXFIT_Z_PERCENTILE_LOW_DEFAULT, step=0.5,
                    key="_w_boxfit_bottom_percentile",
                    help=("percentile 方式の底面を決める、点群の z の分位点。"
                          "ground 方式で地面点が足りないときもこれを使う"),
                )
                ground_radius = st.slider(
                    "Ground Estimation Radius [m]", 0.5,
                    settings.BOX_BOTTOM_GROUND_RADIUS_MAX,
                    value=settings.BOX_BOTTOM_GROUND_RADIUS_DEFAULT, step=0.5,
                    key="_w_boxfit_ground_radius",
                    help=("ground 方式で使う地面点の、箱の外形（BEV の矩形）からの"
                          "距離の上限。車体の下は LiDAR の影になるので、実際には"
                          "周囲の帯の地面点を使う"),
                )
            with bottom_cols[1]:
                min_ground_points = st.slider(
                    "Min Ground Points", 1,
                    settings.BOX_BOTTOM_MIN_GROUND_POINTS_MAX,
                    value=settings.BOX_BOTTOM_MIN_GROUND_POINTS_DEFAULT, step=1,
                    key="_w_boxfit_min_ground_points",
                    help=("ground 方式で、範囲内の地面点がこの数以下なら"
                          "percentile 方式に切り替える（遠方では地面点が疎）"),
                )
                ground_height_method = st.radio(
                    "Ground Height Method", settings.BOX_BOTTOM_GROUND_HEIGHT_METHODS,
                    index=settings.BOX_BOTTOM_GROUND_HEIGHT_METHODS.index(
                        settings.BOX_BOTTOM_GROUND_HEIGHT_METHOD_DEFAULT),
                    horizontal=True, key="_w_boxfit_ground_height_method",
                    help=("median: 地面点の z の中央値 / least_squares: 地面点に"
                          "平面を最小二乗で当てはめ、箱の XY 中心での高さ"
                          "（坂道で手前の地面に偏らない）"),
                )

            st.markdown("**Top Estimation**")
            z_percentile_high = st.slider(
                "z_percentile", 50.0, 100.0,
                value=settings.BOXFIT_Z_PERCENTILE_HIGH_DEFAULT, step=0.5,
                key="_w_boxfit_top_percentile",
                help=("上面を決める、点群の z の分位点。100 にすると"
                      "はみ出した 1 点で箱が縦に伸びる"),
            )

mask_params = {
    "dilation": int(mask_dilation), "erosion": int(mask_erosion),
    # 小物体（短辺が閾値以下）は収縮を弱める
    "small_mask_short_side": int(small_mask_short_side),
    "small_mask_erosion_ratio": float(small_mask_erosion_ratio),
}
# 結合しない場合は空の dict を渡す（推論側が結合を飛ばす）
merge_params = {
    "method": merge_method,
    "label_match": merge_label_match,
    "overlap_threshold": float(merge_overlap),
    "max_hull_gap": float(merge_hull_gap),
    "min_match_frames": int(merge_min_frames),
    "max_same_camera_gap": int(merge_same_cam_gap),
} if enable_merge else {}
box_fitting_params = {
    # ラベルごとの手法と、手法ごとの設定
    "methods": dict(settings.BOXFIT_METHODS),
    "convex_hull_moa": {"angle_step_deg": float(angle_step_deg)},
    # ラベルごとの想定サイズ (length, width, height)。
    # 距離フィルタ（long_side）の長辺に使う
    "box_sizes": {k: [float(x) for x in v]
                  for k, v in settings.DEFAULT_BOX_SIZES.items()},
    # 上面と（percentile 方式の）底面の分位点
    "z_percentiles": [float(z_percentile_low), float(z_percentile_high)],
    # 底面の推定（ラベルごとの方式と ground 方式の設定）
    "bottom": {
        "methods": dict(settings.BOX_BOTTOM_ESTIMATIONS),
        "ground_radius": float(ground_radius),
        "min_ground_points": int(min_ground_points),
        "ground_height_method": str(ground_height_method),
    },
    # 当てはめに使う点数の下限。結合後のグローバルトラック単位で判定する
    "min_points": int(boxfit_min_points),
    # L 字フィルタの共通設定（深度・LiDAR ごとの設定は各 params の lshape_*）
    "lshape": {
        "enabled": bool(lshape_enabled),
        "labels": list(settings.LSHAPE_FILTER_LABELS),
        "angle_step_deg": float(settings.LSHAPE_ANGLE_STEP_DEG),
        "adjust_range_deg": float(settings.LSHAPE_ADJUST_RANGE_DEG),
        "adjust_step_deg": float(settings.LSHAPE_ADJUST_STEP_DEG),
        "boundary_quantile": float(settings.LSHAPE_BOUNDARY_QUANTILE),
        "support_band": float(lshape_support_band),
        "violation_weight": float(lshape_violation_weight),
    },
    # 距離フィルタ（L 字フィルタの直後、深度と LiDAR に別々に適用）
    "distance_filter": {
        "methods": dict(settings.DISTANCE_FILTER_METHODS),
        "quantile": float(distance_quantile),
        "margin": float(distance_margin),
        "dbscan_eps": float(distance_dbscan_eps),
        "dbscan_min_samples": int(distance_dbscan_min),
    },
}
depth_params = {
    "ror_nb_points": int(depth_ror_nb), "ror_radius": float(depth_ror_radius),
    "dbscan_eps": float(depth_dbscan_eps), "dbscan_min_samples": int(depth_dbscan_min),
    # 小物体の ROR nb_points / DBSCAN min_samples にかける倍率（深度のみ）
    "small_mask_nb_points_ratio": float(small_mask_nb_ratio),
    # L 字フィルタ（深度）
    "lshape_bin_deg": float(depth_lshape_bin),
    "lshape_quantile": float(depth_lshape_quantile),
    "lshape_margin_min": float(depth_lshape_margin_min),
    "lshape_margin_ratio": float(depth_lshape_margin_ratio),
}
lidar_params = {
    "min_points": int(min_lidar_points),
    # これ以上の LiDAR 点があるインスタンスは深度点群を使わない
    "max_points_for_depth": int(max_lidar_for_depth),
    # 深度補正（最終の補正）の対応の取り方
    "min_correspondences": int(min_correspondences),
    "correspondence_radius": int(settings.DEPTH_CORRECTION_NEIGHBOR_RADIUS_PX),
    # LiDAR を基準に深度点群を補正する方式。run に記録するため lidar_params に入れる
    "depth_correction": depth_correction_method,
    "ror_nb_points": int(lidar_ror_nb), "ror_radius": float(lidar_ror_radius),
    "dbscan_eps": float(lidar_dbscan_eps), "dbscan_min_samples": int(lidar_dbscan_min),
    # L 字フィルタ（LiDAR）
    "lshape_bin_deg": float(lidar_lshape_bin),
    "lshape_quantile": float(lidar_lshape_quantile),
    "lshape_margin": float(lidar_lshape_margin),
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
                            merge_params=merge_params,
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
                            merge_params=merge_params,
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
                        f"ann {run['nbr_annotations']}  lidar={run['use_lidar']}"
                        # tools/check_depth_boxfitting.py の --run に渡す ID
                        f"  run={run['id']}")

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

# 基準センサーの ego_pose。パイプラインが点群を揃えている座標系はこれ。
# 表示でも同じ基準に揃えないと、カメラごとの時刻差ぶん（自車 10 m/s・
# 時刻差 25 ms で 0.25 m）ずれて、継ぎ目が実際より悪く見える
reference_ego_pose = next(
    (
        f["ego_pose"] for f in list_frames_by_scene(dataset_id, scene_token)
        if f["sample_token"] == selected_sample["token"]
        and f["channel"] == settings.EGO_REFERENCE_CHANNEL and f.get("ego_pose")
    ),
    None,
)

depth_tab_view, pointcloud_tab_view, fitting_tab_view, cam_tab_view = st.tabs(
    ["Depth Estimation", "PointCloud", "Box Fitting", "Box on Cam"]
)

# ------------------------------------------------------------------
# Depth Estimation タブ
# ------------------------------------------------------------------
with depth_tab_view:
    view_col, opt_col = st.columns([8, 1])

    with opt_col:
        render_sample_token(selected_sample["token"])
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
        render_sample_token(selected_sample["token"])
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
            # **key に run を含める。** 固定キーだと、LiDAR ありの run で
            # チェックした状態が LiDAR なしの run へ持ち越され、
            # 無効化されたまま前の run の点群が表示され続ける。
            # run ごとに別ウィジェットにすれば、切り替え時に既定値へ戻る
            key=f"_w_pc_lidar_inst_{view_run_id}",
            disabled=not has_lidar_instances,
            help=None if has_lidar_instances else "この run は Use LiDAR=False で実行されています",
        )
        # 状態が残った場合の保険。無効な run では必ず表示しない
        show_lidar_instances = show_lidar_instances and has_lidar_instances
        show_raw_depth = st.checkbox(
            "Raw Depth", value=settings.SHOW_RAW_DEPTH_POINTCLOUD,
            key="_w_pc_raw_depth",
        )
        show_depth_instances = st.checkbox(
            "Depth Instances", value=True, key="_w_pc_depth_inst",
        )

        # 保存済みの点群は最終段階（距離フィルタ後）だけ。途中段階を選ぶと、
        # 推論時と同じパラメータで自動的に再計算して表示する。
        # 既定は最終段階。名前を変えた古い選択肢が残っていたら戻す
        for _key in (W_PC_POINTS, OPT_PC_POINTS):
            if st.session_state.get(_key) not in (None, *POINTS_MODES):
                st.session_state.pop(_key, None)
        S.init_sticky(W_PC_POINTS, OPT_PC_POINTS, POINTS_MODE_FINAL)
        points_mode = st.radio(
            "Instance Points", POINTS_MODES, key=W_PC_POINTS,
            on_change=S.sync_sticky, args=(W_PC_POINTS, OPT_PC_POINTS),
            help=("外れ値除去（ROR / DBSCAN）・L 字フィルタ・距離フィルタの"
                  "適用前後を切り替える。"
                  "保存済みの点群は距離フィルタ後（当てはめに使ったもの）。"
                  "それ以外を選ぶと、推論時と同じパラメータで再計算して表示する"
                  "（Filter Params で Apply した後はその値で再計算する）"),
        )

        S.init_sticky(W_PC_DEPTH_INST, OPT_PC_DEPTH_INST, DEPTH_INSTANCE_RAW)
        depth_instance_mode = st.radio(
            "Depth Instance", DEPTH_INSTANCE_MODES, key=W_PC_DEPTH_INST,
            on_change=S.sync_sticky, args=(W_PC_DEPTH_INST, OPT_PC_DEPTH_INST),
            help=("深度点群を補正前（raw）と、LiDAR を基準に補正した後"
                  "（LiDAR corrected）のどちらで表示するか。"
                  "補正できなかったインスタンスは補正前で表示する"),
        )

        show_lshape = st.checkbox(
            "L-shape Filter", value=False, key="_w_pc_lshape",
            help=("L 字フィルタの途中経過を重ねる。赤い × / ＋ が手前輪郭の点"
                  "（深度 / LiDAR）、半透明の面が検出した L 字（車両の手前 2 面）。"
                  "面の色は Instance Color の指定に従う。"
                  "Depth Instances / LiDAR Instances の表示に合わせて出す。"
                  "深度は補正できた場合は補正後の点群で判定しているので、"
                  "Depth Instance=raw だと点群と少しずれて見える"),
        )

        show_fitted_boxes = st.checkbox(
            "Fitted Boxes", value=False, key="_w_pc_fitted_boxes",
            help=("Box Fitting で推定した 3D ボックスを点群に重ねる。"
                  "色は Instance Color の指定に従う"),
        )

        S.init_sticky(W_PC_COLOR, OPT_PC_COLOR, COLOR_MODE_LABEL)
        pc_color_mode = st.radio(
            "Instance Color", BOXFIT_COLOR_MODES, key=W_PC_COLOR,
            on_change=S.sync_sticky, args=(W_PC_COLOR, OPT_PC_COLOR),
        )

        st.markdown("**Cameras**")
        enabled_channels = {
            sensor["channel"] for sensor in cam_sensors
            if st.checkbox(sensor["channel"], value=True,
                           key=f"_w_pc_cam_{sensor['channel']}")
        }


    raw_lidar_points = ground_points = raw_depth_points = None
    instance_groups: list[dict] = []
    lshape_items: list[dict] = []
    lshape_missing = 0
    # 途中段階を見るために、推論時と同じパラメータで自動的に再計算したか
    auto_refilter = False
    fitted_boxes: list[dict] = []
    refilter_summary: dict[str, int] | None = None

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
                chunks.append(depth_to_ego_points(
                    depth, frame, stride=4,
                    reference_ego_pose=reference_ego_pose,
                ))
            if chunks:
                raw_depth_points = np.vstack(chunks)

        if show_depth_instances or show_lidar_instances or show_fitted_boxes:
            # 対象カメラのフレームだけ読む。全フレームぶん点群を読むと重い
            tokens = tuple(
                f["token"] for f in frames if f["channel"] in enabled_channels
            )
            refilter_params = st.session_state.get(PC_REFILTER_PARAMS)
            if (not refilter_params and run_info
                    and points_mode != POINTS_MODE_FINAL):
                # 途中段階は保存していないので、推論時と同じパラメータで
                # 作り直す（Apply していなくても選べるようにするため）
                refilter_params = run_refilter_params(run_info)
                auto_refilter = True

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
                # 再フィルタの結果はカラム名が違う（DB のものと混ぜない）。
                # 候補を前から順に見て、最初に値のあるものを表示する。
                # L 字・距離フィルタ後は、適用しなかったインスタンスでは None
                # なので 1 つ前の段階へ落とす
                corrected = depth_instance_mode == DEPTH_INSTANCE_CORRECTED
                depth_candidates = {
                    POINTS_MODE_RAW: ["points_raw_ego"],
                    POINTS_MODE_FILTERED: ["points_filtered_ego"],
                    POINTS_MODE_LSHAPE: ["points_lshape_ego", "points_filtered_ego"],
                    POINTS_MODE_DISTANCE: ["points_distance_ego", "points_lshape_ego",
                                           "points_filtered_ego"],
                }[points_mode]
                if corrected:
                    # 補正後で見る。補正できなかったものは補正前へ落とす
                    depth_candidates = {
                        POINTS_MODE_RAW: ["points_corrected_raw_ego"],
                        POINTS_MODE_FILTERED: ["points_corrected_filtered_ego"],
                        POINTS_MODE_LSHAPE: ["points_corrected_lshape_ego",
                                             "points_corrected_filtered_ego"],
                        POINTS_MODE_DISTANCE: ["points_corrected_distance_ego",
                                               "points_corrected_lshape_ego",
                                               "points_corrected_filtered_ego"],
                    }[points_mode] + depth_candidates
                lidar_candidates = {
                    POINTS_MODE_RAW: ["points_lidar_raw_ego"],
                    POINTS_MODE_FILTERED: ["points_lidar_filtered_ego"],
                    POINTS_MODE_LSHAPE: ["points_lidar_lshape_ego",
                                         "points_lidar_filtered_ego"],
                    POINTS_MODE_DISTANCE: ["points_lidar_distance_ego",
                                           "points_lidar_lshape_ego",
                                           "points_lidar_filtered_ego"],
                }[points_mode]

                def _first(item: dict, keys: list[str]):
                    return next((item[k] for k in keys if item.get(k)), None)

                flat = [
                    {**i,
                     # 深度を使わないインスタンス（LiDAR が十分）は、
                     # 補正後で見るときは出さない
                     "_depth_display": (
                         _first(i, depth_candidates)
                         if not corrected or i.get("depth_used", True) else None
                     ),
                     "_lidar_display": _first(i, lidar_candidates)}
                    for i in flat
                ]
                depth_key, lidar_key = "_depth_display", "_lidar_display"
                def _removed(item: dict, kind: str, column: str = "lshape_filter") -> int:
                    return int(((item.get(column) or {}).get(kind) or {})
                               .get("num_removed", 0))

                refilter_summary = {
                    "raw": sum(i["num_points_raw"] for i in flat),
                    "kept": sum(i["num_points_kept"] for i in flat),
                    "instances": len(flat),
                    "lidar_raw": sum(i.get("num_lidar_raw", 0) for i in flat),
                    "lidar_kept": sum(i.get("num_lidar_kept", 0) for i in flat),
                    "lshape_depth": sum(_removed(i, "depth") for i in flat),
                    "lshape_lidar": sum(_removed(i, "lidar") for i in flat),
                    "distance_depth": sum(
                        _removed(i, "depth", "distance_filter") for i in flat),
                    "distance_lidar": sum(
                        _removed(i, "lidar", "distance_filter") for i in flat),
                    # 推論時の点数。差が出たときの切り分けに使う
                    "stored_raw": sum(i.get("stored_raw") or 0 for i in flat),
                    "stored_kept": sum(i.get("stored_kept") or 0 for i in flat),
                    "stored_lidar_raw": sum(i.get("stored_lidar_raw") or 0 for i in flat),
                    "stored_lidar_kept": sum(i.get("stored_lidar_kept") or 0 for i in flat),
                    "stored_lshape_depth": sum(i.get("stored_lshape_depth") or 0 for i in flat),
                    "stored_lshape_lidar": sum(i.get("stored_lshape_lidar") or 0 for i in flat),
                    "stored_distance_depth": sum(
                        i.get("stored_distance_depth") or 0 for i in flat),
                    "stored_distance_lidar": sum(
                        i.get("stored_distance_lidar") or 0 for i in flat),
                }
            else:
                fittings = load_box_fittings(
                    view_run_id, tokens,
                    include_points=(show_depth_instances or show_lidar_instances),
                ) if tokens else {}
                flat = [fit for items_ in fittings.values() for fit in items_]
                depth_key, lidar_key = POINTS_COLUMNS[POINTS_MODE_FINAL]
                if depth_instance_mode == DEPTH_INSTANCE_CORRECTED:
                    # 補正後の点群で表示する。補正できなかったインスタンス
                    # （LiDAR 点が足りない等）は補正前のまま表示する。
                    # **深度点群を使わなかったインスタンス（LiDAR が十分）は
                    # 何も出さない。** raw のほうは比較のために出す
                    # 元の dict は書き換えない（キャッシュを共有しているため）
                    flat = [
                        {**fit, "_depth_display": (
                            (fit.get("points_depth_corrected_ego")
                             or fit.get("points_depth_ego"))
                            if fit.get("depth_used", True) else None
                        )}
                        for fit in flat
                    ]
                    depth_key = "_depth_display"
                refilter_summary = None
                box_source = fittings

            if show_lshape:
                # 推論時の記録（DB）または Apply の結果のどちらでも、
                # 各インスタンスの lshape_filter に入っている
                lshape_items, lshape_missing = lshape_overlays(
                    flat, depth=show_depth_instances, lidar=show_lidar_instances,
                    # 面は点群と同じ色（Instance Color の指定）で塗る
                    color_fn=lambda f: instance_color(f, pc_color_mode),
                )

            if show_depth_instances:
                instance_groups += group_instance_points(
                    flat, color_mode=pc_color_mode, points_key=depth_key,
                    key_fn=lambda f: instance_legend_key(f, pc_color_mode),
                    color_fn=lambda k: instance_color(
                        {"global_track_id": k, "track_id": k, "label": k},
                        pc_color_mode),
                )
            if show_lidar_instances and lidar_key:
                instance_groups += group_instance_points(
                    flat, color_mode=pc_color_mode, points_key=lidar_key,
                    # 色は Instance Color に予約されているので、
                    # 深度由来との区別は形で付ける
                    symbol=SYMBOL_LIDAR, size=MARKER_SIZE_LIDAR,
                    # **表示では重複除去しない。**
                    # カメラ間で点群がどう重なっているかを見るのが目的なので、
                    # 重複したまま描く（重なりは中抜きマーカーで判別する）。
                    # 重複除去は結合（combine_summaries）の側だけで行う
                    key_fn=lambda f: instance_legend_key(f, pc_color_mode),
                    color_fn=lambda k: instance_color(
                        {"global_track_id": k, "track_id": k, "label": k},
                        pc_color_mode),
                )

            if show_fitted_boxes:
                for fit in (f for items_ in box_source.values() for f in items_):
                    if not has_box(fit):
                        continue
                    key = instance_legend_key(fit, pc_color_mode)
                    if enabled_keys is not None and key not in enabled_keys:
                        continue
                    fitted_boxes.append({
                        "key": f"{fit['label']}#{fit['track_id']}",
                        # 点群と同じ色にして、どの点群に当てた箱かを分かるようにする
                        "color": instance_color(fit, pc_color_mode),
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
            # 点群は基準 ego 座標にあるので、視点も基準の姿勢で決める
            ego_pose = reference_ego_pose or next(
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
            st.markdown("**L-shape Filter**")
            run_lshape = (
                ((run_info or {}).get("box_fitting_params") or {}).get("lshape") or {}
            )
            lshape_on_new = st.checkbox(
                "Enable L-shape Filter",
                # L 字の設定が無い run（導入前）は無効から始める
                value=bool(run_lshape.get("enabled", False)),
                key=f"{prefix}_lshape_enabled",
            )
            common_cols = st.columns(4)
            with common_cols[0]:
                lshape_band_new = _slider(
                    "Support Band [m]", "ls_band", 0.05,
                    settings.LSHAPE_SUPPORT_BAND_MAX,
                    float(run_lshape.get("support_band",
                                         settings.LSHAPE_SUPPORT_BAND_DEFAULT)), 0.05)
            with common_cols[1]:
                lshape_violation_new = _slider(
                    "Violation Weight", "ls_violation", 0.0,
                    settings.LSHAPE_VIOLATION_WEIGHT_MAX,
                    float(run_lshape.get("violation_weight",
                                         settings.LSHAPE_VIOLATION_WEIGHT_DEFAULT)), 0.25)
            st.caption("Depth")
            ld_cols = st.columns(4)
            with ld_cols[0]:
                d_lbin_new = _slider(
                    "Azimuth Bin [deg]", "d_lbin", 0.1, 5.0,
                    float(run_depth.get("lshape_bin_deg",
                                        settings.DEPTH_LSHAPE_BIN_DEG_DEFAULT)), 0.1)
            with ld_cols[1]:
                d_lq_new = _slider(
                    "Near Quantile [%]", "d_lq", 0.0, 50.0,
                    float(run_depth.get("lshape_quantile",
                                        settings.DEPTH_LSHAPE_QUANTILE_DEFAULT)), 1.0)
            with ld_cols[2]:
                d_lmin_new = _slider(
                    "Margin Min [m]", "d_lmin", 0.05, 2.0,
                    float(run_depth.get("lshape_margin_min",
                                        settings.DEPTH_LSHAPE_MARGIN_MIN_DEFAULT)), 0.05)
            with ld_cols[3]:
                d_lratio_new = _slider(
                    "Margin Ratio", "d_lratio", 0.0, 0.2,
                    float(run_depth.get("lshape_margin_ratio",
                                        settings.DEPTH_LSHAPE_MARGIN_RATIO_DEFAULT)), 0.01)
            st.caption("LiDAR")
            ll_cols = st.columns(4)
            with ll_cols[0]:
                l_lbin_new = _slider(
                    "Azimuth Bin [deg]", "l_lbin", 0.1, 5.0,
                    float(run_lidar.get("lshape_bin_deg",
                                        settings.LIDAR_LSHAPE_BIN_DEG_DEFAULT)), 0.1)
            with ll_cols[1]:
                l_lq_new = _slider(
                    "Near Quantile [%]", "l_lq", 0.0, 50.0,
                    float(run_lidar.get("lshape_quantile",
                                        settings.LIDAR_LSHAPE_QUANTILE_DEFAULT)), 1.0)
            with ll_cols[2]:
                l_lmargin_new = _slider(
                    "Margin [m]", "l_lmargin", 0.05, 2.0,
                    float(run_lidar.get("lshape_margin",
                                        settings.LIDAR_LSHAPE_MARGIN_DEFAULT)), 0.05)
            st.markdown("**Distance Filter**")
            run_distance = (
                ((run_info or {}).get("box_fitting_params") or {})
                .get("distance_filter") or {}
            )
            # 方式は run の値（導入前・旧方式の run は config の値で試せるようにする）
            distance_methods = dict(
                run_distance.get("methods") or settings.DISTANCE_FILTER_METHODS
            )
            st.caption("方式: " + distance_methods_text(distance_methods))
            dist_cols = st.columns(4)
            with dist_cols[0]:
                dist_q_new = _slider(
                    "Distance Quantile [%]", "dist_q", 0.0,
                    settings.DISTANCE_FILTER_QUANTILE_MAX,
                    float(run_distance.get(
                        "quantile", settings.DISTANCE_FILTER_QUANTILE_DEFAULT)), 1.0)
            with dist_cols[1]:
                dist_margin_new = _slider(
                    "Distance Margin [m]", "dist_margin", 0.0,
                    settings.DISTANCE_FILTER_MARGIN_MAX,
                    float(run_distance.get(
                        "margin", settings.DISTANCE_FILTER_MARGIN_DEFAULT)), 0.05)
            with dist_cols[2]:
                dist_eps_new = _slider(
                    "Distance DBSCAN eps [m]", "dist_eps", 0.05,
                    settings.DISTANCE_FILTER_DBSCAN_EPS_MAX,
                    float(run_distance.get(
                        "dbscan_eps", settings.DISTANCE_FILTER_DBSCAN_EPS_DEFAULT)),
                    0.05)
            with dist_cols[3]:
                dist_min_new = _slider(
                    "Distance DBSCAN min_samples", "dist_min", 1,
                    settings.DISTANCE_FILTER_DBSCAN_MIN_SAMPLES_MAX,
                    int(run_distance.get(
                        "dbscan_min_samples",
                        settings.DISTANCE_FILTER_DBSCAN_MIN_SAMPLES_DEFAULT)), 1)

            apply_col, reset_col, _spacer = st.columns([1, 1, 6])
            with apply_col:
                if st.button("Apply", type="primary", width="stretch",
                             disabled=not view_run_id):
                    # 対象ラベルや内部の定数（境界の分位点など）は run の値を
                    # 引き継ぎ、UI で変えた値だけ上書きする。L 字の設定が無い
                    # run（導入前）には config の値を補う
                    lshape_base = {
                        "labels": list(settings.LSHAPE_FILTER_LABELS),
                        "angle_step_deg": float(settings.LSHAPE_ANGLE_STEP_DEG),
                        "adjust_range_deg": float(settings.LSHAPE_ADJUST_RANGE_DEG),
                        "adjust_step_deg": float(settings.LSHAPE_ADJUST_STEP_DEG),
                        "boundary_quantile": float(settings.LSHAPE_BOUNDARY_QUANTILE),
                        **run_lshape,
                    }
                    st.session_state[PC_REFILTER_PARAMS] = {
                        "depth": {
                            "ror_nb_points": int(depth_ror_nb_new),
                            "ror_radius": float(depth_ror_radius_new),
                            "dbscan_eps": float(depth_eps_new),
                            "dbscan_min_samples": int(depth_min_new),
                            # スライダーは無いが、run の値を引き継がないと
                            # 推論時に残った小物体の点が消える
                            "small_mask_nb_points_ratio": float(run_depth.get(
                                "small_mask_nb_points_ratio", 1.0)),
                            "lshape_bin_deg": float(d_lbin_new),
                            "lshape_quantile": float(d_lq_new),
                            "lshape_margin_min": float(d_lmin_new),
                            "lshape_margin_ratio": float(d_lratio_new),
                        },
                        "lidar": {
                            "ror_nb_points": int(lidar_ror_nb_new),
                            "ror_radius": float(lidar_ror_radius_new),
                            "dbscan_eps": float(lidar_eps_new),
                            "dbscan_min_samples": int(lidar_min_new),
                            # この 2 つは Filter Params にスライダーが無い。
                            # run の値を引き継がないと推論時と条件が変わり、
                            # 推論では消えたインスタンスが復活してしまう
                            "min_points": int(run_lidar.get(
                                "min_points",
                                settings.LIDAR_MIN_POINTS_DEFAULT)),
                            "max_points_for_depth": int(run_lidar.get(
                                "max_points_for_depth",
                                settings.LIDAR_MAX_POINTS_FOR_DEPTH_DEFAULT)),
                            "lshape_bin_deg": float(l_lbin_new),
                            "lshape_quantile": float(l_lq_new),
                            "lshape_margin": float(l_lmargin_new),
                        },
                        "lshape": {
                            **lshape_base,
                            "enabled": bool(lshape_on_new),
                            "support_band": float(lshape_band_new),
                            "violation_weight": float(lshape_violation_new),
                        },
                        # 想定サイズ（box_sizes）は送らない。サービスが run の
                        # 値で補う（導入前の run は config の値）
                        "distance": {
                            "methods": distance_methods,
                            "quantile": float(dist_q_new),
                            "margin": float(dist_margin_new),
                            "dbscan_eps": float(dist_eps_new),
                            "dbscan_min_samples": int(dist_min_new),
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
                lshape=lshape_items,
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
            if show_lshape and lshape_missing:
                st.caption(
                    f"L 字の可視化情報が無いインスタンスが {lshape_missing} 件あります"
                    "（可視化に対応する前の run）。再推論するか、Filter Params で "
                    "Apply すると表示されます"
                )
            if refilter_summary is not None:
                s = refilter_summary
                raw_total, kept_total, n_inst = s["raw"], s["kept"], s["instances"]
                lidar_raw, lidar_kept = s["lidar_raw"], s["lidar_kept"]
                stored_raw, stored_kept = s["stored_raw"], s["stored_kept"]
                stored_lidar_raw = s["stored_lidar_raw"]
                stored_lidar_kept = s["stored_lidar_kept"]
                ratio = kept_total / raw_total * 100 if raw_total else 0.0
                lidar_text = (
                    f" / LiDAR {lidar_raw:,} → {lidar_kept:,} 点"
                    if lidar_raw else ""
                )
                head = (
                    "推論時と同じパラメータで途中段階を再計算して表示中"
                    if auto_refilter else "再フィルタ適用中"
                )
                st.caption(
                    f"{head}: {n_inst} インスタンス / "
                    f"深度 {raw_total:,} → {kept_total:,} 点 ({ratio:.0f}% 残存)"
                    f"{lidar_text}。"
                    "この結果は保存されません（run の記録は実行時のまま）"
                )
                applied = st.session_state.get(PC_REFILTER_PARAMS)
                if applied and pc_color_mode == COLOR_MODE_GLOBAL:
                    st.caption(
                        "Global Track の色分けは推論時のカメラ間結合の結果です"
                        "（再フィルタでは結合をやり直さないため、Apply した条件で"
                        "結合がどう変わるかは反映されません）"
                    )
                if applied and run_info and lshape_params_differ(applied, run_info):
                    st.caption(
                        "L 字フィルタの設定が推論時と異なるため、"
                        "下の比較の「L 字除去」は一致しなくて正常です"
                    )
                if applied and run_info and distance_params_differ(applied, run_info):
                    st.caption(
                        "距離フィルタの設定が推論時と異なるため、"
                        "下の比較の「距離除去」は一致しなくて正常です"
                    )
                # 推論時との比較。**raw が一致するかで原因を切り分けられる**
                # 一致しない → マスク・深度の範囲（max_depth）が違う
                # raw だけ一致 → 外れ値除去の条件かラベルの倍率が違う
                def _compare(name: str, stored: int, now: int) -> str:
                    mark = "一致" if stored == now else "不一致"
                    return f"{name} {stored:,} → {now:,}（{mark}）"

                if stored_raw or stored_lidar_raw:
                    lines = []
                    if stored_raw or raw_total:
                        lines.append(
                            "深度 " + _compare("raw", stored_raw, raw_total)
                            + " / " + _compare("kept", stored_kept, kept_total)
                        )
                    if stored_lidar_raw or lidar_raw:
                        lines.append(
                            "LiDAR "
                            + _compare("raw", stored_lidar_raw, lidar_raw)
                            + " / "
                            + _compare("kept", stored_lidar_kept, lidar_kept)
                        )
                    if any(s[k] for k in ("lshape_depth", "lshape_lidar",
                                          "stored_lshape_depth",
                                          "stored_lshape_lidar")):
                        lines.append(
                            "L 字除去 "
                            + _compare("深度", s["stored_lshape_depth"], s["lshape_depth"])
                            + " / "
                            + _compare("LiDAR", s["stored_lshape_lidar"], s["lshape_lidar"])
                        )
                    if any(s[k] for k in ("distance_depth", "distance_lidar",
                                          "stored_distance_depth",
                                          "stored_distance_lidar")):
                        lines.append(
                            "距離除去 "
                            + _compare("深度", s["stored_distance_depth"],
                                       s["distance_depth"])
                            + " / "
                            + _compare("LiDAR", s["stored_distance_lidar"],
                                       s["distance_lidar"])
                        )
                    st.caption("推論時との比較: " + " ｜ ".join(lines))

# ------------------------------------------------------------------
# Box Fitting タブ
# ------------------------------------------------------------------
with fitting_tab_view:
    bf_view_col, bf_opt_col = st.columns([8, 1])

    with bf_opt_col:
        render_sample_token(selected_sample["token"])
        compare_gt = st.checkbox(
            "Compare with GT", value=False, key="_w_bf_compare_gt",
            help="GT ボックスの BEV を左に並べて表示する",
        )
        bf_has_lidar = bool(run_info and run_info["use_lidar"])
        show_bf_lidar = st.checkbox(
            "LiDAR Instances", value=False,
            # PointCloud タブと同じ理由で run ごとにキーを分ける
            key=f"_w_bf_lidar_{view_run_id}",
            disabled=not bf_has_lidar,
            help=None if bf_has_lidar else "この run は Use LiDAR=False で実行されています",
        )
        show_bf_lidar = show_bf_lidar and bf_has_lidar
        show_bf_depth = st.checkbox(
            "Depth Instances", value=True, key="_w_bf_depth")
        show_hull = st.checkbox(
            "Convex-Hull", value=False, key="_w_bf_hull",
            help="当てはめに使った凸包を重ねる")
        S.init_sticky(W_BF_DEPTH_INST, OPT_BF_DEPTH_INST, DEPTH_INSTANCE_CORRECTED)
        bf_depth_mode = st.radio(
            "Depth Instance", DEPTH_INSTANCE_MODES, key=W_BF_DEPTH_INST,
            on_change=S.sync_sticky, args=(W_BF_DEPTH_INST, OPT_BF_DEPTH_INST),
            help=("深度点群を補正前（raw）と、LiDAR を基準に補正した後"
                  "（LiDAR corrected）のどちらで表示するか。"
                  "**当てはめは補正後の点群で行っている**ので、"
                  "ボックスと見比べるなら LiDAR corrected を選ぶ"),
        )

        S.init_sticky(W_BF_COLOR, OPT_BF_COLOR, COLOR_MODE_LABEL)
        bf_color_mode = st.radio(
            "Instance Color", BOXFIT_COLOR_MODES, key=W_BF_COLOR,
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
        # **実際に当てはめへ使った点群だけを出す。**
        # 深度点群を使わなかったインスタンス（LiDAR が十分）は深度を出さない。
        # 補正後で見るときは、補正できなかったものだけ補正前へ落とす
        # （元の dict は書き換えない。キャッシュ共有のため）
        bf_flat = [
            {**fit, "_depth_display": (
                None if not fit.get("depth_used", True)
                else (
                    (fit.get("points_depth_corrected_ego")
                     or fit.get("points_depth_ego"))
                    if bf_depth_mode == DEPTH_INSTANCE_CORRECTED
                    else fit.get("points_depth_ego")
                )
            )}
            for fit in bf_flat
        ]
        bf_depth_key = "_depth_display"

        def _bf_color(item: dict) -> str:
            return instance_color(item, bf_color_mode)

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
            for fit in bf_flat if has_box(fit)
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
                bf_flat, color_mode=bf_color_mode, points_key=bf_depth_key,
                key_fn=lambda f: instance_legend_key(f, bf_color_mode),
                color_fn=lambda k: instance_color(
                    {"global_track_id": k, "track_id": k, "label": k},
                    bf_color_mode))
        if show_bf_lidar:
            bf_groups += group_instance_points(
                bf_flat, color_mode=bf_color_mode, points_key="points_lidar_ego",
                # 深度点群と同じ軸で色分けする。渡さないと Label へ落ちる
                key_fn=lambda f: instance_legend_key(f, bf_color_mode),
                color_fn=lambda k: instance_color(
                    {"global_track_id": k, "track_id": k, "label": k},
                    bf_color_mode),
                symbol=SYMBOL_LIDAR, size=MARKER_SIZE_LIDAR)

        # GT は sample 単位。**基準 ego 座標へ変換する**。
        # 推定ボックスが基準 ego にあるので、カメラごとの ego_pose を使うと
        # 時刻差ぶんずれて並べた比較が成立しない
        gt_boxes = []
        if compare_gt:
            ego_pose = reference_ego_pose or next(
                (f["ego_pose"] for f in frames if f.get("ego_pose")), None
            )
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
            render_merge_diagnostics(run_info)
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

# ------------------------------------------------------------------
# Box on Cam タブ（3D ボックスをカメラ画像へ投影）
# ------------------------------------------------------------------
with cam_tab_view:
    cam_view_col, cam_opt_col = st.columns([8, 1])

    with cam_opt_col:
        render_sample_token(selected_sample["token"])
        cam_compare_gt = st.checkbox(
            "Compare with GT", value=False, key="_w_cam_compare_gt",
            help="GT の 3D ボックスを左に並べて表示する",
        )
        S.init_sticky(W_CAM_COLOR, OPT_CAM_COLOR, COLOR_MODE_LABEL)
        cam_color_mode = st.radio(
            "Instance Color", BOXFIT_COLOR_MODES, key=W_CAM_COLOR,
            on_change=S.sync_sticky, args=(W_CAM_COLOR, OPT_CAM_COLOR),
        )
        S.init_sticky(W_CAM_TEXT, OPT_CAM_TEXT, TEXT_MODE_TRACK)
        cam_text_mode = st.radio(
            "Instance text", BOX3D_TEXT_MODES, key=W_CAM_TEXT,
            on_change=S.sync_sticky, args=(W_CAM_TEXT, OPT_CAM_TEXT),
        )

    if not view_run_id:
        with cam_view_col:
            st.info("保存済みの推論結果がありません。")
    else:
        cam_fittings = load_box_fittings(view_run_id, frame_tokens)
        cam_flat = [f for items_ in cam_fittings.values() for f in items_
                    if has_box(f)]

        # 凡例は config のラベル定義から出す。検出結果に含まれるものだけを
        # 出すと、推論のたびに並びとチェック状態が変わってしまう
        with cam_opt_col:
            counts: dict[str, int] = {}
            for fit in cam_flat:
                counts[fit["label"]] = counts.get(fit["label"], 0) + 1
            enabled_labels = render_label_legend(
                all_labels(), counts=counts,
                key_prefix="boxfit_cam_legend",
            )

        shown = [f for f in cam_flat if f["label"] in enabled_labels]

        # GT はカメラごとの ego_pose で変換する。sample_data の時刻が
        # カメラ間でわずかに違うため、1 つの ego_pose で済ませると
        # 投影がずれる
        annotations = (
            list_gt_annotations(dataset_id, selected_sample["token"])
            if cam_compare_gt else []
        )

        cam_items = []
        for sensor in cam_sensors:
            frame = frames_by_channel.get(sensor["channel"])
            if frame is None:
                continue
            image, _frame = get_keyframe_image(
                dataset_id, dataset["dataroot"],
                selected_sample["token"], sensor["token"],
            )
            calib = frame["calibrated_sensor"]
            width, height = frame["width"], frame["height"]

            # そのカメラのフレームで推定されたボックスだけを描く。
            # 他カメラのボックスは、画角内に入っても別インスタンスなので混ぜない
            boxes = [f for f in shown if f["sample_data_token"] == frame["token"]]

            gt_boxes = []
            if cam_compare_gt and annotations:
                gt_boxes = [
                    box for box in gt_boxes_in_ego(annotations, frame["ego_pose"])
                    if box["label"] in enabled_labels
                    and box_center_in_fov(box, calib, width, height)
                ]

            cam_items.append({
                "channel": sensor["channel"], "image": image,
                "calibrated_sensor": calib,
                "boxes": boxes, "gt_boxes": gt_boxes,
            })

        with cam_view_col:
            st.caption(
                f"推定ボックス {sum(len(i['boxes']) for i in cam_items)} 件"
                + (f" / GT {sum(len(i['gt_boxes']) for i in cam_items)} 件"
                   if cam_compare_gt else "")
            )
            if cam_compare_gt:
                render_box3d_comparison_grid(
                    cam_items, color_mode=cam_color_mode, text_mode=cam_text_mode,
                )
            else:
                render_box3d_grid(
                    cam_items, columns=2,
                    color_mode=cam_color_mode, text_mode=cam_text_mode,
                )

S.render_selection_sidebar(dataset_name=dataset["name"], scene_name=scene["name"])
