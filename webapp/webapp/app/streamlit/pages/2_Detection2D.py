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

## ボックスの見せ方

ラベル文字は画像に焼き込まない。1600x900 を2列で並べると潰れて読めない。
右側の凡例に色見本を出し、チェックで表示・非表示を切り替える。
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
from app.services.detection2d_service import (
    delete_run,
    get_run,
    resolve_display_run,
    save_detection_run,
)
from app.services.sample_service import get_skipped_sample_indices
from app.services.frame_image import get_keyframe_image
from app.streamlit import state as S
from app.streamlit.components.det2d_viewer import (
    TEXT_MODES,
    count_by_label,
    labels_in_items,
    render_camera_comparison_grid,
    render_camera_grid,
    render_label_legend,
)
from app.streamlit.components.waypoint_viewer import render_scene_waypoint_view
from app.streamlit.data_access import (
    clear_caches,
    get_dataset,
    get_scene,
    list_detection_runs,
    list_frames_by_scene,
    list_gt_boxes_2d,
    list_samples,
    list_sensors,
    load_detection_run_boxes,
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
RESULTS = S.DET2D_RESULTS
PARTIAL_SINCE = S.DET2D_PARTIAL_SINCE
SAVED_JOB_ID = S.DET2D_SAVED_JOB_ID
VIEW_RUN_ID = S.DET2D_VIEW_RUN_ID

st.session_state.setdefault(PARTIAL_SINCE, 0)

# --- 初期表示: DB に保存済みの run から読み込む -------------------------------
# 優先順は detection2d_service 側に記述（トラッキングの参照先 → 最新の成功 run）。
# 推論を実行した後は、その結果をそのまま表示し続ける
if RESULTS not in st.session_state:
    run_id, reason = resolve_display_run(dataset_id, scene_token)
    if run_id:
        st.session_state[RESULTS] = load_detection_run_boxes(run_id)
        st.session_state[VIEW_RUN_ID] = run_id
    else:
        st.session_state[RESULTS] = {}
        st.session_state[VIEW_RUN_ID] = None


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
        sample_interval_col, score_threshold_col, nms_threshold_col = st.columns(3)

        with sample_interval_col:
            st.markdown("**Sample Interval**")
            sample_interval = st.number_input(
                "Sample Interval", min_value=1, max_value=10,
                value=settings.DET2D_DEFAULT_SAMPLE_INTERVAL, step=1,
                label_visibility="collapsed",
            )
            sample_indices = get_skipped_sample_indices(len(samples), sample_interval)
            n_groups = len(label_groups())
            st.caption(f"{len(sample_indices)} samples: {sample_indices}")
            st.caption(
                f"推論回数 = {len(sample_indices)} × {n_cameras} cam × {n_groups} grp "
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
            st.table(score_thresholds, border="horizontal", width="content")

        with nms_threshold_col:
            st.markdown("**NMS Threshold**")
            nms_threshold_ratio = st.slider(
                "Ratio", key="nms_threshold_ratio",
                min_value=0.0, max_value=2.0, value=1.0, step=0.05,
            )
            nms_same = scaled_nms_same_class_ious(nms_threshold_ratio)
            nms_cross = min(
                settings.DET2D_NMS_CROSS_CLASS_IOU * nms_threshold_ratio, 1.0
            )
            same_nms_col, cross_nms_col = st.columns([2, 1])
            with same_nms_col:
                st.table(nms_same, border="horizontal", width="content")
            with cross_nms_col:
                st.text(f"Cross-class NMS IOU: {nms_cross:.3f}")

# ------------------------------------------------------------------
# Sample selection & map
# ------------------------------------------------------------------
with map_col:
    # 表示順は「地図 → スライダー」だが、地図はスライダーの値に依存する。
    # Streamlit は上から順に描画するので、先に空のコンテナで場所だけ確保し、
    # スライダーを読んでから、そのコンテナへ遡って描画する。
    # （逆順に書くと、スライダーを動かした値が地図に1回遅れて反映される）
    waypoint_container = st.container()

    default_idx = sample_indices[len(sample_indices) // 2] if sample_indices else 0
    selected_sample_idx = st.select_slider(
        "Select Sample",
        options=sample_indices or [0],
        value=default_idx,
    )

    with waypoint_container:
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
        "nms_cross_class_iou": nms_cross,
        "stub_delay_sec": settings.DET2D_STUB_DELAY_SEC,
    }


def _merge_partial(items: list[dict]) -> None:
    """部分結果を sample_data_token 単位で取り込む."""
    store = st.session_state[RESULTS]
    for item in items:
        store[item["sample_data_token"]] = item.get("boxes", [])


def _save_completed_job(job: dict) -> None:
    """完了したジョブを 1 run として DB に保存する.

    キャンセル・失敗した run も status 付きで残す。
    「実行したが失敗した」記録が消えると、同じ設定で何度も試すことになる。
    """
    try:
        params_id = save_detection_run(
            dataset_id, scene_token,
            job=job,
            boxes_by_frame=st.session_state[RESULTS],
            sample_interval=sample_interval,
            score_threshold=score_thresholds,
            nms_same_class_ious=nms_same,
            nms_cross_class_iou=nms_cross,
            model_name=settings.DET2D_MODEL_NAME,
            # 手修正は「いま表示している run」のものだけ引き継ぐ。
            # 全 run から集めると、古いパラメータ時代の修正が混ざる
            inherit_from_params_id=st.session_state.get(VIEW_RUN_ID),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"検出結果の保存に失敗しました: {exc}")
        return

    # 保存した run を「表示中の run」にし、手修正の反映結果を読み直す
    st.session_state[VIEW_RUN_ID] = params_id
    clear_caches()
    st.session_state[RESULTS] = load_detection_run_boxes(params_id)


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
                # 二重保存の防止。ポーリングは1秒ごとに走るため、
                # 完了を複数回検知しうる
                if st.session_state.get(SAVED_JOB_ID) != job_id:
                    st.session_state[SAVED_JOB_ID] = job_id
                    _save_completed_job(job)

                st.session_state.pop(JOB_ID, None)
                if job["status"] == "failed":
                    st.error(job.get("error") or "推論に失敗しました")
                # フラグメント外（画像グリッド）も更新するため全体を再実行する
                st.rerun()

        with status_col:
            progress_area()

# ------------------------------------------------------------------
# Saved runs
# ------------------------------------------------------------------
with param_col:
    runs = list_detection_runs(dataset_id, scene_token)
    if runs:
        with st.expander(f"保存済みの推論結果（{len(runs)} 件）", expanded=False):
            current = st.session_state.get(VIEW_RUN_ID)

            def _run_label(run: dict) -> str:
                mark = "★ " if run["id"] == current else ""
                started = run["started_at"].strftime("%m-%d %H:%M:%S")
                refs = f" / tracking参照 {run['nbr_tracking_runs']}" if run["nbr_tracking_runs"] else ""
                return (f"{mark}{started}  [{run['status']}]  "
                        f"{run['nbr_boxes']} boxes  interval={run['sample_interval']}{refs}")

            options = [r["id"] for r in runs]
            labels = {r["id"]: _run_label(r) for r in runs}
            selected_run = st.radio(
                "表示する run", options,
                index=options.index(current) if current in options else 0,
                format_func=lambda rid: labels[rid],
                key="_w_det2d_run_select",
            )

            show_col, delete_col = st.columns(2)
            with show_col:
                if st.button("この run を表示", width="content",
                             disabled=selected_run == current):
                    st.session_state[VIEW_RUN_ID] = selected_run
                    st.session_state[RESULTS] = load_detection_run_boxes(selected_run)
                    st.rerun()
            with delete_col:
                target = next(r for r in runs if r["id"] == selected_run)
                if target["nbr_tracking_runs"]:
                    # この run を消すと、参照している Instance Tracking の結果と
                    # そこから作られた 3D ボックスまで CASCADE で消える
                    st.warning(
                        f"この run は Instance Tracking {target['nbr_tracking_runs']} 件から"
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
# Predicted bounding boxes view
# ------------------------------------------------------------------
st.divider()

view_col, opt_col = st.columns([9, 1])

# 凡例に件数を出すため、画像とボックスの取得を先に済ませる
selected_sample = samples[selected_sample_idx]
results = st.session_state[RESULTS]

# 表示オプションは正規キーに保存しておく。
# 「Run Inference」で st.rerun() が走ると、その下のウィジェットは
# その実行で生成されず、Streamlit に状態を破棄されてしまうため
OPT_SHOW_BOXES, W_SHOW_BOXES = "det2d_show_boxes", "_w_det2d_show_boxes"
OPT_SHOW_GT,    W_SHOW_GT    = "det2d_show_gt",    "_w_det2d_show_gt"
OPT_MIN_SCORE,  W_MIN_SCORE  = "det2d_min_score",  "_w_det2d_min_score"
OPT_TEXT_MODE,  W_TEXT_MODE  = "det2d_text_mode",  "_w_det2d_text_mode"

# GT との並列表示。チェックすると 1 カメラ 1 行で左右に並べる
show_gt = S.sticky_value(OPT_SHOW_GT, False)

items = []
for sensor in cam_sensors:
    image, frame = get_keyframe_image(
        dataset_id, dataset["dataroot"], selected_sample["token"], sensor["token"]
    )
    token = frame["token"] if frame else None
    item = {
        "channel": sensor["channel"],
        "image": image,
        "boxes": results.get(token, []),
        "pending": token not in results,
    }
    if show_gt:
        # 3D→2D の投影はキャッシュされるので、再実行のたびには走らない
        item["gt_boxes"] = list_gt_boxes_2d(
            dataset_id, selected_sample["token"], sensor["token"]
        )
    items.append(item)

with opt_col:
    S.init_sticky(W_SHOW_BOXES, OPT_SHOW_BOXES, True)
    show_boxes = st.checkbox(
        "Show boxes", key=W_SHOW_BOXES,
        on_change=S.sync_sticky, args=(W_SHOW_BOXES, OPT_SHOW_BOXES),
    )
    S.init_sticky(W_SHOW_GT, OPT_SHOW_GT, False)
    st.checkbox(
        "Compare with GT", key=W_SHOW_GT,
        on_change=S.sync_sticky, args=(W_SHOW_GT, OPT_SHOW_GT),
        help="1カメラ1行で、左に Ground truth・右に推論結果を並べる",
    )
    S.init_sticky(W_MIN_SCORE, OPT_MIN_SCORE, 0.0)
    min_score = st.slider(
        "Min score", 0.0, 1.0, step=0.05, key=W_MIN_SCORE,
        on_change=S.sync_sticky, args=(W_MIN_SCORE, OPT_MIN_SCORE),
    )
    # 画像に重ねる文字。既定は None（枠だけ）。
    # 文字色は枠線と同じにするので、凡例の色と対応が付く
    S.init_sticky(W_TEXT_MODE, OPT_TEXT_MODE, TEXT_MODES[0])
    text_mode = st.radio(
        "Box text", TEXT_MODES, key=W_TEXT_MODE,
        on_change=S.sync_sticky, args=(W_TEXT_MODE, OPT_TEXT_MODE),
    )

    # ラベルは画像に焼き込まず、この凡例の色で判別する。
    # 表示するのは「いま選んでいる sample に出ているラベル」だけ。
    # GT を並べているときは GT 側のラベルも凡例に含める
    legend_source = items
    if show_gt:
        legend_source = [
            {**item, "boxes": (item.get("boxes") or []) + (item.get("gt_boxes") or [])}
            for item in items
        ]
    legend_labels = labels_in_items(legend_source, min_score=0.0)
    if legend_labels:
        enabled_labels = render_label_legend(
            legend_labels,
            counts=count_by_label(legend_source, min_score=0.0),
        )
    else:
        # 凡例が空でも、他 sample の結果を消さないよう None にする
        # （空集合を渡すと「全ラベル非表示」の意味になってしまう）
        enabled_labels = None
        st.caption("このサンプルには表示できるボックスがありません")

with view_col:
    if show_gt:
        render_camera_comparison_grid(
            items,
            min_score=min_score,
            enabled_labels=enabled_labels,
            show_boxes=show_boxes,
            text_mode=text_mode,
        )
    else:
        render_camera_grid(
            items,
            columns=2,
            min_score=min_score,
            enabled_labels=enabled_labels,
            show_boxes=show_boxes,
            text_mode=text_mode,
        )

S.render_selection_sidebar(dataset_name=dataset["name"], scene_name=scene["name"])
