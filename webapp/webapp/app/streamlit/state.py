"""Streamlit のセッション状態管理.

## なぜウィジェットキーと正規キーを分けるのか

Streamlit のマルチページでは、**ウィジェットに紐づく session_state のキーは、
そのウィジェットが描画されない実行で破棄される**。
つまり main.py の `selectbox(key="dataset")` で選んだ値は、
pages/1_Scene_Selection.py に移動した時点で消える可能性がある。

そこで「ウィジェットが持つキー（_w_ 接頭辞）」と
「アプリが参照する正規キー（sel_ 接頭辞）」を分離し、
ウィジェットの on_change で正規キーへ書き写す。
正規キーはどのウィジェットにも紐づかないので、ページを跨いでも保持される。

## 使い方

    # 選択側（main.py）
    st.selectbox(..., key=W_DATASET, on_change=on_dataset_change)

    # 参照側（pages/*.py の先頭）
    dataset_id, scene_token = require_scene()
"""
from __future__ import annotations

from typing import Any

import streamlit as st

# ── 正規キー（アプリが参照する。ウィジェットには絶対に使わない） ──────────────
DATASET_ID = "sel_dataset_id"
SCENE_TOKEN = "sel_scene_token"
SAMPLE_IDX = "sel_sample_idx"

# 実行中ジョブの id（推論サーバー側のジョブ）。
# シーンに紐づくので、シーンを変えたらクリアされる必要がある
DET2D_JOB_ID = "sel_det2d_job_id"

# 画面に出している検出結果まわり。いずれもシーンに紐づく。
# DB を正とするので、これらは「今そこに表示している内容」のキャッシュに過ぎない
DET2D_RESULTS = "sel_det2d_results"        # {sample_data_token: [box, ...]}
DET2D_PARTIAL_SINCE = "sel_det2d_since"    # 受け取り済みの部分結果の件数
DET2D_SAVED_JOB_ID = "sel_det2d_saved_job" # 保存済みジョブ（二重保存の防止）
DET2D_VIEW_RUN_ID = "sel_det2d_view_run"   # 表示に使っている run

# 各推論ステップの実行単位（*Params.id）。ページ間で引き継ぐ
DET2D_PARAMS_ID = "sel_det2d_params_id"
TRACKING_PARAMS_ID = "sel_tracking_params_id"
DEPTH_PARAMS_ID = "sel_depth_params_id"

# ── ウィジェットキー（Streamlit が管理する。破棄され得る） ────────────────────
W_DATASET = "_w_dataset"
W_SCENE = "_w_scene"
W_SAMPLE = "_w_sample"

# 上位の選択が変わったら下位をクリアする依存関係。
# 例: データセットを変えたのにシーン選択が残っていると、
#     別データセットの scene_token を参照して空の結果になる。
_CASCADE: dict[str, tuple[str, ...]] = {
    DATASET_ID: (SCENE_TOKEN, SAMPLE_IDX, DET2D_JOB_ID,
                 DET2D_RESULTS, DET2D_PARTIAL_SINCE,
                 DET2D_SAVED_JOB_ID, DET2D_VIEW_RUN_ID,
                 DET2D_PARAMS_ID, TRACKING_PARAMS_ID, DEPTH_PARAMS_ID),
    SCENE_TOKEN: (SAMPLE_IDX, DET2D_JOB_ID,
                  DET2D_RESULTS, DET2D_PARTIAL_SINCE,
                  DET2D_SAVED_JOB_ID, DET2D_VIEW_RUN_ID,
                  DET2D_PARAMS_ID, TRACKING_PARAMS_ID, DEPTH_PARAMS_ID),
    DET2D_PARAMS_ID: (TRACKING_PARAMS_ID, DEPTH_PARAMS_ID),
    TRACKING_PARAMS_ID: (DEPTH_PARAMS_ID,),
}


# ── 読み書き ──────────────────────────────────────────────────────────────────

def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_selection(key: str, value: Any) -> None:
    """正規キーを更新し、下位の選択をクリアする.

    値が変わらないときは何もしない（不要な下位クリアを避けるため）。
    """
    if st.session_state.get(key) == value:
        return
    st.session_state[key] = value
    clear_downstream(key)


def clear_downstream(key: str) -> None:
    """指定キーより下位の選択を再帰的にクリアする."""
    for child in _CASCADE.get(key, ()):
        st.session_state.pop(child, None)
        clear_downstream(child)


def clear_all() -> None:
    """全ての選択状態をリセットする."""
    for key in (DATASET_ID, SCENE_TOKEN, SAMPLE_IDX, DET2D_JOB_ID,
                DET2D_RESULTS, DET2D_PARTIAL_SINCE,
                DET2D_SAVED_JOB_ID, DET2D_VIEW_RUN_ID,
                DET2D_PARAMS_ID, TRACKING_PARAMS_ID, DEPTH_PARAMS_ID):
        st.session_state.pop(key, None)


# ── ウィジェットからの反映 ────────────────────────────────────────────────────

def sync_from_widget(widget_key: str, canonical_key: str) -> None:
    """ウィジェットの現在値を正規キーへ書き写す（on_change から呼ぶ）."""
    set_selection(canonical_key, st.session_state.get(widget_key))


def on_dataset_change() -> None:
    sync_from_widget(W_DATASET, DATASET_ID)


def on_scene_change() -> None:
    sync_from_widget(W_SCENE, SCENE_TOKEN)


# ── 表示オプションの永続化 ────────────────────────────────────────────────────
#
# ページ途中で st.rerun() や st.stop() が走ると、そこから先のウィジェットは
# その実行では生成されない。Streamlit は生成されなかったウィジェットの状態を
# 破棄するため、次の実行で既定値に戻ってしまう。
# 例: 「Run Inference」で st.rerun() すると、その下にある表示オプションが
#     すべてリセットされる。
#
# ウィジェットの key（_w_ 接頭辞）とは別に、破棄されない正規キーへ値を
# 写しておくことで、中断があっても選択が残る。

def init_sticky(widget_key: str, canonical_key: str, default: Any) -> None:
    """ウィジェットの初期値を、保存済みの正規キーから復元する.

    value= を使わず session_state で与えるのは、
    on_change で session_state に書くのと併用したときの警告を避けるため。
    """
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(canonical_key, default)


def sync_sticky(widget_key: str, canonical_key: str) -> None:
    """ウィジェットの値を正規キーへ写す（on_change コールバック）."""
    st.session_state[canonical_key] = st.session_state.get(widget_key)


def sticky_value(canonical_key: str, default: Any = None) -> Any:
    """保存済みの表示オプションを読む（ウィジェット生成前でも使える）."""
    return st.session_state.get(canonical_key, default)


# ── ページガード ──────────────────────────────────────────────────────────────

def _safe_page_link(path: str, label: str, icon: str) -> None:
    """st.page_link を安全に呼ぶ.

    page_link は「アプリに登録済みのページ」しか解決できず、
    ページ単体でのテスト実行や、まだ存在しないページを指した場合に
    KeyError を投げる。ガードが例外で死ぬと本来のメッセージすら
    出なくなるため、失敗しても素通しにする。
    """
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:  # noqa: BLE001
        st.caption(f"{icon} {label}: `{path}`")


def require_dataset() -> str:
    """データセットが選択済みであることを保証する.

    未選択ならメッセージとリンクを出して st.stop() で描画を打ち切る。
    各ページの先頭で1行呼ぶだけで済むので、ページが増えても破綻しない。
    """
    dataset_id = st.session_state.get(DATASET_ID)
    if not dataset_id:
        st.warning("先にデータセットを選択してください。")
        _safe_page_link("main.py", "データセット選択へ", "📂")
        st.stop()
    return dataset_id


def require_scene() -> tuple[str, str]:
    """データセットとシーンが選択済みであることを保証する."""
    dataset_id = require_dataset()
    scene_token = st.session_state.get(SCENE_TOKEN)
    if not scene_token:
        st.warning("先にシーンを選択してください。")
        _safe_page_link("pages/1_Scene_Selection.py", "シーン選択へ", "🎬")
        st.stop()
    return dataset_id, scene_token


def require_detection2d() -> tuple[str, str, str]:
    """2D 検出の実行結果が存在することを保証する（Instance Tracking ページ用）."""
    dataset_id, scene_token = require_scene()
    params_id = st.session_state.get(DET2D_PARAMS_ID)
    if not params_id:
        st.warning("先に 2D Object Detection を実行してください。")
        _safe_page_link("pages/2_Detection2D.py", "2D Object Detection へ", "🔍")
        st.stop()
    return dataset_id, scene_token, params_id


def require_tracking() -> tuple[str, str, str]:
    """Instance Tracking の実行結果が存在することを保証する（Depth ページ用）."""
    dataset_id, scene_token = require_scene()
    params_id = st.session_state.get(TRACKING_PARAMS_ID)
    if not params_id:
        st.warning("先に Instance Tracking を実行してください。")
        _safe_page_link("pages/3_Instance_Tracking.py", "Instance Tracking へ", "🎯")
        st.stop()
    return dataset_id, scene_token, params_id


# ── サイドバー ────────────────────────────────────────────────────────────────

def render_selection_sidebar(dataset_name: str | None = None,
                             scene_name: str | None = None) -> None:
    """現在の選択状態をサイドバーに表示する（全ページ共通）."""
    with st.sidebar:
        st.caption("現在の選択")
        st.write(f"**データセット**: {dataset_name or '未選択'}")
        st.write(f"**シーン**: {scene_name or '未選択'}")
