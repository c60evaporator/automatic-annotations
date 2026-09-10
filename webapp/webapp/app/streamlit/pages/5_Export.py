"""nuScenes 形式へのエクスポート.

自動生成した 3D アノテーションを、devkit が読める JSON テーブルとして出す。
画像や LiDAR のファイルはコピーしない（元データが大きく、複製の意味がない）。
出力した JSON は元データセットと同じ dataroot に置いて使う。
"""
from __future__ import annotations

import streamlit as st

from app.core.config import get_settings
from app.repositories.nuscenes_export import (
    ANNOTATION_SOURCE_AUTO,
    ANNOTATION_SOURCE_BOTH,
    ANNOTATION_SOURCE_GT,
)
from app.services.nuscenes_export_service import cleanup_export, export_nuscenes
from app.streamlit import state as S
from app.streamlit.data_access import (
    get_dataset,
    list_boxfitting_runs,
    list_scenes,
)

# 表示名 → サービスへ渡す値
SOURCE_LABELS = {
    "自動生成のみ（source=auto）": ANNOTATION_SOURCE_AUTO,
    "GT のみ（source=imported）": ANNOTATION_SOURCE_GT,
    "両方": ANNOTATION_SOURCE_BOTH,
}
# エクスポート結果を保持する（ダウンロードボタンの再描画で消えないように）
EXPORT_RESULT = "sel_export_result"

dataset_id = S.require_dataset()
dataset = get_dataset(dataset_id)
scenes = list_scenes(dataset_id)
settings = get_settings()

st.subheader("nuScenes 形式へのエクスポート")

param_col, info_col = st.columns([2, 1])

with param_col:
    scene_mode = st.radio(
        "対象シーン", ["選択したシーンのみ", "データセット全体"],
        horizontal=True,
    )
    if scene_mode == "選択したシーンのみ":
        default = st.session_state.get(S.SCENE_TOKEN)
        options = [s["token"] for s in scenes]
        labels = {s["token"]: f"{s['name']}（{s['nbr_samples']} samples）"
                  for s in scenes}
        selected_scenes = st.multiselect(
            "シーン", options,
            default=[default] if default in options else options[:1],
            format_func=lambda t: labels[t],
        )
    else:
        selected_scenes = None

    source_label = st.radio("アノテーション", list(SOURCE_LABELS), index=0)
    source = SOURCE_LABELS[source_label]

    # 自動生成を含む場合のみ、run を絞れるようにする
    run_id = None
    if source in (ANNOTATION_SOURCE_AUTO, ANNOTATION_SOURCE_BOTH):
        scene_for_runs = (
            selected_scenes[0] if selected_scenes
            else (scenes[0]["token"] if scenes else None)
        )
        runs = list_boxfitting_runs(dataset_id, scene_for_runs) if scene_for_runs else []
        succeeded = [r for r in runs if r["status"] == "succeeded"]
        if succeeded:
            run_options = [None] + [r["id"] for r in succeeded]
            run_labels = {r["id"]: (
                f"{r['started_at'].strftime('%m-%d %H:%M:%S')}  "
                f"{r['nbr_annotations']} annotations"
            ) for r in succeeded}
            run_id = st.selectbox(
                "Box Fitting run で絞る", run_options,
                format_func=lambda rid: "すべての run" if rid is None else run_labels[rid],
            )
        elif source == ANNOTATION_SOURCE_AUTO:
            st.warning(
                "自動生成のアノテーションがありません。"
                "先に Depth Estimation & Box Fitting を実行してください。"
            )

    version = st.text_input(
        "バージョン名", value="v1.0-auto",
        help="出力ディレクトリ名になる（nuScenes の version に相当）",
    )

with info_col:
    st.caption(
        "**出力されるもの**\n\n"
        "devkit が読む JSON テーブル 13 種類。"
        "画像・LiDAR ファイルは含みません。\n\n"
        "**使い方**\n\n"
        "解凍したフォルダを元データセットの dataroot に置き、"
        "`NuScenes(version=..., dataroot=...)` で読み込みます。"
    )

st.divider()

if st.button("エクスポート", type="primary", disabled=not scenes):
    try:
        with st.spinner("テーブルを組み立て中..."):
            result = export_nuscenes(
                dataset_id,
                scene_tokens=selected_scenes,
                source=source,
                depth_estimation_params_id=run_id,
                version=version or "v1.0-auto",
            )
        st.session_state[EXPORT_RESULT] = result
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"エクスポートに失敗しました: {exc}")

result = st.session_state.get(EXPORT_RESULT)
if result:
    counts = result["counts"]
    st.success(f"{result['version']} を書き出しました")

    # 主要なテーブルだけ先に見せる。全件は下の表で確認できる
    metric_cols = st.columns(4)
    for col, key in zip(
        metric_cols, ["sample_annotation", "instance", "sample", "sample_data"]
    ):
        with col:
            st.metric(key, f"{counts.get(key, 0):,}")

    with st.expander("テーブルごとの件数", expanded=False):
        st.dataframe(
            [{"table": name, "rows": counts.get(name, 0)} for name in counts],
            width="stretch", hide_index=True,
        )

    zip_path = result["zip_path"]
    if zip_path.exists():
        size_mb = zip_path.stat().st_size / 1024 / 1024
        # download_button はファイル全体をメモリに載せる。
        # JSON だけなので数十 MB に収まる前提
        st.download_button(
            f"ダウンロード（{size_mb:.1f} MB）",
            data=zip_path.read_bytes(),
            file_name=f"{result['version']}.zip",
            mime="application/zip",
            type="primary",
        )
    st.caption(f"サーバー上の出力先: `{result['directory']}`")

    if st.button("この出力を削除", width="content"):
        cleanup_export(result["name"])
        st.session_state.pop(EXPORT_RESULT, None)
        st.rerun()

S.render_selection_sidebar(dataset_name=dataset["name"])
