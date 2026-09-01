"""basemap 上に自車軌跡（ego pose）を描画するコンポーネント.

旧実装からの変更点:
  - nuScenes devkit の nusc オブジェクトではなく、Repository が返す
    dict のリストを受け取る。
  - canvas_edge のハードコード辞書を廃止。DB の MapMeta から来る。
  - basemap が無い場合も軌跡だけ描画して落ちないようにした。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from app.services.basemap_service import DEFAULT_SCALE
from app.streamlit.components.basemap import (
    SceneBasemap,
    get_scene_basemap,
    render_basemap_notice,
)
from app.streamlit.data_access import list_waypoints


def format_timestamp(ts_usec: int, *, with_date: bool = True) -> str:
    """nuScenes の UNIX マイクロ秒を読める形式にする."""
    dt = datetime.fromtimestamp(ts_usec / 1_000_000, tz=timezone.utc)
    fmt = "%Y-%m-%d %H:%M:%S.%f" if with_date else "%H:%M:%S.%f"
    return dt.strftime(fmt)[:-3] + " UTC"


def add_basemap(
    fig: go.Figure,
    basemap_img: Image.Image | None,
    canvas_edge: Sequence[float] | None,
    opacity: float = 0.8,
) -> None:
    """basemap をプロットの背面に敷く.

    nuScenes のマップ座標は原点が左下、canvas_edge が [幅 m, 高さ m]。
    Plotly の layout image は左上を基準に置くため、y に canvas_edge[1] を渡す。
    """
    if basemap_img is None or not canvas_edge:
        return
    fig.add_layout_image(
        source=basemap_img,
        xref="x", yref="y",
        x=0, y=canvas_edge[1],
        sizex=canvas_edge[0], sizey=canvas_edge[1],
        sizing="stretch",
        opacity=opacity,
        layer="below",
    )


def plot_waypoints(
    fig: go.Figure,
    x: Sequence[float],
    y: Sequence[float],
    *,
    plot_start_end_points: bool = True,
    start_color: str = "green",
    end_color: str = "red",
    start_end_size: int = 14,
    scatter_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """軌跡と開始点・終了点を描画する."""
    fig.add_trace(go.Scatter(x=list(x), y=list(y), **(scatter_kwargs or {})))
    if plot_start_end_points and len(x):
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]], mode="markers",
            marker=dict(size=start_end_size, color=start_color, symbol="circle"),
            name="Start",
        ))
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]], mode="markers",
            marker=dict(size=start_end_size, color=end_color, symbol="circle"),
            name="End",
        ))
    return fig


def build_waypoint_figure(
    waypoints: list[dict[str, Any]],
    *,
    title: str = "",
    basemap_img: Image.Image | None = None,
    canvas_edge: Sequence[float] | None = None,
    highlight_index: int | None = None,
    height: int = 480,
    fit_to_trajectory: bool = True,
    margin_m: float = 100.0,
) -> go.Figure:
    """自車軌跡の Figure を組み立てる（描画はしない）.

    Figure の生成と描画を分けているのは、テストしやすくするためと、
    同じ Figure を別のレイアウトで使い回せるようにするため。
    """
    fig = go.Figure()
    add_basemap(fig, basemap_img, canvas_edge)

    xs = [w["x"] for w in waypoints]
    ys = [w["y"] for w in waypoints]

    plot_waypoints(
        fig, xs, ys,
        scatter_kwargs={
            "mode": "lines+markers",
            "line": dict(width=2),
            "marker": dict(size=5),
            "name": "Trajectory",
            "customdata": [[w["sample_idx"], w["sample_token"]] for w in waypoints],
            "hovertemplate": (
                "Sample: %{customdata[0]}<br>"
                "X: %{x:.2f}<br>"
                "Y: %{y:.2f}<extra></extra>"
            ),
        },
    )

    if highlight_index is not None and 0 <= highlight_index < len(waypoints):
        fig.add_trace(go.Scatter(
            x=[xs[highlight_index]], y=[ys[highlight_index]],
            mode="markers",
            marker=dict(size=12, color="orange", symbol="circle",
                        line=dict(width=2, color="black")),
            name="Selected Sample",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Global X [m]",
        yaxis_title="Global Y [m]",
        height=height,
        hovermode="closest",
        showlegend=True,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    # X,Y のスケールを揃える（地図なので必須）
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # マップ全体だと軌跡が点にしか見えないので、軌跡の周辺に寄せる。
    # basemap は layout image なので、ズームしても正しい位置に残る。
    if fit_to_trajectory and xs:
        fig.update_xaxes(range=[min(xs) - margin_m, max(xs) + margin_m])
        fig.update_yaxes(range=[min(ys) - margin_m, max(ys) + margin_m])

    return fig


def render_scene_waypoint(
    waypoints: list[dict[str, Any]],
    *,
    title: str = "",
    basemap_img: Image.Image | None = None,
    canvas_edge: Sequence[float] | None = None,
    highlight_index: int | None = None,
    show_sample_info: bool = True,
    height: int = 480,
) -> None:
    """シーンの自車軌跡を描画する（データは呼び出し側が用意する）."""
    if not waypoints:
        st.info("このシーンには表示できる自車位置がありません。")
        return

    fig = build_waypoint_figure(
        waypoints,
        title=title or f"{len(waypoints)} samples",
        basemap_img=basemap_img,
        canvas_edge=canvas_edge,
        highlight_index=highlight_index,
        height=height,
    )
    st.plotly_chart(fig, width="stretch")

    if show_sample_info and highlight_index is not None \
            and 0 <= highlight_index < len(waypoints):
        wp = waypoints[highlight_index]
        st.markdown(
            f"**Time:** {format_timestamp(wp['timestamp'])}  \n"
            f"**Sample token:** `{wp['sample_token']}`"
        )


# ── 取得込みの高レベル部品 ────────────────────────────────────────────────────
#
# ここから下は data_access / basemap コンポーネントに依存し、
# 「シーンを指定するだけで軌跡ビューが出る」ところまでを引き受ける。
# 上の純粋な関数群（build_waypoint_figure など）は依存を持たないので、
# 単体テストや別データ源からの描画にはそちらを使う。


def build_scene_waypoint_figure(
    dataset_id: str,
    dataroot: str,
    scene_token: str,
    *,
    title: str = "",
    highlight_index: int | None = None,
    height: int = 480,
    scale: float = DEFAULT_SCALE,
    **figure_kwargs: Any,
) -> tuple[go.Figure, list[dict[str, Any]], SceneBasemap]:
    """シーンの軌跡 Figure を組み立てて返す（描画はしない）.

    Figure を返すのは、後段のページで検出ボックスやマスクの trace を
    重ねてから描画したいケースがあるため。

    Returns:
        (figure, waypoints, basemap)
    """
    basemap = get_scene_basemap(dataset_id, dataroot, scene_token, scale)
    waypoints = list_waypoints(dataset_id, scene_token)
    fig = build_waypoint_figure(
        waypoints,
        title=title or f"{len(waypoints)} samples",
        basemap_img=basemap.image,
        canvas_edge=basemap.canvas_edge,
        highlight_index=highlight_index,
        height=height,
        **figure_kwargs,
    )
    return fig, waypoints, basemap


def render_scene_waypoint_view(
    dataset_id: str,
    dataroot: str,
    scene_token: str,
    *,
    title: str = "",
    highlight_index: int | None = None,
    show_sample_info: bool = True,
    show_notice: bool = True,
    height: int = 480,
    scale: float = DEFAULT_SCALE,
) -> list[dict[str, Any]]:
    """シーンを指定するだけで軌跡ビュー一式を描画する.

    basemap の解決・waypoint の取得・描画・注意書きまでをまとめて行う。
    複数ページで同じブロックを書き写さずに済ませるための入口。

    Returns:
        描画に使った waypoints（呼び出し側で sample 選択などに使える）
    """
    basemap = get_scene_basemap(dataset_id, dataroot, scene_token, scale)
    if show_notice:
        render_basemap_notice(basemap)

    waypoints = list_waypoints(dataset_id, scene_token)
    render_scene_waypoint(
        waypoints,
        title=title,
        basemap_img=basemap.image,
        canvas_edge=basemap.canvas_edge,
        highlight_index=highlight_index,
        show_sample_info=show_sample_info,
        height=height,
    )
    return waypoints
