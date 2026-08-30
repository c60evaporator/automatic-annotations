import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

from utils.scene_utils import get_samples_in_scene
from utils.map_utils import get_basemap_img, get_canvas_edge
from utils.scene_utils import get_sample_utctime

BASEMAP_MARGIN = 10.0  # meters

def render_basemap(nusc,
                   basemap_cache,
                   canvas_edge_config,
                   scene_token,
                   fig
):
    basemap_img = get_basemap_img(nusc, basemap_cache, scene_token)
    canvas_edge = get_canvas_edge(nusc, canvas_edge_config, scene_token)
    fig.add_layout_image(
        source=basemap_img,
        xref="x",
        yref="y",
        x=0,
        y=canvas_edge[1],
        sizex=canvas_edge[0],
        sizey=canvas_edge[1],
        sizing="stretch",
        opacity=0.8,
        layer="below",
    )

def plot_waypoints(
    fig, x, y,
    plot_start_end_points=True,
    start_color="green",
    end_color="red",
    start_end_size=14,
    scatter_kwargs=None,
):
    """
    Plotly上にStart/Endを含めてwaypointを描画する
    """
    # 軌跡をプロット
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            **(scatter_kwargs or {}),
        )
    )
    if plot_start_end_points:
        # 開始点をプロット
        fig.add_trace(
            go.Scatter(
                x=[x.iloc[0] if hasattr(x, "iloc") else x[0]],
                y=[y.iloc[0] if hasattr(y, "iloc") else y[0]],
                mode="markers",
                marker=dict(
                    size=start_end_size,
                    color=start_color,
                    symbol="circle",
                ),
                name="Start",
            )
        )
        # 終了点をプロット
        fig.add_trace(
            go.Scatter(
                x=[x.iloc[-1] if hasattr(x, "iloc") else x[-1]],
                y=[y.iloc[-1] if hasattr(y, "iloc") else y[-1]],
                mode="markers",
                marker=dict(
                    size=start_end_size,
                    color=end_color,
                    symbol="circle",
                ),
                name="End",
            )
        )
    return fig

def render_scene_waypoint(
    nusc,
    basemap_cache,
    canvas_edge_config,
    scene_token,
    highlight_index=None,
):
    """
    scene内の各sampleのego poseを取得して
    basemapに重ねてPlotly上に軌跡を描画する
    """
    scene = nusc.get("scene", scene_token)
    # scene内の全sampleのego poseを取得
    waypoint_df = get_samples_in_scene(nusc, scene_token)

    # basemapを軌跡の範囲で切り出してラスター表示
    basemap_img = get_basemap_img(nusc, basemap_cache, scene_token)
    canvas_edge = get_canvas_edge(nusc, canvas_edge_config, scene_token)
    fig = go.Figure()
    fig.add_layout_image(
        source=basemap_img,
        xref="x",
        yref="y",
        x=0,
        y=canvas_edge[1],
        sizex=canvas_edge[0],
        sizey=canvas_edge[1],
        sizing="stretch",
        opacity=0.8,
        layer="below",
    )

    # 軌跡と開始点・終了点をプロット
    scatter_kwargs = {
        "mode": "lines+markers",
        "line": dict(width=2),
        "marker": dict(size=5),
        "name": "Trajectory",
        "customdata": waypoint_df[["sample_idx"]],
        "hovertemplate": (
            "Sample: %{customdata[0]}<br>"
            "X: %{x:.2f}<br>"
            "Y: %{y:.2f}<extra></extra>"
        ),
    }
    plot_waypoints(
        fig,
        waypoint_df["x"],
        waypoint_df["y"],
        plot_start_end_points=True,
        start_color="green",
        end_color="red",
        start_end_size=14,
        scatter_kwargs=scatter_kwargs,
    )

    # ハイライトする点をプロット
    if highlight_index is not None and 0 <= highlight_index < len(waypoint_df):
        fig.add_trace(
            go.Scatter(
                x=[waypoint_df["x"].iloc[highlight_index]],
                y=[waypoint_df["y"].iloc[highlight_index]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="orange",
                    symbol="circle",
                ),
                name="Selected Sample",
            )
        )
    
    # プロット設定
    fig.update_layout(
        title=f"{scene['name']} ({len(waypoint_df)} samples)",
        xaxis_title="Global X [m]",
        yaxis_title="Global Y [m]",
        height=600,
        hovermode="closest",
        showlegend=True,
    )
    # X,Yスケールを同一にする
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    # プロット
    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ハイライト選択されている場合、そのサンプルの情報を表示
    if highlight_index is not None and 0 <= highlight_index < len(waypoint_df):
        st.markdown(
            f"**Time:** {get_sample_utctime(waypoint_df, highlight_index, add_date=True)}  \n"
            f"**Sample token:** {waypoint_df.iloc[highlight_index]['sample_token']}"
        )
