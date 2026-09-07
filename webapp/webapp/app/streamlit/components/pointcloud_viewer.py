"""点群の 3D 表示（Plotly）.

複数の点群を 1 つの figure に重ねる:
  生 LiDAR（地面あり／なし）／生の深度点群／インスタンスごとの点群

Plotly の 3D 散布図はブラウザへ JSON で送られるため、点数がそのまま
転送量になる（10 万点で約 1.6MB）。深度画像 1 枚で 144 万点あるので、
描画前に必ず間引く。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common.point_ops import downsample_to_max
from app.streamlit.components.det2d_viewer import color_for as color_for_label
from app.streamlit.components.instance_tracking_viewer import (
    COLOR_MODE_TRACK,
    color_for_track,
)

# 背景として描く点群の色
COLOR_RAW_LIDAR = "#b0b0b0"    # 薄い灰色
COLOR_GROUND = "#8b6b4a"       # 茶色
COLOR_RAW_DEPTH = "#9fd8e8"    # 薄い水色

# 点の大きさ
MARKER_SIZE_BACKGROUND = 1
MARKER_SIZE_INSTANCE = 2


def _add_points(
    fig: go.Figure,
    points: np.ndarray,
    *,
    name: str,
    color: str,
    size: int,
    max_points: int,
    opacity: float = 1.0,
) -> int:
    """点群を 1 トレースとして追加する（上限まで間引く）.

    Returns:
        実際に描いた点数。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return 0
    reduced = downsample_to_max(points, max_points)
    fig.add_trace(go.Scatter3d(
        x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
        mode="markers", name=name,
        marker=dict(size=size, color=color, opacity=opacity),
        hoverinfo="name",
    ))
    return reduced.shape[0]


def build_pointcloud_figure(
    *,
    raw_lidar: np.ndarray | None = None,
    ground: np.ndarray | None = None,
    raw_depth: np.ndarray | None = None,
    instance_groups: Sequence[dict[str, Any]] = (),
    max_points_per_trace: int = 50_000,
    height: int = 720,
) -> tuple[go.Figure, dict[str, int]]:
    """点群を重ねた figure を組み立てる.

    Args:
        instance_groups: ``{"key": 表示名, "color": 色, "points": (N,3)}`` のリスト

    Returns:
        (figure, {トレース名: 描画点数})。点数は UI に出して、
        間引きが効いていることを確認できるようにする。
    """
    fig = go.Figure()
    counts: dict[str, int] = {}

    # 背景から先に描く。後から描いたものが手前に見えるため、
    # 注目したいインスタンス点群を最後に置く
    if raw_lidar is not None:
        counts["Raw LiDAR"] = _add_points(
            fig, raw_lidar, name="Raw LiDAR", color=COLOR_RAW_LIDAR,
            size=MARKER_SIZE_BACKGROUND, max_points=max_points_per_trace,
            opacity=0.5,
        )
    if ground is not None:
        counts["Ground"] = _add_points(
            fig, ground, name="Ground", color=COLOR_GROUND,
            size=MARKER_SIZE_BACKGROUND, max_points=max_points_per_trace,
            opacity=0.5,
        )
    if raw_depth is not None:
        counts["Raw Depth"] = _add_points(
            fig, raw_depth, name="Raw Depth", color=COLOR_RAW_DEPTH,
            size=MARKER_SIZE_BACKGROUND, max_points=max_points_per_trace,
            opacity=0.5,
        )

    for group in instance_groups:
        drawn = _add_points(
            fig, group["points"], name=group["key"], color=group["color"],
            size=MARKER_SIZE_INSTANCE, max_points=max_points_per_trace,
        )
        counts[group["key"]] = counts.get(group["key"], 0) + drawn

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis_title="X [m] (前方)",
            yaxis_title="Y [m] (左方)",
            zaxis_title="Z [m] (上方)",
            # 実寸比を保つ。自動だと z 方向が極端に伸びて形が読めない
            aspectmode="data",
        ),
    )
    return fig, counts


def group_instance_points(
    fittings: Iterable[dict[str, Any]],
    *,
    color_mode: str,
    points_key: str,
    enabled_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Box Fitting の結果を、色分けの単位ごとの点群にまとめる.

    トレース数が増えると Plotly が重くなるので、
    同じ色になるものは 1 トレースへ結合する。
    """
    from common.point_ops import points_from_json

    buckets: dict[str, list[np.ndarray]] = {}
    for fit in fittings:
        key = (
            str(fit.get("track_id")) if color_mode == COLOR_MODE_TRACK
            else str(fit.get("label"))
        )
        if enabled_keys is not None and key not in enabled_keys:
            continue
        points = points_from_json(fit.get(points_key))
        if points.shape[0] == 0:
            continue
        buckets.setdefault(key, []).append(points)

    color_fn = (
        color_for_track if color_mode == COLOR_MODE_TRACK else color_for_label
    )
    return [
        {"key": key, "color": color_fn(key), "points": np.vstack(chunks)}
        for key, chunks in buckets.items()
    ]


def render_pointcloud(
    fig: go.Figure, counts: dict[str, int], *, show_counts: bool = True
) -> None:
    st.plotly_chart(fig, width="stretch")
    if show_counts and counts:
        st.caption(
            "描画点数: "
            + " / ".join(f"{name} {n:,}" for name, n in counts.items() if n)
        )
