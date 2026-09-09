"""BEV（上から見た平面図）での 3D ボックス表示.

Box Fitting の結果を、インスタンス点群に重ねて上から確認する。
PointCloud タブの 3D 表示と違い、向きと寸法の当てはまりは
真上から見たほうが判断しやすい。

座標は ego（x=前方 / y=左方）。**画面右を前方(x)、画面上を左方(y)** にして、
点群ビューの Top View と同じ見え方に揃える。
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common.point_ops import downsample_to_max

# 自車位置のマーカー
EGO_MARKER_COLOR = "#444444"
# GT ボックスの色（推定ボックスとの区別を色でつける）
GT_BOX_COLOR = "#2ca02c"


def box_corners_bev(
    center_xy: Sequence[float], width: float, length: float, yaw: float
) -> np.ndarray:
    """BEV のボックス 4 隅を返す（閉じた 5 点）.

    Args:
        width: 車体の横幅（y 方向）、length: 前後長（x 方向）、yaw: [rad]

    nuScenes の size は [width, length, height] の順なので、
    取り違えると縦横が入れ替わったボックスになる。
    """
    half_l, half_w = length / 2.0, width / 2.0
    local = np.array([
        [half_l, half_w], [half_l, -half_w],
        [-half_l, -half_w], [-half_l, half_w], [half_l, half_w],
    ])
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return local @ rotation.T + np.asarray(center_xy, dtype=float)[:2]


def _add_polygon(
    fig: go.Figure, points: np.ndarray, *, name: str, color: str,
    width: int = 2, dash: str | None = None, showlegend: bool = False,
) -> None:
    fig.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1], mode="lines", name=name,
        line=dict(color=color, width=width, dash=dash),
        showlegend=showlegend, hoverinfo="name",
    ))


def build_bev_figure(
    *,
    instance_groups: Sequence[dict[str, Any]] = (),
    boxes: Sequence[dict[str, Any]] = (),
    hulls: Sequence[dict[str, Any]] = (),
    title: str = "",
    max_points_per_trace: int = 20_000,
    height: int = 640,
    axis_range: tuple[float, float, float, float] | None = None,
) -> go.Figure:
    """BEV の figure を組み立てる.

    Args:
        instance_groups: ``{"key", "color", "points": (N,3)}``（ego 座標）
        boxes: ``{"key", "color", "center_xy", "width", "length", "yaw"}``
        hulls: ``{"key", "color", "points": (M,2)}``
        axis_range: ``(xmin, xmax, ymin, ymax)``。左右に並べるとき、
            両方の figure で同じ範囲にすると位置を比較できる
    """
    fig = go.Figure()

    for group in instance_groups:
        points = np.asarray(group["points"], dtype=float)
        if points.shape[0] == 0:
            continue
        reduced = downsample_to_max(points, max_points_per_trace)
        fig.add_trace(go.Scattergl(
            x=reduced[:, 0], y=reduced[:, 1], mode="markers",
            name=group["key"],
            marker=dict(size=3, color=group["color"], opacity=0.6),
            hoverinfo="name",
        ))

    for hull in hulls:
        points = np.asarray(hull["points"], dtype=float)
        if points.shape[0] < 3:
            continue
        # 閉じた多角形にする
        closed = np.vstack([points, points[:1]])
        _add_polygon(fig, closed, name=f"hull {hull['key']}",
                     color=hull["color"], width=1, dash="dot")

    for box in boxes:
        corners = box_corners_bev(
            box["center_xy"], box["width"], box["length"], box["yaw"]
        )
        _add_polygon(fig, corners, name=box["key"], color=box["color"], width=2)
        # 前方向が分かるよう、前面の中点へ短い線を引く
        front = (corners[0] + corners[1]) / 2.0
        center = np.asarray(box["center_xy"], dtype=float)[:2]
        _add_polygon(fig, np.vstack([center, front]),
                     name=box["key"], color=box["color"], width=1)

    # 自車位置
    fig.add_trace(go.Scatter(
        x=[0.0], y=[0.0], mode="markers", name="ego",
        marker=dict(size=9, color=EGO_MARKER_COLOR, symbol="x"),
        showlegend=False, hoverinfo="name",
    ))

    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=0, r=0, t=30 if title else 10, b=0),
        showlegend=False,
        xaxis_title="X [m] (前方)",
        yaxis_title="Y [m] (左方)",
    )
    # 実寸比を保つ。崩れると当てはまりの善し悪しが読めない
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if axis_range is not None:
        xmin, xmax, ymin, ymax = axis_range
        fig.update_xaxes(range=[xmin, xmax])
        fig.update_yaxes(range=[ymin, ymax])
    return fig


def combined_axis_range(
    point_sets: Iterable[np.ndarray], *, margin: float = 3.0
) -> tuple[float, float, float, float] | None:
    """複数の点群をまとめて収める表示範囲を求める.

    GT と推定を左右に並べるとき、範囲を揃えないと
    同じ位置の物体が別の場所にあるように見える。
    """
    xs, ys = [], []
    for points in point_sets:
        points = np.asarray(points, dtype=float)
        if points.ndim == 2 and points.shape[0]:
            xs.append(points[:, 0])
            ys.append(points[:, 1])
    if not xs:
        return None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return (
        float(x.min() - margin), float(x.max() + margin),
        float(y.min() - margin), float(y.max() + margin),
    )


def render_bev(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch")
