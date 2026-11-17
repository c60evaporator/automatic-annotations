"""Track Smoother タブの BEV 表示（トラックの平滑化と、動きに応じた補正の結果）.

純粋な描画関数だけを置く（data_access に依存しない）。座標は呼び出し側で
表示する sample の基準 ego 座標に揃えてから渡す。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

from app.streamlit.components.bev_viewer import EGO_MARKER_COLOR

KIND_STATIC = "static"
KIND_MOVING = "moving"
KIND_SHORT = "short"
KIND_TEXT = {KIND_STATIC: "静止", KIND_MOVING: "移動", KIND_SHORT: "短い"}

SELECTED_OPACITY = 1.0
OTHER_OPACITY = 0.3
LINE_OPACITY = 0.45
LEGEND_COLOR = "#888888"


def box_polygon(center_xy, length: float, width: float, yaw: float) -> np.ndarray:
    """BEV の箱の 4 隅（閉じた多角形、5 点）."""
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-math.sin(yaw), math.cos(yaw)])
    c = np.asarray(center_xy, dtype=float)
    hl, hw = length / 2.0, width / 2.0
    pts = [c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw,
           c - u * hl + v * hw, c - u * hl - v * hw]
    return np.stack(pts)


def _joined(polylines: list[np.ndarray]) -> tuple[list, list]:
    """複数の線を None 区切りで 1 本の trace にまとめる."""
    xs: list = []
    ys: list = []
    for line in polylines:
        xs.extend(line[:, 0].tolist() + [None])
        ys.extend(line[:, 1].tolist() + [None])
    return xs, ys


def _circle(center, radius: float, n: int = 48) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)


def build_track_smoother_figure(
    observations: list[dict[str, Any]],
    static_groups: list[dict[str, Any]],
    *,
    height: int = 720,
) -> go.Figure:
    """トラックの観測点・平滑化した軌跡・静止の中央値・ボックスを BEV で描く.

    Args:
        observations: フレームごとの観測（表示する sample の基準 ego 座標）。
            ``{"global_id", "box_merged_id", "kind", "order", "color", "selected",
            "observed_xy", "final_xy", "smoothed_xy" | None, "outlier",
            "position_corrected", "yaw_corrected", "hover",
            "box" | None, "before_box" | None}``。box は ``(中心 XY, 長さ, 幅, yaw)``
        static_groups: ``{"color", "median_xy", "radius"}``（静止トラックの中央値と判定半径）
    """
    fig = go.Figure()
    by_track: dict[str, list[dict[str, Any]]] = {}
    for o in observations:
        by_track.setdefault(o["global_id"], []).append(o)

    for global_id, obs in by_track.items():
        obs = sorted(obs, key=lambda o: o["order"])
        color = obs[0]["color"]

        # 平滑化した軌跡（移動トラックだけ）。結合した box_merged_track の間は破線
        moving = [o for o in obs if o["kind"] == KIND_MOVING and o["smoothed_xy"] is not None]
        solid, dashed = [], []
        for a, b in zip(moving, moving[1:]):
            segment = np.array([a["smoothed_xy"], b["smoothed_xy"]], dtype=float)
            (solid if a["box_merged_id"] == b["box_merged_id"] else dashed).append(segment)
        for segments, dash in ((solid, "solid"), (dashed, "dash")):
            if segments:
                xs, ys = _joined(segments)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", opacity=LINE_OPACITY, hoverinfo="skip",
                    line=dict(color=color, width=2, dash=dash), showlegend=False,
                ))

        # ボックス（選んだ sample は濃く、それ以外は薄く）。角度を直したフレームは修正前を点線で
        for selected, opacity in ((False, OTHER_OPACITY), (True, SELECTED_OPACITY)):
            boxes = [box_polygon(*o["box"]) for o in obs
                     if o.get("box") is not None and o["selected"] == selected]
            befores = [box_polygon(*o["before_box"]) for o in obs
                       if o.get("before_box") is not None and o["selected"] == selected]
            for polys, dash in ((boxes, "solid"), (befores, "dot")):
                if polys:
                    xs, ys = _joined(polys)
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", opacity=opacity, hoverinfo="skip",
                        line=dict(color=color, width=2 if selected else 1, dash=dash),
                        showlegend=False,
                    ))

        # 位置を直したフレーム: 修正前（×）→ 修正後（丸）の矢印
        corrected = [o for o in obs if o["position_corrected"]]
        if corrected:
            xs, ys, sizes = [], [], []
            for o in corrected:
                xs += [o["observed_xy"][0], o["final_xy"][0], None]
                ys += [o["observed_xy"][1], o["final_xy"][1], None]
                sizes += [0, 9, 0]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", opacity=LINE_OPACITY, hoverinfo="skip",
                line=dict(color=color, width=1),
                marker=dict(symbol="arrow", angleref="previous", size=sizes, color=color),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[o["final_xy"][0] for o in corrected], y=[o["final_xy"][1] for o in corrected],
                mode="markers", hoverinfo="skip", showlegend=False,
                marker=dict(symbol="circle", color=color, size=6,
                            opacity=[SELECTED_OPACITY if o["selected"] else OTHER_OPACITY
                                     for o in corrected]),
            ))

        # 観測点（修正前の位置）
        symbols = []
        for o in obs:
            if o["kind"] == KIND_SHORT:
                symbols.append("circle-open")
            elif o["outlier"] or o["position_corrected"]:
                symbols.append("x")
            else:
                symbols.append("circle")
        fig.add_trace(go.Scatter(
            x=[o["observed_xy"][0] for o in obs], y=[o["observed_xy"][1] for o in obs],
            mode="markers", name=str(global_id), showlegend=False,
            text=[o["hover"] for o in obs], hovertemplate="%{text}<extra></extra>",
            marker=dict(
                symbol=symbols, color=color,
                size=[12 if o["selected"] else 7 for o in obs],
                opacity=[SELECTED_OPACITY if o["selected"] else OTHER_OPACITY for o in obs],
                line=dict(color=["black" if o["selected"] else color for o in obs],
                          width=[2 if o["selected"] else 0 for o in obs]),
            ),
        ))

    # 静止トラックの中央値と判定半径
    for group in static_groups:
        circle = _circle(group["median_xy"], group["radius"])
        fig.add_trace(go.Scatter(
            x=circle[:, 0], y=circle[:, 1], mode="lines", opacity=LINE_OPACITY,
            hoverinfo="skip", showlegend=False,
            line=dict(color=group["color"], width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=[group["median_xy"][0]], y=[group["median_xy"][1]], mode="markers",
            hoverinfo="skip", showlegend=False,
            marker=dict(symbol="diamond", size=13, color=group["color"],
                        line=dict(color="black", width=1)),
        ))

    # 自車（原点）
    fig.add_trace(go.Scatter(
        x=[0.0], y=[0.0], mode="markers", name="ego", showlegend=False, hoverinfo="name",
        marker=dict(size=11, color=EGO_MARKER_COLOR, symbol="x"),
    ))

    # 凡例（色はトラックごとなので、形と線の意味だけを灰色で示す）
    legend = [
        ("● 非外れ値", dict(mode="markers", marker=dict(symbol="circle", color=LEGEND_COLOR))),
        ("× 外れ値 / 修正前", dict(mode="markers", marker=dict(symbol="x", color=LEGEND_COLOR))),
        ("○ 短いトラック", dict(mode="markers",
                              marker=dict(symbol="circle-open", color=LEGEND_COLOR))),
        ("◆ 静止中央値", dict(mode="markers", marker=dict(symbol="diamond", color=LEGEND_COLOR))),
        ("─ 平滑化軌跡", dict(mode="lines", line=dict(color=LEGEND_COLOR))),
        ("┄ 結合区間", dict(mode="lines", line=dict(color=LEGEND_COLOR, dash="dash"))),
    ]
    for name, style in legend:
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, showlegend=True, **style))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="X [m] (前方)", yaxis_title="Y [m] (左方)",
        dragmode="pan", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
