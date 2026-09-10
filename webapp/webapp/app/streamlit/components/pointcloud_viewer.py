"""点群の 3D 表示（Plotly）.

複数の点群を 1 つの figure に重ねる:
  生 LiDAR（地面あり／なし）／生の深度点群／インスタンスごとの点群

Plotly の 3D 散布図はブラウザへ JSON で送られるため、点数がそのまま
転送量になる（10 万点で約 1.6MB）。深度画像 1 枚で 144 万点あるので、
描画前に必ず間引く。
"""
from __future__ import annotations

import math
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

# 自車の姿勢を示す軸の色（x=前方 / y=左方 / z=上方）
AXIS_COLORS = ("red", "green", "blue")
AXIS_NAMES = ("X", "Y", "Z")
AXIS_LINE_WIDTH = 4
# 矢先の円錐の大きさ [m]
AXIS_TIP_SIZE = 0.8

# 3D ボックスの枠線の太さ
BOX_LINE_WIDTH = 3
# 直方体の 12 辺（頂点インデックスの組）。
# 頂点は下面 0-3 → 上面 4-7 の順で作る
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),      # 下面
    (4, 5), (5, 6), (6, 7), (7, 4),      # 上面
    (0, 4), (1, 5), (2, 6), (3, 7),      # 垂直
)

# 視点プリセット。点群は ego 座標（x=前方 / y=左方 / z=上方）。
#
#   Top View     … 真上から見下ろす。画面右が前方(x)、画面上が左方(y)
#   Forward View … 真後ろから前方を見る。画面左が左方(y)、画面上が上方(z)
#   Global View  … global 座標に対して向きが固定された視点
#
# eye はシーンを正規化した座標での視点位置。up で画面の上方向を決めると、
# 視線方向 × up の外積が画面右になり、残りの軸の向きが定まる。
VIEW_TOP = "Top View"
VIEW_FORWARD = "Forward View"
VIEW_GLOBAL = "Global View"
VIEWS = (VIEW_GLOBAL, VIEW_TOP, VIEW_FORWARD)

CAMERA_PRESETS = {
    # +z から見下ろす。up=+y にすると画面上が左方(y)、画面右が前方(x)
    VIEW_TOP: dict(
        eye=dict(x=0.0, y=0.0, z=2.5),
        up=dict(x=0.0, y=1.0, z=0.0),
        center=dict(x=0.0, y=0.0, z=0.0),
    ),
    # -x（車両の後方）から前方を見る。up=+z で画面上が上方(z)、
    # 視線(+x) × up(+z) = -y が画面右 → +y（左方）が画面左に来る
    VIEW_FORWARD: dict(
        eye=dict(x=-2.5, y=0.0, z=0.0),
        up=dict(x=0.0, y=0.0, z=1.0),
        center=dict(x=0.0, y=0.0, z=0.0),
    ),
}


def global_camera(
    ego_quaternion, default_eye, default_up
) -> dict[str, dict[str, float]]:
    """global 座標に対して向きが固定された視点を作る.

    Args:
        ego_quaternion: ego → global の回転（EgoPose.rotation）

    NOTE: **回転行列の転置を掛けること。**
    点群は ego 座標なので、ego 上のベクトル v は global では ``R @ v`` に見える。
    global で一定の方向 v_global を ego 上で表すには ``R.T @ v_global`` が要る。
    ``R @ v_global`` にすると、視点が車両の回転の 2 倍で回ってしまう
    （点群が global 座標なら ``R`` のままでよいが、ここでは ego 座標）。
    """
    from app.services.geometry.transform import (
        normalize_quaternion,
        quaternion_to_rotation_matrix,
    )

    rotation = quaternion_to_rotation_matrix(normalize_quaternion(ego_quaternion))
    eye = rotation.T @ np.asarray(default_eye, dtype=np.float64)
    up = rotation.T @ np.asarray(default_up, dtype=np.float64)
    return dict(
        eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
        up=dict(x=float(up[0]), y=float(up[1]), z=float(up[2])),
        center=dict(x=0.0, y=0.0, z=0.0),
    )

# 視点プリセット。ego 座標は x=前方 / y=左方 / z=上方。
#
#   Top View     … 真上から見下ろす。画面右が前方(x)、画面上が左方(y)
#   Forward View … 真後ろから前方を見る。画面左が左方(y)、画面上が上方(z)
#
# eye はシーンを正規化した座標での視点位置。
# up で画面の上方向を指定すると、残りの軸の向きは自動的に決まる
# （視線方向 × up の外積が画面右になる）。
VIEW_TOP = "Top View"
VIEW_FORWARD = "Forward View"
CAMERA_PRESETS = {
    # +z から見下ろす。up=+y にすると画面上が左方、画面右が前方になる
    VIEW_TOP: dict(
        eye=dict(x=0.0, y=0.0, z=2.5),
        up=dict(x=0.0, y=1.0, z=0.0),
        center=dict(x=0.0, y=0.0, z=0.0),
    ),
    # -x（車両の後方）から前方を見る。up=+z で画面上が上方、
    # 視線(+x) × up(+z) = -y が画面右 → +y（左方）が画面左に来る
    VIEW_FORWARD: dict(
        eye=dict(x=-2.5, y=0.0, z=0.0),
        up=dict(x=0.0, y=0.0, z=1.0),
        center=dict(x=0.0, y=0.0, z=0.0),
    ),
}


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


def box_corners_3d(
    center: Sequence[float], size_wlh: Sequence[float], yaw: float
) -> np.ndarray:
    """3D ボックスの 8 頂点を返す ``(8, 3)``.

    Args:
        size_wlh: nuScenes の並び ``[width, length, height]``。
            取り違えると縦横が入れ替わった箱になる
    """
    width, length, height = (float(v) for v in size_wlh)
    half_l, half_w, half_h = length / 2.0, width / 2.0, height / 2.0
    local = np.array([
        [half_l, half_w, -half_h], [half_l, -half_w, -half_h],
        [-half_l, -half_w, -half_h], [-half_l, half_w, -half_h],
        [half_l, half_w, half_h], [half_l, -half_w, half_h],
        [-half_l, -half_w, half_h], [-half_l, half_w, half_h],
    ])
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rotation = np.array([
        [cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0],
    ])
    return local @ rotation.T + np.asarray(center, dtype=float)


def add_boxes(
    fig: go.Figure, boxes: Sequence[dict[str, Any]]
) -> int:
    """3D ボックスを枠線で追加する.

    Args:
        boxes: ``{"key", "color", "center", "size_wlh", "yaw"}`` のリスト

    Returns:
        描いたボックス数。

    同じ色のボックスは 1 トレースにまとめ、辺の間を None で区切る。
    ボックスごとにトレースを作ると、数十個で Plotly が目に見えて重くなる。
    """
    by_color: dict[str, list[np.ndarray]] = {}
    for box in boxes:
        corners = box_corners_3d(box["center"], box["size_wlh"], box.get("yaw") or 0.0)
        by_color.setdefault(box["color"], []).append(corners)

    for color, corner_sets in by_color.items():
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for corners in corner_sets:
            for start, end in BOX_EDGES:
                xs += [corners[start, 0], corners[end, 0], None]
                ys += [corners[start, 1], corners[end, 1], None]
                zs += [corners[start, 2], corners[end, 2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines", name="fitted box",
            line=dict(color=color, width=BOX_LINE_WIDTH),
            showlegend=False, hoverinfo="skip",
        ))
    return len(boxes)


def add_ego_axes(
    fig: go.Figure,
    *,
    length: float = 5.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
) -> None:
    """自車の姿勢を示す軸を追加する.

    Args:
        length: 軸の長さ [m]
        rotation: 3x3 の回転行列。**点群は ego 座標なので既定は恒等**
            （global 座標で描くときは ego→global の回転を渡す）

    向きが分からないと、点群だけ見ても前後左右が判断できない。
    x=赤 / y=緑 / z=青 は 3D ツールの慣習に合わせている。
    """
    axis_origin = np.asarray(origin, dtype=float)
    axis_rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
    # 列ベクトルが各軸の方向
    axis_vectors = length * axis_rotation

    for index, (color, name) in enumerate(zip(AXIS_COLORS, AXIS_NAMES)):
        direction = axis_vectors[:, index]
        tip = axis_origin + direction
        fig.add_trace(go.Scatter3d(
            x=[axis_origin[0], tip[0]],
            y=[axis_origin[1], tip[1]],
            z=[axis_origin[2], tip[2]],
            mode="lines", name=name,
            line=dict(color=color, width=AXIS_LINE_WIDTH),
            showlegend=False, hoverinfo="name",
        ))
        # 線だけだと向き（どちらが先か）が分からないので矢先を付ける
        unit = direction / max(float(np.linalg.norm(direction)), 1e-9)
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[unit[0]], v=[unit[1]], w=[unit[2]],
            colorscale=[[0, color], [1, color]], showscale=False,
            sizemode="absolute", sizeref=AXIS_TIP_SIZE,
            anchor="tip", name=name, hoverinfo="name",
        ))


def build_pointcloud_figure(
    *,
    raw_lidar: np.ndarray | None = None,
    ground: np.ndarray | None = None,
    raw_depth: np.ndarray | None = None,
    instance_groups: Sequence[dict[str, Any]] = (),
    boxes: Sequence[dict[str, Any]] = (),
    max_points_per_trace: int = 50_000,
    height: int = 720,
    camera: dict[str, Any] | None = None,
    view_revision: str = "",
    axis_length: float | None = 5.0,
) -> tuple[go.Figure, dict[str, int]]:
    """点群を重ねた figure を組み立てる.

    Args:
        instance_groups: ``{"key": 表示名, "color": 色, "points": (N,3)}`` のリスト
        boxes: ``{"key", "color", "center", "size_wlh", "yaw"}`` のリスト
        camera: Plotly の scene.camera 設定。None なら Plotly 既定
        view_revision: uirevision に渡す値。**同じ値の間はユーザーの回転操作が
            保持され、値が変わったときだけ視点がプリセットへ戻る**。
            これを固定にすると、ボタンを押しても視点が変わらなくなる

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

    if boxes:
        counts["Fitted boxes"] = add_boxes(fig, boxes)

    # 軸は最後に足す。点群より手前に描いて隠れないようにする
    if axis_length:
        add_ego_axes(fig, length=axis_length)

    scene = dict(
        xaxis_title="X [m] (前方)",
        yaxis_title="Y [m] (左方)",
        zaxis_title="Z [m] (上方)",
        # 実寸比を保つ。自動だと z 方向が極端に伸びて形が読めない
        aspectmode="data",
    )
    if camera is not None:
        scene["camera"] = camera

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(itemsizing="constant"),
        scene=scene,
        # 再描画のたびに視点が初期化されると、点群を回して見る作業が続かない。
        # uirevision が同じ間は Plotly が視点を保持する
        uirevision=view_revision or "pointcloud",
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
