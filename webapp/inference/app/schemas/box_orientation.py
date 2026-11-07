"""Box Orientation Estimation（AI モデルによるボックス向きの再推定）のスキーマ.

保存済みの Box Fitting run に対して、global_track_id × sample ごとに
カメラ画像から物体の向きを推定する。**ボックスには反映しない**（結果を返すだけ）。

webapp は run の Box Fitting 行（カメラごと）を候補として渡し、
推論サーバーがグループ（global_track_id × sample）ごとにカメラを選ぶ。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OrientationCandidate(BaseModel):
    """向き推定の候補（Box Fitting の 1 行 = 1 カメラのインスタンス）."""
    box_fitting_id: str
    global_track_id: str
    sample_token: str
    sample_data_token: str
    # グループのラベル（カメラ間結合の多数決後）
    label: str
    channel: str
    # データセットルートからの画像パス
    filename: str
    # トラッキングの外接矩形 [xmin, ymin, xmax, ymax]（画素、xmax/ymax は排他）
    bbox: list[float]
    # トラッキングのマスク（COCO 非圧縮 RLE）。背景を落とす方式のラベルだけ入る
    mask_rle: dict[str, Any] | None = None
    # 当てはめに使った点群の実点数（深度 + LiDAR。深度を使わなかった場合も数える）
    num_points: int = 0
    # グループの代表行（ボックスを持つ行）か
    is_primary: bool = False
    # 代表行のボックス中心（基準 ego 座標）。代表行以外は None
    center_ego: list[float] | None = None
    # カメラの位置をボックスと同じ（基準 ego）座標系へ移すのに使う
    calibrated_sensor: dict[str, Any] = Field(default_factory=dict)
    ego_pose: dict[str, Any] = Field(default_factory=dict)
    reference_ego_pose: dict[str, Any] | None = None


class OrientationRequest(BaseModel):
    dataroot: str
    # ラベルごとの推定方式（webapp の BOX_ORIENTATION_ESTIMATIONS）。None は推定しない
    methods: dict[str, str | None] = Field(default_factory=dict)
    candidates: list[OrientationCandidate] = Field(default_factory=list)
    stub_delay_sec: float | None = None


class OrientationResult(BaseModel):
    """向き推定の結果（1 グループ = global_track_id × sample）."""
    # ボックスを持つ代表行（結果の保存先）
    box_fitting_id: str
    global_track_id: str
    sample_token: str
    label: str
    method: str
    # estimated / no_box / unsupported / failed
    status: str
    # 推定に使ったカメラの行
    source_box_fitting_id: str | None = None
    channel: str | None = None
    bbox: list[float] | None = None
    # モデルの出力角度 [deg]（モデル固有の定義。PartHOE はカメラから見た向き）
    angle_deg: float | None = None
    # 推定の確からしさ。PartHOE は argmax のビンの確率、
    # Orient-Anything は向きを持つ物体である確率（公式の confidence）
    confidence: float | None = None
    # 推定した向き（基準 ego 座標の yaw [rad]）
    yaw_ego: float | None = None
    error: str | None = None
