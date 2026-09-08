"""Depth Estimation & Box Fitting のリクエスト／レスポンススキーマ.

処理は 3 段階で、粒度がそれぞれ違う:

  1. 深度推定    … カメラ × sample（カメラを外側にループする。
                    将来 Pose-Conditioned な複数フレーム推定へ広げるため）
  2. LiDAR 統合  … sample（sweep をキーフレーム座標へ揃えて合体）
  3. Box Fitting … インスタンス

LiDAR の読み込みと地面除去は use_lidar に関係なく常に実施する。
UI が比較用に生 LiDAR と地面を表示できるようにするためで、
use_lidar は「フィッティングに使うか」だけを制御する。

深度マップと LiDAR 点群は DERIVED_ROOT へ直接書き、パスだけを返す。
書き込み先は webapp が先に作成した run の id（params_id）で決まる。
ジョブ id ではなく run の id にするのは、後から
「この run のファイル」をディレクトリごと消せるようにするため。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SensorFrame(BaseModel):
    """処理対象のセンサーフレーム（カメラ・LiDAR 共通）."""
    sample_data_token: str
    sample_token: str
    filename: str
    channel: str
    sample_idx: int = 0
    timestamp: int = 0
    # LiDAR の sweep 統合で、キーフレーム（座標系の基準）を見分けるのに使う
    is_key_frame: bool = True
    width: int | None = None
    height: int | None = None
    # 座標変換に使う。webapp が DB から解決したものをそのまま渡す
    ego_pose: dict[str, Any] = Field(default_factory=dict)
    calibrated_sensor: dict[str, Any] = Field(default_factory=dict)


class InstanceRef(BaseModel):
    """マスクの供給元となるトラッキングのインスタンス."""
    instance_tracking_2d_id: str
    sample_data_token: str
    track_id: str
    label: str
    mask_rle: dict[str, Any]  # COCO 非圧縮 RLE


class BoxFittingRequest(BaseModel):
    dataroot: str
    # 書き込み先の run（webapp が先に作成済み）
    params_id: str
    # DERIVED_ROOT からの相対パス。既に params_id を含む
    output_dir: str

    camera_frames: list[SensorFrame] = Field(min_length=1)
    # sample ごとのキーフレーム（座標系の基準）
    lidar_frames: list[SensorFrame] = Field(default_factory=list)
    # 統合対象の LiDAR フレーム（sweep 含む）。
    # webapp 側が num_lidar_sweeps に応じて選び、時刻順で渡す
    lidar_sweeps: list[SensorFrame] = Field(default_factory=list)
    instances: list[InstanceRef] = Field(default_factory=list)

    # LiDAR をフィッティングに使うか（読み込みと地面除去は常に行う）
    use_lidar: bool = False
    num_lidar_sweeps: int = 1

    # UI のタブに対応した設定群
    mask_params: dict[str, Any] = Field(default_factory=dict)
    depth_params: dict[str, Any] = Field(default_factory=dict)
    lidar_params: dict[str, Any] = Field(default_factory=dict)
    box_fitting_params: dict[str, Any] = Field(default_factory=dict)

    # DB へ保存する点群の上限（間引き後）
    stored_points_max: int = 500
    # 深度マップの保存倍率。1/2 で容量 1/4
    depth_downscale: float = 0.5

    stub_delay_sec: float | None = None


class DepthEstimationResult(BaseModel):
    """深度推定の結果（1 カメラフレーム）."""
    sample_data_token: str
    depth_path: str
    depth_width: int
    depth_height: int
    scale: float | None = None
    shift: float | None = None
    min_depth: float | None = None
    max_depth: float | None = None
    num_points: int = 0
    error: str | None = None


class LidarPointcloudResult(BaseModel):
    """LiDAR 統合点群の結果（1 sample）."""
    sample_token: str
    sample_data_token: str
    pointcloud_path: str
    coordinate_frame: str = "ego"
    num_sweeps: int = 1
    num_points: int = 0
    num_ground_points: int = 0
    error: str | None = None


class BoxFittingResult(BaseModel):
    """Box Fitting の結果（1 インスタンス）.

    ボックスを作れなかった場合も status 付きで返す。
    「点が少なすぎた」のか「マスク内に点が無い」のかで、
    閾値を下げれば拾えるかどうかの判断が変わる。
    """
    instance_tracking_2d_id: str
    sample_data_token: str
    track_id: str
    label: str
    status: str
    mask_rle_closed: dict[str, Any] | None = None
    # {"points": [[x, y, z], ...]}（ego 座標、間引き済み）
    points_depth_ego: dict[str, Any] | None = None
    points_lidar_ego: dict[str, Any] | None = None
    # 間引き前の実点数。信頼度の判断にはこちらを使う
    num_points_depth: int = 0
    num_points_lidar: int = 0
    depth_align_scale: float | None = None
    depth_align_shift: float | None = None
    center_ego: list[float] | None = None
    size_wlh: list[float] | None = None
    yaw_ego: float | None = None
    fitting_score: float | None = None
    error: str | None = None


class RefilterInstance(BaseModel):
    """再フィルタ対象のインスタンス."""
    id: str
    track_id: str | None = None
    label: str | None = None
    # クロージング後のマスク（DB に保存済みのものをそのまま渡す）
    mask_rle_closed: dict[str, Any]


class RefilterFrame(BaseModel):
    """再フィルタ対象のフレーム."""
    # DERIVED_ROOT からの相対パス（DepthEstimation.depth_path）
    depth_path: str
    calibrated_sensor: dict[str, Any]
    instances: list[RefilterInstance] = Field(default_factory=list)
    # .npz に内部パラメータが無い run 向けのフォールバック
    width: int | None = None
    height: int | None = None


class RefilterRequest(BaseModel):
    """保存済みの深度マップから点群を作り直す.

    パラメータ調整のたびにパイプライン全体を回さずに済ませるためのもの。
    点群そのものは送らず、**保存済みファイルのパスとマスクだけ**を渡す
    （/derived は webapp と共有マウントされている）。
    """
    frames: list[RefilterFrame] = Field(min_length=1)
    depth_params: dict[str, Any] = Field(default_factory=dict)
    # LiDAR 側は未実装（use_lidar の混合を入れるときに使う）
    lidar_params: dict[str, Any] = Field(default_factory=dict)
    stored_points_max: int = 500
    max_depth: float | None = None


class RefilterInstanceResult(BaseModel):
    id: str
    track_id: str | None = None
    label: str | None = None
    # 間引き前の点数（フィルタ前 / フィルタ後）
    num_points_raw: int = 0
    num_points_kept: int = 0
    # 間引き後の座標。前後で同じボクセルサイズを使う
    points_raw_ego: dict[str, Any] | None = None
    points_filtered_ego: dict[str, Any] | None = None


class RefilterResponse(BaseModel):
    instances: list[RefilterInstanceResult]
    elapsed_sec: float


class BoxFittingPartial(BaseModel):
    """ポーリングで逐次返す部分結果.

    段階ごとに形が違うので、種別を付けて中身は dict で渡す。
    UI は kind を見て振り分ける。
    """
    kind: str  # 'depth' | 'lidar' | 'box'
    data: dict[str, Any]


class BoxFittingJobResult(BaseModel):
    num_depth_frames: int
    num_lidar_frames: int
    num_box_fittings: int
    num_fitted: int
    inference_time: float
    depth_estimations: list[DepthEstimationResult]
    lidar_pointclouds: list[LidarPointcloudResult]
    box_fittings: list[BoxFittingResult]


class JobResponse(BaseModel):
    """ジョブの登録・状態取得（他ステップと同じ形に揃える）."""
    job_id: str
    kind: str
    status: str
    total: int = 0
    processed: int = 0
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    elapsed_sec: float = 0.0
    result: dict | None = None
    partial_count: int = 0
    partial: list[BoxFittingPartial] = Field(default_factory=list)
