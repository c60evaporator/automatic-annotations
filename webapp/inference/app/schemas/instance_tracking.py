"""Instance Tracking (SAM2) のリクエスト／レスポンススキーマ.

設計:
  - **推論は「プロンプト区間」ごとに走る。**
    Detection2D の sample_interval ごとにプロンプト（検出ボックス）を与え、
    次のプロンプト sample までを伝播させる。
    例: interval=4 なら sample 0→4、4→8、8→12 …
  - **入力フレームには非キーフレーム（sweep）も含む。**
    伝播の精度を上げるために使う。結果自体は表示に使わないが、
    どのフレームを使ったかは呼び出し側が決めて渡す
    （推論サーバーは DB を持たないため）。
  - **区間の境界では track_id を引き継ぐ。**
    前の区間から伝播したインスタンスと、今回のプロンプトで得た
    インスタンスを IoU で貪欲マッチングする。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.services.tracking_match import (
    IOU_METHOD_BOX,
    LABEL_MATCH_LABEL,
)


class TrackingFrameRef(BaseModel):
    """トラッキングに使う 1 フレーム."""
    sample_data_token: str
    sample_token: str
    filename: str
    channel: str
    sample_idx: int
    # キーフレームかどうか。結果を保存・表示するのはキーフレームのみ
    is_key_frame: bool = True
    timestamp: int = 0
    width: int | None = None
    height: int | None = None


class PromptBox(BaseModel):
    """SAM2 に与えるプロンプト（Detection2D のボックス）."""
    sample_data_token: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str
    score: float | None = None
    # 由来の Detection2D 行。結果から辿れるようにしておく
    detection_2d_id: str | None = None


class InstanceTrackingRequest(BaseModel):
    dataroot: str
    # 全対象フレーム（sweep 含む）。sample_idx 昇順で渡す想定
    frames: list[TrackingFrameRef] = Field(min_length=1)
    # プロンプト。sample_interval ごとのキーフレームぶん
    prompts: list[PromptBox] = Field(default_factory=list)

    model_id: str | None = None

    # プロンプトを与える間隔（Detection2D の run から引き継ぐ）
    sample_interval: int = 1
    # 1 sample あたりのフレーム数（1 ならキーフレームのみ）
    num_sweeps: int = 1

    # 区間境界での track_id 引き継ぎ判定
    iou_threshold: float = 0.5
    iou_method: str = IOU_METHOD_BOX
    iou_label_match: str = LABEL_MATCH_LABEL
    # iou_label_match='category_group' のときに使う。
    # ラベル体系は webapp 側の設定なので、解決済みのものを受け取る
    label_to_category_group: dict[str, str] = Field(default_factory=dict)

    mask_score_threshold: float = 0.5

    # スタブ用: 1 推論あたりの待ち時間（秒）。本実装では無視される
    stub_delay_sec: float | None = None


class TrackedInstance(BaseModel):
    """1 フレーム内の 1 インスタンス."""
    track_id: str
    label: str
    score: float | None = None
    # COCO 非圧縮 RLE: {"size": [h, w], "counts": [...]}
    mask_rle: dict[str, Any]
    mask_area: int
    # マスクの外接矩形
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    # プロンプト由来のインスタンスなら、その Detection2D 行
    detection_2d_id: str | None = None
    # このフレームがプロンプトを与えた sample かどうか
    is_prompt_frame: bool = False


class TrackingFrameResult(BaseModel):
    sample_data_token: str
    sample_token: str
    sample_idx: int
    channel: str
    is_key_frame: bool = True
    instances: list[TrackedInstance] = Field(default_factory=list)
    inference_time: float | None = None
    error: str | None = None


class InstanceTrackingResult(BaseModel):
    num_frames: int
    num_instances: int
    num_tracks: int
    inference_time: float
    frames: list[TrackingFrameResult]


class JobResponse(BaseModel):
    """ジョブの登録・状態取得（Detection2D と同じ形に揃える）."""
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
    partial: list[TrackingFrameResult] = Field(default_factory=list)
