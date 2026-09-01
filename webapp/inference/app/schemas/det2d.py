"""2D Object Detection のリクエスト／レスポンススキーマ.

設計:
  - **入力はフレームのリストで渡す。** 推論サーバーは DB を持たないので、
    「どの画像を処理するか」は webapp 側が list_frames_by_scene() の結果から
    組み立てて渡す。ファイルの実体は共有マウント（/data）越しに読むため、
    画像を HTTP で送らない。シーン単位の一括処理でも転送量が増えない。
  - **閾値はカテゴリグループ単位で持つ。** プロンプトをまとめて投げる単位が
    グループなので、閾値も同じ粒度にしておくとリクエストが素直になる。
    ラベル → グループの対応は webapp 側の設定なので、
    サーバーは解決済みのグループ定義を受け取るだけにする。
  - **1ジョブ = 1回の推論実行。** DB の Detection2DParams 1行に対応する。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FrameRef(BaseModel):
    """処理対象の1フレーム（画像1枚）."""
    sample_data_token: str
    sample_token: str
    # Dataset.dataroot からの相対パス（SampleData.filename）
    filename: str
    channel: str | None = None
    width: int | None = None
    height: int | None = None
    # シーン内でのサンプル連番。進捗表示と結果の並び替えに使う
    sample_idx: int = 0


class LabelGroup(BaseModel):
    """プロンプトをまとめて投げる単位.

    閾値を並列の dict で3つ渡す形にすると、キーの取りこぼしが
    実行時まで分からない。グループごとに1オブジェクトへまとめる。
    """
    name: str
    labels: list[str] = Field(min_length=1)
    score_threshold: float = 0.3
    # 同一クラス内の NMS。グループ内のボックスに適用する
    nms_same_class_iou: float = 0.6


class Detection2DRequest(BaseModel):
    # 画像の解決に必要（DATA_ROOT/<dataroot>/<filename>）
    dataroot: str
    # sample_idx 昇順で渡す想定（サーバー側でも並べ替える）
    frames: list[FrameRef] = Field(min_length=1)

    model_id: str | None = None

    label_groups: list[LabelGroup] = Field(min_length=1)

    # クラスをまたぐ NMS。全グループの推論が終わったフレーム単位で適用する
    nms_cross_class_iou: float = 0.85

    # スタブ用: 1推論あたりの待ち時間（秒）。本実装では無視される
    stub_delay_sec: float | None = None


class BBox2D(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str
    score: float
    # どのグループの推論で得られたか（デバッグ・色分け用）
    group: str | None = None


class Detection2DFrameResult(BaseModel):
    """1フレーム（sample × camera）分の結果.

    ジョブ実行中も partial として逐次返す単位。
    """
    sample_data_token: str
    sample_token: str
    sample_idx: int
    channel: str | None = None
    boxes: list[BBox2D] = Field(default_factory=list)
    inference_time: float | None = None
    error: str | None = None


class Detection2DResult(BaseModel):
    """ジョブ完了時に返る結果."""
    num_frames: int
    num_boxes: int
    inference_time: float
    frames: list[Detection2DFrameResult]


class JobResponse(BaseModel):
    """ジョブの登録・状態取得の共通レスポンス."""
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
    # これまでに完了したフレーム数
    partial_count: int = 0
    # since で要求した差分（完了フレーム）
    partial: list[Detection2DFrameResult] = Field(default_factory=list)
