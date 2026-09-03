"""自動アノテーションパイプラインの中間出力保存用テーブル.

パイプラインの3ステップと、それぞれの実行パラメータ（*Params）・結果テーブルの対応:

  Step 1  2D Object Detection      : Detection2DParams        → Detection2D
  Step 2  Instance Tracking (SAM2) : InstanceTracking2DParams → InstanceTracking2D
  Step 3  Depth Est. & Box Fitting : DepthEstimationParams    → DepthEstimation
                                                              → SampleAnnotation(source='auto')

*Params は「1回の推論実行」を表すレコードで、後段は前段の *Params.id を参照する。
これにより「どの検出結果からどのトラッキングが生まれたか」を辿れ、
*Params 行を削除すれば CASCADE でその実行の成果物だけが消える。
"""
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.annotation import SampleAnnotation
    from app.models.dataset import Dataset
    from app.models.scene import Scene
    from app.models.sensor import SampleData


# 推論実行の状態。3ステップの *Params で共通に使う。
#   running   : 実行中（ended_at は NULL）
#   succeeded : 正常終了。後段の入力として選ばれるのはこれだけ
#   failed    : 例外で中断
#   cancelled : ユーザーがキャンセル
#
# ended_at の NULL 判定だけでは「失敗」と「キャンセル」を区別できず、
# 「最新の成功した run」を引く問合せも書けないため、明示的な列を持つ。
RUN_STATUS_RUNNING = "running"
RUN_STATUS_SUCCEEDED = "succeeded"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_CANCELLED = "cancelled"
RUN_STATUSES = (
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    RUN_STATUS_FAILED,
    RUN_STATUS_CANCELLED,
)

# インスタンスの由来。区間境界の sample には両方が存在する:
#   prompt     … その sample のプロンプト（Detection2D のボックス）から得た結果
#   propagated … 前の区間から伝播してきた結果
# 通常表示では prompt を優先し、比較表示では左右に並べる。
# 両方を残さないと「伝播がどれだけずれたか」を後から確認できない。
INSTANCE_ORIGIN_PROMPT = "prompt"
INSTANCE_ORIGIN_PROPAGATED = "propagated"
INSTANCE_ORIGINS = (INSTANCE_ORIGIN_PROMPT, INSTANCE_ORIGIN_PROPAGATED)

# トラッキングの track_id 引き継ぎ判定に使う IoU の計算方法
IOU_METHOD_BOX = "box"     # マスクの外接矩形どうしの IoU
IOU_METHOD_MASK = "mask"   # マスクどうしの IoU
IOU_METHODS = (IOU_METHOD_BOX, IOU_METHOD_MASK)

# 上記マッチング時に、ラベルの一致をどこまで要求するか
IOU_LABEL_MATCH_LABEL = "label"                    # ラベル完全一致
IOU_LABEL_MATCH_CATEGORY_GROUP = "category_group"  # カテゴリグループ一致
IOU_LABEL_MATCH_NONE = "none"                      # ラベルを問わない
IOU_LABEL_MATCHES = (
    IOU_LABEL_MATCH_LABEL,
    IOU_LABEL_MATCH_CATEGORY_GROUP,
    IOU_LABEL_MATCH_NONE,
)


# =============================================================================
# Step 1: 2D Object Detection (Grounding DINO)
# =============================================================================

class Detection2DParams(Base):
    """2D物体検出の実行単位とパラメータ"""
    __tablename__ = "detection_2d_params"
    __table_args__ = (
        # 「このシーンで最後に成功した run」を引くための複合インデックス。
        # ORDER BY started_at DESC LIMIT 1 を index scan で返せる
        Index("ix_detection_2d_params_scene_started", "scene_token", "started_at"),
        # status での絞り込みを伴う検索用
        Index("ix_detection_2d_params_scene_status_started",
              "scene_token", "status", "started_at"),
    )
    # Columns
    id:         Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 推論はシーン単位で実行する
    scene_token: Mapped[str] = mapped_column(
        ForeignKey("scenes.token", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String, nullable=False)  # 'IDEA-Research/grounding-dino-base' 等
    sample_interval: Mapped[int] = mapped_column(Integer, nullable=False)  # 推論を実施するキーフレーム間隔
    # ラベル体系のマッピング（いずれも {key: value} の dict）
    # 元 nuScenes の category_name → 推論で使用するラベル名
    nusc_category_to_label:  Mapped[dict] = mapped_column(JSON, nullable=False)
    # 推論ラベル名 → 最終的な nuScenes category_name
    label_to_nusc_category:  Mapped[dict] = mapped_column(JSON, nullable=False)
    # 推論ラベル名 → カテゴリグループ（ラベルプロンプトをまとめて推論する単位）
    label_to_category_group: Mapped[dict] = mapped_column(JSON, nullable=False)
    # 閾値はいずれもカテゴリごとに変えるため {category: value} の dict
    score_threshold:      Mapped[dict] = mapped_column(JSON, nullable=False)
    nms_same_class_ious:  Mapped[dict] = mapped_column(JSON, nullable=False)
    nms_cross_class_ious: Mapped[dict] = mapped_column(JSON, nullable=False)
    # 実行結果メタ
    num_inferences: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    # 純粋な推論時間（オーバーヘッドを除くため ended_at - started_at より短い）
    inference_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 実行状態（RUN_STATUS_* のいずれか）
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text(f"'{RUN_STATUS_RUNNING}'")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # 実行中は NULL。完了時に埋める
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    scene:   Mapped["Scene"]   = relationship()
    detection_2ds: Mapped[list["Detection2D"]] = relationship(
        back_populates="detection_2d_params",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    instance_tracking_params: Mapped[list["InstanceTracking2DParams"]] = relationship(
        back_populates="detection_2d_params",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Detection2D(Base):
    """2D物体検出の結果（Grounding DINO）"""
    __tablename__ = "detection_2ds"
    __table_args__ = (
        # 実行単位 × フレームでの取得（描画時の主クエリ）
        Index("ix_detection_2ds_params_sample_data", "detection_2d_params_id", "sample_data_token"),
    )
    # Columns
    id:         Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample_data_token: Mapped[str] = mapped_column(
        ForeignKey("sample_data.token", ondelete="CASCADE"), nullable=False, index=True
    )
    detection_2d_params_id: Mapped[str] = mapped_column(
        ForeignKey("detection_2d_params.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 2D bounding box（画像ピクセル座標）
    xmin:  Mapped[int] = mapped_column(Integer, nullable=False)
    ymin:  Mapped[int] = mapped_column(Integer, nullable=False)
    xmax:  Mapped[int] = mapped_column(Integer, nullable=False)
    ymax:  Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # UI 上で手修正されたか
    manually_modified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("0")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    sample_data: Mapped["SampleData"] = relationship(back_populates="detection_2ds")
    detection_2d_params: Mapped["Detection2DParams"] = relationship(
        back_populates="detection_2ds"
    )
    instance_tracking_2ds: Mapped[list["InstanceTracking2D"]] = relationship(
        back_populates="detection_2d",
        passive_deletes=True,
    )


# =============================================================================
# Step 2: Instance Tracking (SAM2)
# =============================================================================

class InstanceTracking2DParams(Base):
    """インスタンストラッキングの実行単位とパラメータ"""
    __tablename__ = "instance_tracking_2d_params"
    __table_args__ = (
        # 「このシーンで最後に成功した run」を引くための複合インデックス。
        # ORDER BY started_at DESC LIMIT 1 を index scan で返せる
        Index("ix_instance_tracking_2d_params_scene_started", "scene_token", "started_at"),
        # status での絞り込みを伴う検索用
        Index("ix_instance_tracking_2d_params_scene_status_started",
              "scene_token", "status", "started_at"),
    )
    # Columns
    id:         Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scene_token: Mapped[str] = mapped_column(
        ForeignKey("scenes.token", ondelete="CASCADE"), nullable=False, index=True
    )
    # プロンプトとして使用した検出結果の実行単位（系譜）
    detection_2d_params_id: Mapped[str] = mapped_column(
        ForeignKey("detection_2d_params.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String, nullable=False)  # 'facebook/sam2.1-hiera-large' 等
    # プロンプトを与える間隔。Detection2D の run から引き継ぐ
    sample_interval: Mapped[int] = mapped_column(Integer, nullable=False)
    # 1 sample あたり何フレーム使うか（1 ならキーフレームのみ）。
    # 非キーフレームは伝播の精度を上げるために使い、結果自体は表示に使わない
    num_sweeps: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    # マスク採用閾値・トラック管理
    mask_score_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    # 既存トラックと同一とみなす IoU 閾値（これ未満なら新規 track_id を発番）。
    # 前のプロンプト区間から伝播したインスタンスと、
    # 次のプロンプトで得たインスタンスの照合に使う
    new_track_iou_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    # 上記照合の IoU 計算方法（IOU_METHOD_*）
    iou_method: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text(f"'{IOU_METHOD_BOX}'")
    )
    # 上記照合でラベル一致をどこまで要求するか（IOU_LABEL_MATCH_*）
    iou_label_match: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text(f"'{IOU_LABEL_MATCH_LABEL}'")
    )
    # 連続で見失った際にトラックを打ち切るフレーム数
    max_lost_frames: Mapped[int] = mapped_column(Integer, nullable=False)
    # 実行結果メタ
    num_inferences: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    num_tracks:     Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    inference_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 実行状態（RUN_STATUS_* のいずれか）
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text(f"'{RUN_STATUS_RUNNING}'")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    scene:   Mapped["Scene"]   = relationship()
    detection_2d_params: Mapped["Detection2DParams"] = relationship(
        back_populates="instance_tracking_params"
    )
    instance_tracking_2ds: Mapped[list["InstanceTracking2D"]] = relationship(
        back_populates="instance_tracking_2d_params",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    depth_estimation_params: Mapped[list["DepthEstimationParams"]] = relationship(
        back_populates="instance_tracking_2d_params",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class InstanceTracking2D(Base):
    """2Dインスタンスセグメンテーション＋トラッキングの結果（SAM2）.

    1行 = 1フレーム内の1インスタンス。
    """
    __tablename__ = "instance_tracking_2ds"
    __table_args__ = (
        # 実行単位 × フレームでの取得（描画時の主クエリ）
        Index(
            "ix_instance_tracking_2ds_params_sample_data",
            "instance_tracking_2d_params_id",
            "sample_data_token",
        ),
        # track_id 単位での時系列取得（トラック可視化・3D化の入力）
        Index(
            "ix_instance_tracking_2ds_params_track",
            "instance_tracking_2d_params_id",
            "track_id",
        ),
        # 由来で絞った取得（比較表示・通常表示の切り替え）
        Index(
            "ix_instance_tracking_2ds_params_origin",
            "instance_tracking_2d_params_id",
            "sample_data_token",
            "origin",
        ),
    )
    # Columns
    id:         Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample_data_token: Mapped[str] = mapped_column(
        ForeignKey("sample_data.token", ondelete="CASCADE"), nullable=False, index=True
    )
    instance_tracking_2d_params_id: Mapped[str] = mapped_column(
        ForeignKey("instance_tracking_2d_params.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # プロンプトとして与えた検出ボックス（トラック継続フレームでは NULL）
    detection_2d_id: Mapped[str | None] = mapped_column(
        ForeignKey("detection_2ds.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # 実行単位内で一意なトラック識別子（カメラ channel を跨がない前提）
    track_id: Mapped[str] = mapped_column(String, nullable=False)
    # インスタンスの由来（INSTANCE_ORIGIN_*）。
    # 区間境界の sample には prompt と propagated の両方の行が入る
    origin: Mapped[str] = mapped_column(
        String, nullable=False,
        server_default=text(f"'{INSTANCE_ORIGIN_PROMPT}'"),
    )
    label:    Mapped[str] = mapped_column(String, nullable=False)
    # マスク本体は COCO RLE 形式で保持: {"size": [h, w], "counts": "..."}
    # NOTE: 生の bool 配列は SQLite に載せないこと。RLE でもシーン全体で数十MB規模に
    #       なる場合は mask_path 方式（.npz をディスクに置く）へ切り替える
    mask_rle:  Mapped[dict] = mapped_column(JSON, nullable=False)
    mask_area: Mapped[int]  = mapped_column(Integer, nullable=False)
    # マスクの外接矩形（描画・デバッグ用のキャッシュ）
    xmin: Mapped[int] = mapped_column(Integer, nullable=False)
    ymin: Mapped[int] = mapped_column(Integer, nullable=False)
    xmax: Mapped[int] = mapped_column(Integer, nullable=False)
    ymax: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 3D 化の結果ひも付けられた Instance（Step 3 完了後に埋まる）
    instance_token: Mapped[str | None] = mapped_column(
        ForeignKey("instances.token", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    manually_modified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("0")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    sample_data: Mapped["SampleData"] = relationship(back_populates="instance_tracking_2ds")
    instance_tracking_2d_params: Mapped["InstanceTracking2DParams"] = relationship(
        back_populates="instance_tracking_2ds"
    )
    detection_2d: Mapped["Detection2D | None"] = relationship(
        back_populates="instance_tracking_2ds"
    )


# =============================================================================
# Step 3: Depth Estimation & Box Fitting (Depth-Anything-3)
# =============================================================================

class DepthEstimationParams(Base):
    """深度推定＋3Dボックスフィッティングの実行単位とパラメータ"""
    __tablename__ = "depth_estimation_params"
    __table_args__ = (
        # 「このシーンで最後に成功した run」を引くための複合インデックス。
        # ORDER BY started_at DESC LIMIT 1 を index scan で返せる
        Index("ix_depth_estimation_params_scene_started", "scene_token", "started_at"),
        # status での絞り込みを伴う検索用
        Index("ix_depth_estimation_params_scene_status_started",
              "scene_token", "status", "started_at"),
    )
    # Columns
    id:         Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scene_token: Mapped[str] = mapped_column(
        ForeignKey("scenes.token", ondelete="CASCADE"), nullable=False, index=True
    )
    # マスクの供給元となるトラッキング実行単位（系譜）
    instance_tracking_2d_params_id: Mapped[str] = mapped_column(
        ForeignKey("instance_tracking_2d_params.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String, nullable=False)  # 'depth-anything-3-*' 等
    sample_interval: Mapped[int] = mapped_column(Integer, nullable=False)
    # 深度のスケール合わせ方式: 'lidar_lstsq' | 'lidar_median' | 'none'
    depth_alignment_method: Mapped[str] = mapped_column(String, nullable=False)
    # LiDAR 点群と推定点群の混合方式: 'lidar_only' | 'depth_only' | 'mixed'
    point_fusion_mode: Mapped[str] = mapped_column(String, nullable=False)
    # ボックス当てはめ方式: 'pca' | 'l_shape' | 'min_area_rect'
    box_fitting_method: Mapped[str] = mapped_column(String, nullable=False)
    # マスク内点群のフィルタ設定
    min_points_per_box:  Mapped[int]   = mapped_column(Integer, nullable=False)
    outlier_percentile:  Mapped[float] = mapped_column(Float, nullable=False)
    max_depth:           Mapped[float] = mapped_column(Float, nullable=False)  # [m] これ以遠は棄却
    # カテゴリごとの寸法事前分布など、方式依存の追加設定
    extra_options: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # 実行結果メタ
    num_inferences: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    num_boxes:      Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    inference_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 実行状態（RUN_STATUS_* のいずれか）
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text(f"'{RUN_STATUS_RUNNING}'")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    scene:   Mapped["Scene"]   = relationship()
    instance_tracking_2d_params: Mapped["InstanceTracking2DParams"] = relationship(
        back_populates="depth_estimation_params"
    )
    depth_estimations: Mapped[list["DepthEstimation"]] = relationship(
        back_populates="depth_estimation_params",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    # この実行で生成された3Dボックス（source='auto'）
    # この実行が生成した3Dボックスは、実行単位の削除で一緒に消す（再推論のやり直し用）。
    # GT（source='imported'）は depth_estimation_params_id が NULL なのでこの集合に入らず残る
    annotations: Mapped[list["SampleAnnotation"]] = relationship(
        back_populates="depth_estimation_params",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DepthEstimation(Base):
    """深度推定の結果（1行 = 1カメラフレーム）.

    深度マップ本体は DB に入れず、.npz としてディスクに保存してパスのみ保持する。
    """
    __tablename__ = "depth_estimations"
    __table_args__ = (
        Index(
            "ix_depth_estimations_params_sample_data",
            "depth_estimation_params_id",
            "sample_data_token",
        ),
    )
    # Columns
    id:         Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample_data_token: Mapped[str] = mapped_column(
        ForeignKey("sample_data.token", ondelete="CASCADE"), nullable=False, index=True
    )
    depth_estimation_params_id: Mapped[str] = mapped_column(
        ForeignKey("depth_estimation_params.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 深度マップ .npz のパス（settings.DATA_ROOT からの相対パス）
    depth_path: Mapped[str] = mapped_column(String, nullable=False)
    # LiDAR に合わせたスケール補正係数（depth_metric = raw * scale + shift）
    scale: Mapped[float | None] = mapped_column(Float, nullable=True)
    shift: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 補正後の統計（UI のカラーマップレンジ決定に使用）
    min_depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    # LiDAR と混合した後の有効点数
    num_points: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    sample_data: Mapped["SampleData"] = relationship(back_populates="depth_estimations")
    depth_estimation_params: Mapped["DepthEstimationParams"] = relationship(
        back_populates="depth_estimations"
    )
