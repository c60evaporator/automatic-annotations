"""環境変数・設定（Pydantic Settings）"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """アプリケーション設定.

    値は .env または環境変数から読み込む。
    ホスト側のデータフォルダ（HOST_DATA_ROOT）は docker-compose 側で
    DATA_ROOT にマウントされる想定で、アプリからは DATA_ROOT のみを参照する。
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    APP_NAME:  str = "automatic-annotation-app"
    LOG_LEVEL: str = "INFO"

    # --- データ配置 -------------------------------------------------------
    # コンテナ内でのデータセットルート（ホスト側 HOST_DATA_ROOT のマウント先）
    DATA_ROOT: Path = Path("/data")
    # 推論の派生成果物（深度マップ .npz 等）の保存先
    DERIVED_ROOT: Path = Path("/data/_derived")

    # --- データベース -----------------------------------------------------
    SQLITE_PATH: Path = Path("/db/app.db")
    SQL_ECHO:    bool = False
    # 書き込みロック待ちの上限（ミリ秒）。推論結果の一括書き込みと
    # UI の読み取りが競合したときの "database is locked" を防ぐ
    SQLITE_BUSY_TIMEOUT_MS: int = 10_000

    # --- 推論サーバー -----------------------------------------------------
    INFERENCE_BASE_URL:    str   = "http://inference:8000"
    INFERENCE_TIMEOUT_SEC: float = 1800.0

    # --- ラベル変換 -----------------------------------------------------
    NUSC_CATEGORY_TO_LABEL: dict[str, str] = {  # nuScenes category -> 検出ラベル
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.construction": "construction_vehicle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.trailer": "trailer",
        "movable_object.barrier": "barrier",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bicycle": "bicycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
    }
    LABEL_TO_NUSC_CATEGORY: dict[str, str] = {  # 検出ラベル -> nuScenes category
        "car": "vehicle.car",
        "truck": "vehicle.truck",
        "construction_vehicle": "vehicle.construction",
        "bus": "vehicle.bus.rigid",
        "trailer": "vehicle.trailer",
        "barrier": "movable_object.barrier",
        "motorcycle": "vehicle.motorcycle",
        "bicycle": "vehicle.bicycle",
        "pedestrian": "human.pedestrian.adult",
        "traffic_cone": "movable_object.trafficcone",
    }
    LABEL_TO_CATEGORY_GROUP: dict[str, str] = {  # 検出ラベル -> カテゴリグループ
        "car": "vehicle",
        "truck": "vehicle",
        "construction_vehicle": "vehicle",
        "bus": "vehicle",
        "trailer": "vehicle",
        "barrier": "road_object",
        "traffic_cone": "road_object",
        "motorcycle": "two_wheeler",
        "bicycle": "two_wheeler",
        "pedestrian": "pedestrian",
    }

    # --- 表示 -------------------------------------------------------------
    # カメラ画像の表示順（channel 名）。
    # DB から取る順序は channel 名の昇順（CAM_BACK が先頭）になるため、
    # 確認頻度の高い CAM_FRONT を先頭に置く。
    # 画像は 2 列グリッドに並ぶので、先頭 2 つが 1 行目になる。
    # ここに無い channel は末尾へ回る（channel 名順）
    CAM_DISPLAY_ORDER: list[str] = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
    ]

    # Radius Outlier Removal の nb_points にかけるラベルごとの倍率。
    # 小さい物体（traffic_cone / pedestrian）は点群自体が疎で、
    # 全ラベル共通の nb_points だと点が丸ごと消えてしまう。
    # 実際に使う値は「指定した nb_points × この倍率」を四捨五入したもの。
    # Depth 側・LiDAR 側の両方で共通に使う
    NB_POINTS_RATIO: dict[str, float] = {
        "car": 1.0,
        "truck": 1.0,
        "construction_vehicle": 1.0,
        "bus": 1.0,
        "trailer": 1.0,
        "barrier": 1.0,
        "traffic_cone": 0.4,
        "motorcycle": 1.0,
        "bicycle": 1.0,
        "pedestrian": 0.8,
    }

    # --- 2D Object Detection ---------------------------------------------
    DET2D_DEFAULT_SAMPLE_INTERVAL: int = 4
    DET2D_DEFAULT_SCORE_THRESHOLDS: dict[str, float] = {
        "vehicle": 0.35,
        "road_object": 0.25,
        "two_wheeler": 0.3,
        "pedestrian": 0.3,
    }
    DET2D_NMS_SAME_CLASS_IOUS: dict[str, float] = {
        "vehicle": 0.7,
        "road_object": 0.6,
        "two_wheeler": 0.6,
        "pedestrian": 0.6,
    }
    DET2D_NMS_CROSS_CLASS_IOU: float = 0.85

    # 再実行時、この IoU 以上で重なる手修正ボックスがあれば、
    # 推論ボックスを手修正ボックスで置き換える（推論サーバーへは送らない）
    DET2D_MANUAL_REPLACE_IOU: float = 0.5

    # 1シーンあたり保持する run の上限。超えたら古いものから削除する。
    # ただし Instance Tracking から参照されている run は削除しない
    # （消すとトラッキング結果と 3D ボックスまで CASCADE で消えるため）
    DET2D_MAX_RUNS_PER_SCENE: int = 10

    # run の記録に残すモデル名（推論サーバー側の実体と合わせる）
    DET2D_MODEL_NAME: str = "groundingdino_swinb_cogcoor"

    # --- Instance Tracking -----------------------------------------------
    SWEEPS_PER_SAMPLE: int = 6
    DEFAULT_TRACKING_NUM_SWEEPS: int = 2
    DEFAULT_TRACKING_IOU_THRESHOLD: float = 0.5
    DEFAULT_TRACKING_IOU_METHOD: str = "box"
    DEFAULT_TRACKING_IOU_LABEL_MATCH: str = "label"
    DEFAULT_TRACKING_MASK_SCORE_THRESHOLD: float = 0.5
    TRACKING_MAX_RUNS_PER_SCENE: int = 10
    TRACKING_MODEL_NAME: str = "sam2.1_hiera_large"
    TRACKING_STUB_DELAY_SEC: float | None = 0.05

    # --- Depth Estimation & Box Fitting ------------------------------------
    DEPTH_MODEL_NAME: str = "depth-anything-3-large"
    # 深度マップの保存倍率。1/2 にすると容量は 1/4（1 run 約 565MB → 141MB）
    DEPTH_MAP_DOWNSCALE: float = 0.5
    # 保持する run 数。派生ファイルが 1 run 約 280MB あるため少なめにする
    DEPTH_MAX_RUNS_PER_SCENE: int = 3
    # DB に保存するインスタンス点群の上限（ボクセル間引き後）
    BOXFIT_STORED_POINTS_MAX: int = 500
    # Plotly へ渡す点数の上限。超えると転送量と描画が重くなる
    POINTCLOUD_DISPLAY_MAX_POINTS: int = 50_000

    # マスクのクロージング（General タブ）
    MASK_DILATION_DEFAULT: int = 5
    MASK_DILATION_MAX: int = 31
    MASK_EROSION_DEFAULT: int = 5
    MASK_EROSION_MAX: int = 31

    # 深度点群のフィルタ（Depth Estimation タブ）
    DEPTH_ROR_NB_POINTS_DEFAULT: int = 8
    DEPTH_ROR_NB_POINTS_MAX: int = 50
    DEPTH_ROR_RADIUS_DEFAULT: float = 0.6
    DEPTH_ROR_RADIUS_MAX: float = 5.0
    DEPTH_DBSCAN_EPS_DEFAULT: float = 1.0
    DEPTH_DBSCAN_EPS_MAX: float = 5.0
    DEPTH_DBSCAN_MIN_SAMPLES_DEFAULT: int = 12
    DEPTH_DBSCAN_MIN_SAMPLES_MAX: int = 100

    # LiDAR 点群のフィルタ（LiDAR Pointcloud タブ）
    LIDAR_NUM_SWEEPS_DEFAULT: int = 5
    LIDAR_MIN_POINTS_DEFAULT: int = 10
    LIDAR_MIN_POINTS_MAX: int = 200
    LIDAR_ROR_NB_POINTS_DEFAULT: int = 4
    LIDAR_ROR_NB_POINTS_MAX: int = 50
    LIDAR_ROR_RADIUS_DEFAULT: float = 0.8
    LIDAR_ROR_RADIUS_MAX: float = 5.0
    LIDAR_DBSCAN_EPS_DEFAULT: float = 0.8
    LIDAR_DBSCAN_EPS_MAX: float = 5.0
    LIDAR_DBSCAN_MIN_SAMPLES_DEFAULT: int = 5
    LIDAR_DBSCAN_MIN_SAMPLES_MAX: int = 100

    # 点群ビューの Global View の視点。
    # global 座標に対する既定の視線位置と上方向で、
    # ego_pose の回転を打ち消して適用される（車両が回っても向きが変わらない）
    DEFAULT_GLOBAL_POINTCLOUD_EYE: list[float] = [-1.0, -0.5, 1.5]
    DEFAULT_GLOBAL_POINTCLOUD_UP: list[float] = [0.0, 0.0, 1.0]

    # 点群ビューの既定表示
    SHOW_RAW_LIDAR: bool = False
    SHOW_RAW_LIDAR_GROUND: bool = False
    SHOW_RAW_DEPTH_POINTCLOUD: bool = False

    BOXFIT_STUB_DELAY_SEC: float | None = 0.02

    # スタブ推論の1回あたりの待ち時間（本実装に差し替えたら None にする）
    DET2D_STUB_DELAY_SEC: float | None = 0.05

    @property
    def database_url(self) -> str:
        """SQLAlchemy 用の同期 DSN.

        Streamlit は同期実行モデルであり、DB も単一ファイルの SQLite なので
        AsyncSession は利点がなく、複雑さだけが増すため同期ドライバを使う。
        """
        return f"sqlite+pysqlite:///{self.SQLITE_PATH}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """設定のシングルトン。プロセス内で1度だけ読み込む。"""
    return Settings()
