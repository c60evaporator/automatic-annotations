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
        "movable_object.trafficcone": "trafficcone",
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
        "trafficcone": "movable_object.trafficcone",
    }
    LABEL_TO_CATEGORY_GROUP: dict[str, str] = {  # 検出ラベル -> カテゴリグループ
        "car": "vehicle",
        "truck": "vehicle",
        "construction_vehicle": "vehicle",
        "bus": "vehicle",
        "trailer": "vehicle",
        "barrier": "road_object",
        "trafficcone": "road_object",
        "motorcycle": "two_wheeler",
        "bicycle": "two_wheeler",
        "pedestrian": "pedestrian",
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
