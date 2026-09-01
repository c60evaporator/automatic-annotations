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
    NUSC_CATEGORY_TO_LABEL: dict[str, str] = { # nuScenesのcategoryを2D Object Detectionのラベルに変換するマッピング
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
    LABEL_TO_NUSC_CATEGORY: dict[str, str] = { # 2D Object DetectionのラベルをnuScenesのcategoryに変換するマッピング
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
    LABEL_TO_CATEGORY_GROUP: dict[str, str] = { # 2D Object Detectionのラベル → カテゴリグループ（ラベルプロンプトをまとめて推論する単位）
        "car": "vehicle",
        "truck": "vehicle",
        "construction_vehicle": "vehicle",
        "bus": "vehicle",
        "trailer": "vehicle",
        "barrier": "road_object",
        "motorcycle": "two_wheeler",
        "bicycle": "two_wheeler",
        "pedestrian": "pedestrian",
        "trafficcone": "road_object",
    }

    # --- 2D Object Detection ---------------------------------------------
    DET2D_DEFAULT_SAMPLE_INTERVAL: int = 4
    DET2D_DEFAULT_SCORE_THRESHOLDS: dict[str, float] = {
        "vehicle": 0.35,
        "road_object": 0.25,
        "two_wheeler": 0.3,
        "pedestrian": 0.3,
    }
    # GT_MATCH_IOU_THRESHOLDS: dict[str, float] = {
    #     "vehicle": 0.8,
    #     "road_object": 0.8,
    #     "two_wheeler": 0.8,
    #     "pedestrian": 0.8,
    # }
    DET2D_NMS_SAME_CLASS_IOUS: dict[str, float] = {
        "vehicle": 0.7,
        "road_object": 0.6,
        "two_wheeler": 0.6,
        "pedestrian": 0.6,
    }
    DET2D_NMS_CROSS_CLASS_IOU: float = 0.85

    # TODO: Will be deleted
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
