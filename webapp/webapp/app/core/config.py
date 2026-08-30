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
