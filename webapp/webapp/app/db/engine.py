"""SQLAlchemy Engine の生成と SQLite の PRAGMA 設定"""
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _apply_pragmas(dbapi_connection, connection_record) -> None:
    """接続ごとに SQLite の PRAGMA を適用する.

    PRAGMA は接続単位の設定なので、プールから新しい接続が払い出される
    たびに適用する必要がある（一度実行して終わりではない）。
    """
    settings = get_settings()
    cursor = dbapi_connection.cursor()
    try:
        # 外部キー制約はデフォルトで無効。ON にしないと ondelete='CASCADE' が効かない
        cursor.execute("PRAGMA foreign_keys=ON")
        # 書き込み中でも読み取りをブロックしない（推論書き込み中の UI 表示のため）
        cursor.execute("PRAGMA journal_mode=WAL")
        # WAL では NORMAL でも十分な耐久性があり、書き込みが大幅に速い
        cursor.execute("PRAGMA synchronous=NORMAL")
        # ロック競合時に即エラーにせず待つ
        cursor.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()


def create_app_engine(url: str | None = None, *, echo: bool | None = None) -> Engine:
    """アプリ用 Engine を生成する（PRAGMA 適用込み）."""
    settings = get_settings()
    url = url or settings.database_url
    echo = settings.SQL_ECHO if echo is None else echo

    # SQLite ファイルの置き場所を用意しておく
    if settings.SQLITE_PATH.parent and not settings.SQLITE_PATH.parent.exists():
        settings.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        url,
        echo=echo,
        # Streamlit は ScriptRunner スレッドでスクリプトを再実行するため、
        # 接続を生成したスレッド以外からの利用を許可する必要がある
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    event.listen(engine, "connect", _apply_pragmas)
    logger.info("engine created: %s", url)
    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """プロセス内で共有する Engine.

    Streamlit から使う場合は、再実行のたびに Engine が作られないよう
    さらに @st.cache_resource でラップした薄い関数を用意すること。
    """
    return create_app_engine()


def create_migration_engine(url: str | None = None) -> Engine:
    """Alembic 専用の Engine.

    アプリ用と分けている理由:
      - batch モードはテーブルを作り直して移行するため、外部キーが ON だと
        再作成の過程で ON DELETE CASCADE が発火して行が消える危険がある。
        よって PRAGMA foreign_keys は適用しない（= OFF のまま）。
      - マイグレーションは単発実行なので接続プールは不要。
    """
    settings = get_settings()
    url = url or settings.database_url
    if settings.SQLITE_PATH.parent and not settings.SQLITE_PATH.parent.exists():
        settings.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        url,
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
