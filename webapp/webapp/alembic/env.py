"""Alembic 実行環境.

SQLite 向けのポイント:
  - render_as_batch=True
      SQLite は ALTER TABLE でのカラム削除・型変更・制約変更ができない。
      batch モードは「新テーブル作成 → データコピー → 旧テーブル削除 → リネーム」
      に展開してくれるため、これが無いと大半のマイグレーションが失敗する。
  - 外部キーは OFF のまま実行する（create_migration_engine を使用）。
      batch モードはテーブルを作り直すため、FK が ON だと途中で
      ON DELETE CASCADE が発火して行が消える危険がある。
  - Base.metadata には naming_convention が必要（app/db/base.py 参照）。
      制約に名前が無いと batch 操作が対象を特定できない。
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection

from app.core.config import get_settings
from app.db.base import Base
from app.db.engine import create_migration_engine

# Alembic が全テーブルを検出できるよう、models パッケージを import する。
# app/models/__init__.py が全モデルを import している前提。
import app.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

settings = get_settings()


def _get_url() -> str:
    """接続先 URL を決定する.

    優先順位:
      1. `alembic -x db_url=...` で明示指定された値（テスト・一時 DB 用）
      2. Settings（= .env / 環境変数）
    """
    x_args = context.get_x_argument(as_dictionary=True)
    return x_args.get("db_url") or settings.database_url


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    """autogenerate の対象から除外するオブジェクトを判定する.

    SQLite が内部生成する sqlite_ 系テーブルを拾わないようにする。
    """
    if type_ == "table" and name is not None and name.startswith("sqlite_"):
        return False
    return True


def _configure(connection: Connection | None = None, url: str | None = None) -> None:
    """オンライン／オフライン共通の context 設定."""
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        # --- SQLite 必須設定 ---
        render_as_batch=True,
        # --- 差分検出の精度を上げる ---
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
        # 単一 DB 構成なのでリテラルバインドで十分
        literal_binds=(connection is None),
        dialect_opts={"paramstyle": "named"},
    )


def run_migrations_offline() -> None:
    """DBに接続せず SQL を出力するモード（`alembic upgrade head --sql`）."""
    _configure(url=_get_url())
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """実 DB に接続して適用するモード."""
    engine = create_migration_engine(_get_url())
    with engine.connect() as connection:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
