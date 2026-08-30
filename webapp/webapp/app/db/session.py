"""Session ファクトリ.

Streamlit は同期実行モデルであり、DB も単一ファイルの SQLite なので
AsyncSession は使わず同期 Session に統一する。
"""
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from app.db.engine import get_engine


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    """Session ファクトリ.

    expire_on_commit=False が重要:
    デフォルトの True では commit 直後に全属性が期限切れになり、
    Streamlit 側で描画に使おうとした瞬間に再クエリが走る。
    with ブロックを抜けた後だと DetachedInstanceError になる。
    """
    return sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
        autoflush=False,
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    """トランザクション境界付きの Session.

    使い方:
        with session_scope() as session:
            repo = SceneRepository(session)
            scenes = repo.list_by_dataset(dataset_id)

    NOTE: ORM インスタンスをこのブロックの外へ持ち出さないこと。
    Streamlit の @st.cache_data は戻り値をシリアライズして保持するため、
    Repository 層は ORM オブジェクトではなく dict / dataclass を返す設計にする。
    """
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def read_only_session() -> Iterator[Session]:
    """読み取り専用の Session（commit しない）.

    参照だけの画面描画で使う。書き込みが混ざっても commit されないので、
    意図しない更新を防げる。
    """
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
