"""SQLAlchemy の DeclarativeBase.

命名規約（naming_convention）は SQLite にとって必須級の設定。
SQLite は ALTER TABLE で制約を変更できないため、Alembic は
`render_as_batch=True` でテーブルを作り直してマイグレーションする。
このとき制約に名前が付いていないと、batch 操作が対象の制約を特定できず
`ValueError: Constraint must have a name` で失敗する。
最初のマイグレーションを切る前にここを決めておくこと（後から変えると
既存 DB の制約名と食い違う）。
"""
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix":  "ix_%(column_0_label)s",
    "uq":  "uq_%(table_name)s_%(column_0_name)s",
    "ck":  "ck_%(table_name)s_%(constraint_name)s",
    "fk":  "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk":  "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
