"""add distance filter

Revision ID: f2a6d8c3b915
Revises: e9b3c5d71a24
Create Date: 2026-10-14 01:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2a6d8c3b915'
down_revision: str | None = 'e9b3c5d71a24'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        # 既存行は距離フィルタを適用していないので NULL
        batch_op.add_column(sa.Column('distance_filter', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        batch_op.drop_column('distance_filter')
