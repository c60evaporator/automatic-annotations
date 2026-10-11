"""add lshape filter

Revision ID: e9b3c5d71a24
Revises: d4f7a2c19e83
Create Date: 2026-10-10 22:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9b3c5d71a24'
down_revision: str | None = 'd4f7a2c19e83'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        # 既存行は L 字フィルタを適用していないので NULL
        batch_op.add_column(sa.Column('lshape_filter', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        batch_op.drop_column('lshape_filter')
