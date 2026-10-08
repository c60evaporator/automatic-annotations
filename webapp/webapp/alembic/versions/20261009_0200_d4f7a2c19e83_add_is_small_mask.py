"""add is small mask

Revision ID: d4f7a2c19e83
Revises: c8e1f4b60d92
Create Date: 2026-10-09 02:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4f7a2c19e83'
down_revision: str | None = 'c8e1f4b60d92'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        # 既存行は小物体用の処理を適用していないので False
        batch_op.add_column(sa.Column(
            'is_small_mask', sa.Boolean(), nullable=False,
            server_default=sa.text('0'),
        ))


def downgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        batch_op.drop_column('is_small_mask')
