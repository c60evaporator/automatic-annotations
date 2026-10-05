"""add depth used

Revision ID: c8e1f4b60d92
Revises: b7d2e5a93c41
Create Date: 2026-09-25 03:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8e1f4b60d92'
down_revision: str | None = 'b7d2e5a93c41'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        # 既存行は深度点群を使って当てはめているので True
        batch_op.add_column(sa.Column(
            'depth_used', sa.Boolean(), nullable=False, server_default=sa.text('1')
        ))


def downgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        batch_op.drop_column('depth_used')
