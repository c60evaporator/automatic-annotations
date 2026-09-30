"""add merge params and diagnostics

Revision ID: a1c4d9e7b2f0
Revises: 3e72669dffbc
Create Date: 2026-09-23 01:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1c4d9e7b2f0'
down_revision: str | None = '3e72669dffbc'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('depth_estimation_params', schema=None) as batch_op:
        # NOT NULL の列は server_default を付ける（既存行があるため）
        batch_op.add_column(sa.Column(
            'merge_params', sa.JSON(), nullable=False, server_default=sa.text("'{}'")
        ))
        batch_op.add_column(sa.Column('merge_diagnostics', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('depth_estimation_params', schema=None) as batch_op:
        batch_op.drop_column('merge_diagnostics')
        batch_op.drop_column('merge_params')
