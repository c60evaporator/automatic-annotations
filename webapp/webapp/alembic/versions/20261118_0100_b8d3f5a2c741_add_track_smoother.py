"""add track smoother

Revision ID: b8d3f5a2c741
Revises: a7c4e2f91b36
Create Date: 2026-11-18 01:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8d3f5a2c741'
down_revision: str | None = 'a7c4e2f91b36'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        # 既存行は動きに基づく結合の無い run なので NULL（global_track_id がそれに当たる）
        batch_op.add_column(sa.Column('box_merged_track_id', sa.String(), nullable=True))
    with op.batch_alter_table('depth_estimation_params', schema=None) as batch_op:
        batch_op.add_column(sa.Column('track_smoother_diagnostics', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('depth_estimation_params', schema=None) as batch_op:
        batch_op.drop_column('track_smoother_diagnostics')
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        batch_op.drop_column('box_merged_track_id')
