"""add box orient merge

Revision ID: a7c4e2f91b36
Revises: f2a6d8c3b915
Create Date: 2026-11-13 01:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7c4e2f91b36'
down_revision: str | None = 'f2a6d8c3b915'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        # 既存行は凸包の結合だけの run なので NULL（global_track_id がそれに当たる）
        batch_op.add_column(sa.Column('hull_merged_track_id', sa.String(), nullable=True))
    with op.batch_alter_table('depth_estimation_params', schema=None) as batch_op:
        batch_op.add_column(sa.Column('box_merge_diagnostics', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('depth_estimation_params', schema=None) as batch_op:
        batch_op.drop_column('box_merge_diagnostics')
    with op.batch_alter_table('box_fittings', schema=None) as batch_op:
        batch_op.drop_column('hull_merged_track_id')
