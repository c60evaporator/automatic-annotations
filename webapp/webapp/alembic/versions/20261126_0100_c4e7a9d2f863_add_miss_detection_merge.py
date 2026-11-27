"""add miss detection merge

Revision ID: c4e7a9d2f863
Revises: b8d3f5a2c741
Create Date: 2026-11-26 01:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4e7a9d2f863'
down_revision: str | None = 'b8d3f5a2c741'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('instance_tracking_2d_params', schema=None) as batch_op:
        # 既存の run は検出漏れの結合を行っていない
        batch_op.add_column(sa.Column('miss_detection_merge', sa.Boolean(), server_default=sa.text('0'), nullable=False))
        batch_op.add_column(sa.Column('miss_detection_merge_iou_threshold', sa.Float(), server_default=sa.text('0.5'), nullable=False))


def downgrade() -> None:
    with op.batch_alter_table('instance_tracking_2d_params', schema=None) as batch_op:
        batch_op.drop_column('miss_detection_merge_iou_threshold')
        batch_op.drop_column('miss_detection_merge')
