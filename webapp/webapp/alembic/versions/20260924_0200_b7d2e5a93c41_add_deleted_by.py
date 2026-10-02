"""add deleted by

Revision ID: b7d2e5a93c41
Revises: a1c4d9e7b2f0
Create Date: 2026-09-24 02:00:00.000000+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7d2e5a93c41'
down_revision: str | None = 'a1c4d9e7b2f0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('detection_2ds', schema=None) as batch_op:
        batch_op.add_column(sa.Column('deleted_by', sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('detection_2ds', schema=None) as batch_op:
        batch_op.drop_column('deleted_by')
