"""Add workspace_path to tasks for isolated per-task workspace tracking

Revision ID: 003_add_task_workspace_path
Revises: 002_update_embedding_dimension
Create Date: 2026-08-21 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003_add_task_workspace_path'
down_revision: Union[str, None] = '002_update_embedding_dimension'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'tasks',
        sa.Column('workspace_path', sa.String(length=1024), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('tasks', 'workspace_path')
