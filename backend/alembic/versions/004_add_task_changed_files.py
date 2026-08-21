"""Add changed_files to tasks for persisted review information

Revision ID: 004_add_task_changed_files
Revises: 003_add_task_workspace_path
Create Date: 2026-08-21 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '004_add_task_changed_files'
down_revision: Union[str, None] = '003_add_task_workspace_path'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'tasks',
        sa.Column('changed_files', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('tasks', 'changed_files')
