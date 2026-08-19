"""Update embedding vector dimension from 768 to 3072 for gemini-embedding-2

Revision ID: 002_update_embedding_dimension
Revises: 001_initial_schema
Create Date: 2026-08-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '002_update_embedding_dimension'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'code_chunks',
        'embedding',
        existing_type=Vector(768),
        type_=Vector(3072),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        'code_chunks',
        'embedding',
        existing_type=Vector(3072),
        type_=Vector(768),
        nullable=True,
    )
