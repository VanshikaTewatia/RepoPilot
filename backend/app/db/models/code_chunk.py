"""SQLAlchemy model for syntax-aware code chunks and vector embeddings."""

from typing import TYPE_CHECKING, Any, Dict, Optional
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.repository import Repository


class CodeChunk(Base, TimestampMixin):
    """Represents a syntax-aware parsed code chunk with embedding vector."""

    __tablename__ = "code_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(50), nullable=False, default="python")
    symbol_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    symbol_type: Mapped[str] = mapped_column(String(50), nullable=False, default="block")  # function, class, method, import, block
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # 3072-dimension dense embedding vector for Gemini (gemini-embedding-2)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(3072), nullable=True)

    # Metadata dictionary for syntax details, parameters, AST parent info
    chunk_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        default=dict,
        nullable=False,
    )

    # Relationships
    repository: Mapped["Repository"] = relationship(
        "Repository",
        back_populates="code_chunks",
    )

    def __repr__(self) -> str:
        return (
            f"<CodeChunk id={self.id} file='{self.file_path}' "
            f"symbol='{self.symbol_name}' lines={self.start_line}-{self.end_line}>"
        )
