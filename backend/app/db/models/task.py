"""SQLAlchemy model for agent investigation and repair tasks."""

from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.interaction import Interaction
    from app.db.models.repository import Repository


class Task(Base, TimestampMixin):
    """Represents a debugging or feature engineering task executed by RepoPilot agent."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )  # pending, investigating, planning, editing, testing, approved, completed, failed
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    patch_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    test_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pr_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # Relationships
    repository: Mapped["Repository"] = relationship(
        "Repository",
        back_populates="tasks",
    )
    interactions: Mapped[List["Interaction"]] = relationship(
        "Interaction",
        back_populates="task",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Task id={self.id} title='{self.title}' status='{self.status}' attempts={self.attempts}>"
