"""FastAPI dependencies and database session injection."""

from typing import Annotated, AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

# Type-annotated async DB session dependency
SessionDep = Annotated[AsyncSession, Depends(get_db)]
