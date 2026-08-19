"""Repository management and indexing API routes."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models.repository import Repository
from app.services.indexing.indexer import RepositoryIndexer

router = APIRouter(prefix="/repositories", tags=["Repositories"])


class RepositoryCreate(BaseModel):
    name: str = Field(..., example="my-project")
    local_path: str = Field(..., example="C:/Users/Lakshay/RepoPilot")
    remote_url: str | None = None
    default_branch: str = "main"


class RepositoryResponse(BaseModel):
    id: int
    name: str
    local_path: str
    remote_url: str | None
    default_branch: str
    status: str
    indexed_at: datetime | None


@router.post("", response_model=RepositoryResponse, status_code=status.HTTP_201_CREATED)
async def create_repository(
    payload: RepositoryCreate,
    db: SessionDep,
) -> Repository:
    """Register a new repository for indexing and analysis."""
    repo_path = Path(payload.local_path).resolve()
    if not repo_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Local path does not exist: {payload.local_path}",
        )

    repo = Repository(
        name=payload.name,
        local_path=str(repo_path),
        remote_url=payload.remote_url,
        default_branch=payload.default_branch,
        status="pending",
    )
    db.add(repo)
    await db.commit()
    await db.refresh(repo)
    return repo


@router.get("", response_model=List[RepositoryResponse])
async def list_repositories(db: SessionDep) -> List[Repository]:
    """List all registered repositories."""
    result = await db.execute(select(Repository).order_by(Repository.id.desc()))
    return list(result.scalars().all())


@router.post("/{repo_id}/index")
async def index_repository_endpoint(
    repo_id: int,
    db: SessionDep,
) -> Dict[str, Any]:
    """Trigger syntax-aware indexing and chunk extraction for a repository."""
    result = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = result.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    indexer = RepositoryIndexer(repo_path=repo.local_path, repo_id=repo.id)
    total_chunks, reused_chunks = await indexer.index_repository(db)

    repo.status = "indexed"
    repo.indexed_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "success": True,
        "repository_id": repo_id,
        "total_chunks": total_chunks,
        "reused_chunks": reused_chunks,
        "new_chunks": total_chunks - reused_chunks,
    }
