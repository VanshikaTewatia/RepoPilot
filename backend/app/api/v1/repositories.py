"""Repository management and indexing API routes."""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import SessionDep
from app.core.config import settings
from app.db.models.repository import Repository
from app.services.embeddings.base import EmbeddingRateLimitError
from app.services.github_service import GitHubError, GitHubService, parse_github_url
from app.services.indexing.indexer import RepositoryIndexer

router = APIRouter(prefix="/repositories", tags=["Repositories"])


class RepositoryCreate(BaseModel):
    name: str = Field(..., example="my-project")
    local_path: str = Field(..., example="C:/Users/Lakshay/RepoPilot")
    remote_url: str | None = None
    default_branch: str = "main"


class RepositoryCreateFromGitHub(BaseModel):
    """Register a repository by cloning it from GitHub.

    No token field here by design: GITHUB_TOKEN is read only from server-side
    settings (see app.core.config) and is never accepted from a request body.
    """

    url: str = Field(..., example="https://github.com/octocat/Hello-World")
    name: str | None = Field(default=None, example="Hello-World")
    default_branch: str | None = Field(
        default=None,
        description="Overrides the repository's actual default branch if provided.",
    )


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


@router.post("/github", response_model=RepositoryResponse, status_code=status.HTTP_201_CREATED)
async def create_repository_from_github(
    payload: RepositoryCreateFromGitHub,
    db: SessionDep,
) -> Repository:
    """Register a repository by cloning it from a validated GitHub HTTPS URL.

    Public repositories work with no server configuration. Private repositories
    require GITHUB_TOKEN to be set in the backend environment; the token is never
    read from this request, stored in the database, or echoed back in the response.
    """
    try:
        owner, repo_name = parse_github_url(payload.url)
    except GitHubError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    github = GitHubService(api_base_url=settings.github_api_base_url)
    token = settings.github_token or None

    try:
        meta = await asyncio.to_thread(github.validate_repository, owner, repo_name, token)
    except GitHubError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    workspace_root = Path(settings.workspace_dir).resolve()
    dest_dir = workspace_root / "repos" / f"{owner}_{repo_name}_{uuid.uuid4().hex[:8]}"

    try:
        cloned_path = await asyncio.to_thread(
            github.clone_repository,
            meta["clone_url"],
            dest_dir,
            workspace_root,
            token,
        )
    except GitHubError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    repo = Repository(
        name=payload.name or repo_name,
        local_path=str(cloned_path),
        remote_url=f"https://github.com/{owner}/{repo_name}",
        default_branch=payload.default_branch or meta["default_branch"],
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
    try:
        total_chunks, reused_chunks = await indexer.index_repository(db)
    except EmbeddingRateLimitError as e:
        headers = {"Retry-After": str(int(e.retry_after))} if e.retry_after else None
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Indexing was paused because the Gemini embedding API's rate/quota "
                "limit was exceeded and retries were exhausted. Wait a bit and re-run "
                f"indexing — chunks already embedded will be reused, not re-embedded. ({e})"
            ),
            headers=headers,
        ) from e

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
