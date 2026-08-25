"""Tests for POST /repositories/{id}/index error handling.

Verifies that a quota/rate-limit failure from the embedding pipeline surfaces
as a proper HTTP 429 with a clear detail message (and Retry-After header when
known) instead of an unhandled exception -- which is what previously caused
the frontend to show a generic "Cannot reach backend" error.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.services.embeddings.base import EmbeddingRateLimitError


def _make_db_with_repo(repo) -> MagicMock:
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = repo
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_index_endpoint_returns_429_on_embedding_quota_exhaustion(monkeypatch):
    from app.api.v1.repositories import index_repository_endpoint
    from app.db.models.repository import Repository

    repo = Repository(id=1, name="demo", local_path="/tmp/demo", status="pending")

    async def _raise_quota_error(self, db):
        raise EmbeddingRateLimitError(
            "Gemini embedding quota exceeded for model 'gemini-embedding-2' after 5 attempt(s).",
            retry_after=30.0,
        )

    monkeypatch.setattr(
        "app.api.v1.repositories.RepositoryIndexer.index_repository", _raise_quota_error
    )

    db = _make_db_with_repo(repo)

    with pytest.raises(HTTPException) as exc_info:
        await index_repository_endpoint(1, db)

    assert exc_info.value.status_code == 429
    assert "quota" in exc_info.value.detail.lower() or "rate" in exc_info.value.detail.lower()
    assert "already embedded" in exc_info.value.detail.lower() or "reused" in exc_info.value.detail.lower()
    assert exc_info.value.headers == {"Retry-After": "30"}
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_endpoint_429_omits_retry_after_header_when_unknown(monkeypatch):
    from app.api.v1.repositories import index_repository_endpoint
    from app.db.models.repository import Repository

    repo = Repository(id=2, name="demo2", local_path="/tmp/demo2", status="pending")

    async def _raise_quota_error(self, db):
        raise EmbeddingRateLimitError("Quota exceeded, no Retry-After known.")

    monkeypatch.setattr(
        "app.api.v1.repositories.RepositoryIndexer.index_repository", _raise_quota_error
    )

    db = _make_db_with_repo(repo)

    with pytest.raises(HTTPException) as exc_info:
        await index_repository_endpoint(2, db)

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers is None
