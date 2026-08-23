"""Tests for POST /repositories/github: URL validation, clone confinement,
token non-disclosure, and the untouched local-path registration flow.

GitHubService.validate_repository / clone_repository are mocked -- no real
network calls are made.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.services.github_service import GitHubError


def _make_db() -> MagicMock:
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_create_repository_from_github_success():
    from app.api.v1.repositories import RepositoryCreateFromGitHub, create_repository_from_github

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir) / "workspaces"
        db = _make_db()

        with patch.object(settings, "workspace_dir", workspace_root), \
             patch.object(settings, "github_token", ""), \
             patch(
                 "app.api.v1.repositories.GitHubService.validate_repository",
                 return_value={
                     "clone_url": "https://github.com/octocat/Hello-World.git",
                     "default_branch": "main",
                     "private": False,
                     "full_name": "octocat/Hello-World",
                 },
             ) as mock_validate, \
             patch("app.api.v1.repositories.GitHubService.clone_repository") as mock_clone:
            mock_clone.side_effect = lambda clone_url, dest_dir, workspace_root, token: Path(dest_dir)

            payload = RepositoryCreateFromGitHub(url="https://github.com/octocat/Hello-World")
            repo = await create_repository_from_github(payload, db)

        # No token configured -> validate_repository called with None (public repo works without one).
        mock_validate.assert_called_once_with("octocat", "Hello-World", None)
        assert repo.name == "Hello-World"
        assert repo.remote_url == "https://github.com/octocat/Hello-World"
        assert repo.default_branch == "main"
        assert repo.status == "pending"
        assert str(workspace_root.resolve()) in repo.local_path
        db.add.assert_called_once()
        db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_create_repository_from_github_custom_name_and_branch_override():
    from app.api.v1.repositories import RepositoryCreateFromGitHub, create_repository_from_github

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir) / "workspaces"
        db = _make_db()

        with patch.object(settings, "workspace_dir", workspace_root), \
             patch.object(settings, "github_token", ""), \
             patch(
                 "app.api.v1.repositories.GitHubService.validate_repository",
                 return_value={"clone_url": "https://github.com/a/b.git", "default_branch": "main", "private": False},
             ), \
             patch("app.api.v1.repositories.GitHubService.clone_repository") as mock_clone:
            mock_clone.side_effect = lambda clone_url, dest_dir, workspace_root, token: Path(dest_dir)

            payload = RepositoryCreateFromGitHub(
                url="https://github.com/a/b", name="My Repo", default_branch="develop"
            )
            repo = await create_repository_from_github(payload, db)

        assert repo.name == "My Repo"
        assert repo.default_branch == "develop"  # explicit override wins over GitHub's reported default


@pytest.mark.asyncio
async def test_create_repository_from_github_invalid_url_returns_400():
    from app.api.v1.repositories import RepositoryCreateFromGitHub, create_repository_from_github

    db = _make_db()
    payload = RepositoryCreateFromGitHub(url="git@github.com:octocat/Hello-World.git")

    with pytest.raises(HTTPException) as exc_info:
        await create_repository_from_github(payload, db)

    assert exc_info.value.status_code == 400
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_create_repository_from_github_private_without_token_returns_400():
    from app.api.v1.repositories import RepositoryCreateFromGitHub, create_repository_from_github

    db = _make_db()
    with patch.object(settings, "github_token", ""), \
         patch(
             "app.api.v1.repositories.GitHubService.validate_repository",
             side_effect=GitHubError(
                 "Repository 'acme/secret' was not found. If it is private, configure GITHUB_TOKEN."
             ),
         ):
        payload = RepositoryCreateFromGitHub(url="https://github.com/acme/secret")
        with pytest.raises(HTTPException) as exc_info:
            await create_repository_from_github(payload, db)

    assert exc_info.value.status_code == 400
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_create_repository_from_github_clone_destination_confined_under_workspace_root():
    from app.api.v1.repositories import RepositoryCreateFromGitHub, create_repository_from_github

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir) / "workspaces"
        db = _make_db()

        with patch.object(settings, "workspace_dir", workspace_root), \
             patch.object(settings, "github_token", ""), \
             patch(
                 "app.api.v1.repositories.GitHubService.validate_repository",
                 return_value={"clone_url": "https://github.com/a/b.git", "default_branch": "main", "private": False},
             ), \
             patch("app.api.v1.repositories.GitHubService.clone_repository") as mock_clone:
            mock_clone.side_effect = lambda clone_url, dest_dir, workspace_root, token: Path(dest_dir)

            payload = RepositoryCreateFromGitHub(url="https://github.com/a/b")
            await create_repository_from_github(payload, db)

        dest_dir_arg = Path(mock_clone.call_args[0][1]).resolve()
        confinement_root_arg = Path(mock_clone.call_args[0][2]).resolve()

        assert confinement_root_arg == workspace_root.resolve()
        assert workspace_root.resolve() in dest_dir_arg.parents


@pytest.mark.asyncio
async def test_create_repository_from_github_never_stores_or_forwards_token():
    """The token is used server-side to reach a private repo, but is never
    persisted on the Repository model or present in the returned object."""
    from app.api.v1.repositories import (
        RepositoryCreateFromGitHub,
        RepositoryResponse,
        create_repository_from_github,
    )
    from app.db.models.repository import Repository

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir) / "workspaces"
        db = _make_db()

        with patch.object(settings, "workspace_dir", workspace_root), \
             patch.object(settings, "github_token", "server-secret-token"), \
             patch(
                 "app.api.v1.repositories.GitHubService.validate_repository",
                 return_value={"clone_url": "https://github.com/a/b.git", "default_branch": "main", "private": True},
             ) as mock_validate, \
             patch("app.api.v1.repositories.GitHubService.clone_repository") as mock_clone:
            mock_clone.side_effect = lambda clone_url, dest_dir, workspace_root, token: Path(dest_dir)

            payload = RepositoryCreateFromGitHub(url="https://github.com/a/b")
            repo = await create_repository_from_github(payload, db)

        # Token is used server-side to authenticate the validate/clone calls...
        mock_validate.assert_called_once_with("a", "b", "server-secret-token")
        # ...but there is no column or response field that could ever carry it.
        assert not any("token" in c.name.lower() for c in Repository.__table__.columns)
        assert "token" not in RepositoryResponse.model_fields
        assert "server-secret-token" not in repr(repo.__dict__)


# -------------------------------------------------------------------------
# Existing local-path registration flow is unaffected
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_local_path_registration_flow_unchanged():
    from app.api.v1.repositories import RepositoryCreate, create_repository

    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db()
        payload = RepositoryCreate(name="local-project", local_path=tmpdir)
        repo = await create_repository(payload, db)

    assert repo.name == "local-project"
    assert repo.local_path == str(Path(tmpdir).resolve())
    assert repo.remote_url is None
    db.add.assert_called_once()
    db.commit.assert_awaited()
