"""Tests for the GitHub-backed approval flow: branch, apply, verify, commit,
push, and Pull Request creation -- layered on top of the exact same review
gate (`human_approval_required`) and workspace-isolation guarantees already
covered by test_task_review.py for local-path repositories.

No real network or Docker calls are made: GitHubService's push/PR methods
and the sandbox test runner are mocked at their call sites. GitPython
operations against the "original" repository use real local temp git repos,
matching the existing convention in test_task_review.py.
"""

import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import git
import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.db.models.repository import Repository
from app.db.models.task import Task
from app.services.git_service import GitService
from app.services.github_service import GitHubError
from app.services.workspace_manager import WorkspaceManager

PASSING_TEST_RESULT = {
    "success": True, "exit_code": 0, "output": "1 passed", "passed": 1, "failed": 0, "duration": 0.1,
}
FAILING_TEST_RESULT = {
    "success": False, "exit_code": 1, "output": "1 failed", "passed": 0, "failed": 1, "duration": 0.1,
}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _init_source_repo(root: Path) -> Tuple[Path, str]:
    """Build a real git repository (the 'original' GitHub-cloned repo)."""
    source = root / "gh_repo"
    (source / "src").mkdir(parents=True)
    (source / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    with git.Repo.init(source) as gr:
        with gr.config_writer() as cw:
            cw.set_value("user", "name", "Tester")
            cw.set_value("user", "email", "tester@localhost")
        gr.git.add(A=True)
        gr.index.commit("baseline")
        default_branch = gr.active_branch.name
    return source, default_branch


def _build_github_review_state(root: Path):
    """Create original repo + isolated workspace with agent-style edits.

    Returns (source, workspace, task, repo_obj, default_branch).
    """
    source, default_branch = _init_source_repo(root)
    manager = WorkspaceManager(root_dir=root / "workspaces")
    workspace = manager.create_workspace(1, source)

    (workspace / "src" / "app.py").write_text("VALUE = 42\n", encoding="utf-8")

    GitService.stage_all_changes(workspace)
    patch_text = GitService.get_workspace_diff(workspace)
    changed = GitService.get_changed_files(workspace)

    task = Task(
        id=1,
        repository_id=1,
        title="Fix VALUE",
        description="Set VALUE to 42",
        status="human_approval_required",
        patch_content=patch_text,
        changed_files=changed,
        workspace_path=str(workspace),
    )
    repo_obj = Repository(
        id=1,
        name="gh_repo",
        local_path=str(source),
        remote_url="https://github.com/acme/gh_repo",
        default_branch=default_branch,
    )
    assert "+VALUE = 42" in patch_text
    return source, workspace, task, repo_obj, default_branch


def _make_db(task=None, repo=None) -> MagicMock:
    """Mock async session dispatching scalar results by selected entity."""
    db = MagicMock()

    async def execute(stmt):
        res = MagicMock()
        entity = stmt.column_descriptions[0]["entity"]
        res.scalar_one_or_none.return_value = task if entity is Task else repo
        return res

    db.execute = AsyncMock(side_effect=execute)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


# -------------------------------------------------------------------------
# Approve: GitHub-backed repository
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_approve_github_repo_creates_branch_commits_pushes_and_opens_pr():
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, default_branch = _build_github_review_state(root)
        db = _make_db(task=task, repo=repo_obj)

        with patch.object(settings, "workspace_dir", root / "workspaces"), \
             patch.object(settings, "github_token", "fake-token"), \
             patch("app.api.v1.agent.tools.run_tests", return_value=PASSING_TEST_RESULT) as mock_tests, \
             patch("app.api.v1.agent.GitHubService.push_branch") as mock_push, \
             patch(
                 "app.api.v1.agent.GitHubService.create_pull_request",
                 return_value={"url": "https://github.com/acme/gh_repo/pull/7", "number": 7},
             ) as mock_pr:
            result = await approve_task_fix(1, db)

        assert result["status"] == "approved"
        assert result["pr_url"] == "https://github.com/acme/gh_repo/pull/7"
        assert task.status == "approved"
        assert task.pr_url == "https://github.com/acme/gh_repo/pull/7"

        mock_push.assert_called_once()
        assert mock_push.call_args[0][1] == "repopilot/task-1"

        mock_pr.assert_called_once()
        pr_args = mock_pr.call_args[0]
        assert pr_args[2] == "repopilot/task-1"  # head branch
        assert pr_args[3] == default_branch  # base branch

        mock_tests.assert_called_once()  # final verification ran before push

        # Original repo ends up back on default_branch; the fix lives on the task branch
        with git.Repo(source) as gr:
            assert gr.active_branch.name == default_branch
            assert "repopilot/task-1" in [h.name for h in gr.heads]
            assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
            gr.git.checkout("repopilot/task-1")
            assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 42\n"
            gr.git.checkout(default_branch)

        assert not workspace.exists()
        assert task.workspace_path is None


@pytest.mark.asyncio
async def test_approve_github_repo_final_verification_failure_rolls_back():
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, default_branch = _build_github_review_state(root)
        db = _make_db(task=task, repo=repo_obj)

        with patch.object(settings, "workspace_dir", root / "workspaces"), \
             patch.object(settings, "github_token", "fake-token"), \
             patch("app.api.v1.agent.tools.run_tests", return_value=FAILING_TEST_RESULT), \
             patch("app.api.v1.agent.GitHubService.push_branch") as mock_push:
            with pytest.raises(HTTPException) as exc_info:
                await approve_task_fix(1, db)

        assert exc_info.value.status_code == 502
        assert task.status == "approval_failed"
        mock_push.assert_not_called()

        with git.Repo(source) as gr:
            assert gr.active_branch.name == default_branch
            assert "repopilot/task-1" not in [h.name for h in gr.heads]
            assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"

        # Task stays reviewable/retryable: workspace and stored patch preserved
        assert workspace.exists()
        assert task.workspace_path == str(workspace)
        assert task.patch_content


@pytest.mark.asyncio
async def test_approve_github_repo_push_failure_rolls_back_and_scrubs_token():
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, default_branch = _build_github_review_state(root)
        db = _make_db(task=task, repo=repo_obj)

        with patch.object(settings, "workspace_dir", root / "workspaces"), \
             patch.object(settings, "github_token", "fake-token"), \
             patch("app.api.v1.agent.tools.run_tests", return_value=PASSING_TEST_RESULT), \
             patch(
                 "app.api.v1.agent.GitHubService.push_branch",
                 side_effect=GitHubError("simulated push failure"),
             ):
            with pytest.raises(HTTPException) as exc_info:
                await approve_task_fix(1, db)

        assert exc_info.value.status_code == 502
        assert "simulated push failure" in exc_info.value.detail
        assert "fake-token" not in exc_info.value.detail
        assert task.status == "approval_failed"
        assert "simulated push failure" in task.test_output

        with git.Repo(source) as gr:
            assert gr.active_branch.name == default_branch
            assert "repopilot/task-1" not in [h.name for h in gr.heads]

        assert workspace.exists()


@pytest.mark.asyncio
async def test_approve_github_repo_allows_retry_from_approval_failed_status():
    """approval_failed (not just human_approval_required) is an accepted precondition."""
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, default_branch = _build_github_review_state(root)
        task.status = "approval_failed"
        db = _make_db(task=task, repo=repo_obj)

        with patch.object(settings, "workspace_dir", root / "workspaces"), \
             patch.object(settings, "github_token", "fake-token"), \
             patch("app.api.v1.agent.tools.run_tests", return_value=PASSING_TEST_RESULT), \
             patch("app.api.v1.agent.GitHubService.push_branch"), \
             patch(
                 "app.api.v1.agent.GitHubService.create_pull_request",
                 return_value={"url": "https://github.com/acme/gh_repo/pull/9", "number": 9},
             ):
            result = await approve_task_fix(1, db)

        assert result["status"] == "approved"
        assert task.status == "approved"


# -------------------------------------------------------------------------
# Approve: local-path repository is unaffected
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_approve_local_repo_still_uses_direct_apply_no_branch_or_pr():
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, default_branch = _init_source_repo(root)
        manager = WorkspaceManager(root_dir=root / "workspaces")
        workspace = manager.create_workspace(1, source)
        (workspace / "src" / "app.py").write_text("VALUE = 42\n", encoding="utf-8")
        GitService.stage_all_changes(workspace)
        patch_text = GitService.get_workspace_diff(workspace)

        task = Task(
            id=1,
            repository_id=1,
            title="Fix VALUE",
            description="Set VALUE to 42",
            status="human_approval_required",
            patch_content=patch_text,
            changed_files=GitService.get_changed_files(workspace),
            workspace_path=str(workspace),
        )
        repo_obj = Repository(id=1, name="local_repo", local_path=str(source), remote_url=None)
        db = _make_db(task=task, repo=repo_obj)

        with patch.object(settings, "workspace_dir", root / "workspaces"), \
             patch("app.api.v1.agent.GitHubService.push_branch") as mock_push, \
             patch("app.api.v1.agent.GitHubService.create_pull_request") as mock_pr:
            result = await approve_task_fix(1, db)

        assert result["status"] == "approved"
        assert result.get("pr_url") is None
        assert task.pr_url is None
        mock_push.assert_not_called()
        mock_pr.assert_not_called()
        assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 42\n"


# -------------------------------------------------------------------------
# Reject: unaffected by the GitHub flow
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reject_github_repo_never_branches_or_pushes():
    from app.api.v1.agent import reject_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, default_branch = _build_github_review_state(root)
        db = _make_db(task=task, repo=repo_obj)

        with patch.object(settings, "workspace_dir", root / "workspaces"), \
             patch("app.api.v1.agent.GitHubService.push_branch") as mock_push:
            result = await reject_task_fix(1, db)

        assert result["status"] == "rejected"
        assert task.status == "rejected"
        mock_push.assert_not_called()

        with git.Repo(source) as gr:
            assert "repopilot/task-1" not in [h.name for h in gr.heads]
            assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"

        assert not workspace.exists()
