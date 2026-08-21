"""Focused tests for Stage 2 human approval/rejection of isolated agent tasks."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import git
import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.db.models.repository import Repository
from app.db.models.task import Task
from app.services.git_service import GitService
from app.services.workspace_manager import WorkspaceManager


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _init_source_repo(root: Path) -> Path:
    """Build a real git repository (the 'original' repo the agent fixes)."""
    source = root / "real_repo"
    (source / "src").mkdir(parents=True)
    (source / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "README.md").write_text("readme\n", encoding="utf-8")
    with git.Repo.init(source) as gr:
        with gr.config_writer() as cw:
            cw.set_value("user", "name", "Tester")
            cw.set_value("user", "email", "tester@localhost")
        gr.git.add(A=True)
        gr.index.commit("baseline")
    return source


def _build_review_state(root: Path):
    """Create original repo + isolated workspace with agent-style edits.

    Returns (source, workspace, task, repo_obj, patch, changed_files).
    """
    source = _init_source_repo(root)
    manager = WorkspaceManager(root_dir=root / "workspaces")
    workspace = manager.create_workspace(1, source)

    # Simulate agent edits inside the isolated workspace: modify + new file
    (workspace / "src" / "app.py").write_text("VALUE = 42\n", encoding="utf-8")
    (workspace / "src" / "extra.py").write_text("NEW = True\n", encoding="utf-8")

    GitService.stage_all_changes(workspace)
    patch = GitService.get_workspace_diff(workspace)
    changed = GitService.get_changed_files(workspace)

    task = Task(
        id=1,
        repository_id=1,
        title="Fix VALUE",
        description="Set VALUE to 42",
        status="human_approval_required",
        patch_content=patch,
        changed_files=changed,
        workspace_path=str(workspace),
    )
    repo_obj = Repository(id=1, name="real_repo", local_path=str(source))
    assert "+VALUE = 42" in patch
    assert "+NEW = True" in patch  # new files are part of the applicable diff
    return source, workspace, task, repo_obj, patch, changed


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


def _snapshot(root: Path) -> dict:
    """Map of relative path -> bytes for every file under root."""
    snapshot = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            full = Path(dirpath) / name
            snapshot[str(full.relative_to(root))] = full.read_bytes()
    return snapshot


# -------------------------------------------------------------------------
# Approve
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_approve_applies_patch_and_cleans_up_workspace():
    """human_approval_required -> approve applies diff to real repo and cleans up."""
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)
        db = _make_db(task=task, repo=repo_obj)

        with patch.object(settings, "workspace_dir", root / "workspaces"):
            result = await approve_task_fix(1, db)

        assert result["status"] == "approved"
        assert task.status == "approved"

        # Intended diff applied to the ORIGINAL repository
        assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 42\n"
        assert (source / "src" / "extra.py").read_text(encoding="utf-8") == "NEW = True\n"

        # Isolated workspace cleaned up and reference cleared
        assert not workspace.exists()
        assert task.workspace_path is None


@pytest.mark.asyncio
async def test_approval_conflict_leaves_task_pending():
    """Drifted original repository -> 409 conflict, task stays pending, workspace kept."""
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)

        # Unrelated/stale change lands in the real repo after review started
        drifted = "VALUE = 999\n"
        (source / "src" / "app.py").write_text(drifted, encoding="utf-8")

        db = _make_db(task=task, repo=repo_obj)
        with patch.object(settings, "workspace_dir", root / "workspaces"):
            with pytest.raises(HTTPException) as exc_info:
                await approve_task_fix(1, db)

        assert exc_info.value.status_code == 409
        assert "cannot be applied cleanly" in exc_info.value.detail

        # Task remains pending review; workspace preserved for investigation
        assert task.status == "human_approval_required"
        assert workspace.exists()
        assert task.workspace_path == str(workspace)


@pytest.mark.asyncio
async def test_stale_real_repo_changes_not_silently_overwritten():
    """A conflicting real-repo file keeps its exact bytes after a failed approval."""
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)

        before = _snapshot(source)
        (source / "src" / "app.py").write_text("VALUE = 999\n", encoding="utf-8")
        stale_bytes = (source / "src" / "app.py").read_bytes()

        db = _make_db(task=task, repo=repo_obj)
        with patch.object(settings, "workspace_dir", root / "workspaces"):
            with pytest.raises(HTTPException):
                await approve_task_fix(1, db)

        # Nothing was overwritten or partially applied
        assert (source / "src" / "app.py").read_bytes() == stale_bytes
        after = dict(before)
        after["src\\app.py"] = stale_bytes
        assert _snapshot(source) == after


@pytest.mark.asyncio
async def test_approve_wrong_status_task_is_rejected():
    """Only human_approval_required tasks can be approved."""
    from app.api.v1.agent import approve_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)
        task.status = "failed"

        db = _make_db(task=task, repo=repo_obj)
        with patch.object(settings, "workspace_dir", root / "workspaces"):
            with pytest.raises(HTTPException) as exc_info:
                await approve_task_fix(1, db)

        assert exc_info.value.status_code == 400
        assert task.status == "failed"
        assert workspace.exists()
        assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


# -------------------------------------------------------------------------
# Reject
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reject_sets_status_and_cleans_workspace():
    """human_approval_required -> reject discards workspace, marks rejected."""
    from app.api.v1.agent import reject_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)
        db = _make_db(task=task, repo=repo_obj)

        with patch.object(settings, "workspace_dir", root / "workspaces"):
            result = await reject_task_fix(1, db)

        assert result["status"] == "rejected"
        assert task.status == "rejected"
        assert not workspace.exists()
        assert task.workspace_path is None


@pytest.mark.asyncio
async def test_reject_leaves_original_repository_byte_identical():
    """Rejection must not touch a single byte of the original repository."""
    from app.api.v1.agent import reject_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)
        before = _snapshot(source)

        db = _make_db(task=task, repo=repo_obj)
        with patch.object(settings, "workspace_dir", root / "workspaces"):
            await reject_task_fix(1, db)

        assert _snapshot(source) == before
        assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.asyncio
async def test_reject_wrong_status_task_is_rejected():
    """Only human_approval_required tasks can be rejected."""
    from app.api.v1.agent import reject_task_fix

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)
        task.status = "approved"

        db = _make_db(task=task, repo=repo_obj)
        with patch.object(settings, "workspace_dir", root / "workspaces"):
            with pytest.raises(HTTPException) as exc_info:
                await reject_task_fix(1, db)

        assert exc_info.value.status_code == 400
        assert task.status == "approved"
        assert workspace.exists()


# -------------------------------------------------------------------------
# Diff endpoint persistence
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_diff_endpoint_returns_persisted_review_info():
    """/diff serves persisted patch/changed_files instead of recomputing."""
    from app.api.v1.agent import get_task_diff

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)

        # Drift the live repository AND hide the workspace: if the endpoint
        # recomputed anything it could not return the persisted review data.
        (source / "src" / "app.py").write_text("VALUE = drifted\n", encoding="utf-8")
        workspace.rename(root / "moved_away_ws")
        task.workspace_path = str(root / "moved_away_ws")

        db = _make_db(task=task, repo=repo_obj)
        response = await get_task_diff(1, db)

        assert response["diff"] == review_patch
        assert response["changed_files"] == sorted(changed_files)
        assert response["status"] == "human_approval_required"


# -------------------------------------------------------------------------
# GitService.apply_diff unit checks
# -------------------------------------------------------------------------
def test_apply_diff_rejects_conflicting_target_without_changes():
    """apply_diff refuses to apply when target state conflicts; nothing changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source, workspace, task, repo_obj, review_patch, changed_files = _build_review_state(root)

        (source / "src" / "app.py").write_text("VALUE = 999\n", encoding="utf-8")

        ok, error = GitService.apply_diff(source, review_patch)
        assert ok is False
        assert error
        assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 999\n"
        assert not (source / "src" / "extra.py").exists()


def test_apply_diff_empty_and_missing_target_fail_safely():
    """Empty diffs and missing targets fail safely without raising."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = _init_source_repo(root)

        ok, _ = GitService.apply_diff(source, "")
        assert ok is False

        ok2, err2 = GitService.apply_diff(root / "missing", "diff --git a/x b/x")
        assert ok2 is False
        assert "does not exist" in err2
