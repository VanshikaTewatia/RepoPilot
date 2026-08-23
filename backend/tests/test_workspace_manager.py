"""Focused tests for Stage 1 workspace isolation."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import git
import pytest

from app.core.config import settings
from app.db.models.repository import Repository
from app.services.git_service import GitService
from app.services.workspace_manager import WorkspaceManager

EXCLUDED_JUNK_DIRS = {"__pycache__", "node_modules", ".venv", ".pytest_cache"}


def _make_source_repo(root: Path) -> Path:
    """Build a fake source repository with real files plus junk directories."""
    source = root / "source_repo"
    (source / "src").mkdir(parents=True)
    (source / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "README.md").write_text("demo repo\n", encoding="utf-8")

    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00\x01")
    (source / "node_modules" / "left-pad").mkdir(parents=True)
    (source / "node_modules" / "left-pad" / "index.js").write_text("x\n", encoding="utf-8")
    (source / ".venv" / "lib").mkdir(parents=True)
    (source / ".venv" / "lib" / "site.py").write_text("x\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    return source


def _snapshot(root: Path) -> dict:
    """Map of relative path -> content for every file under root."""
    snapshot = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            snapshot[str(full.relative_to(root))] = full.read_bytes()
    return snapshot


# -------------------------------------------------------------------------
# 1. Workspace contains copied repository files
# -------------------------------------------------------------------------
def test_create_workspace_copies_repository_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = _make_source_repo(root)
        manager = WorkspaceManager(root_dir=root / "workspaces")

        workspace = manager.create_workspace(1, source)

        assert workspace.is_dir()
        assert workspace.parent == (root / "workspaces").resolve()
        assert workspace.name.startswith("task_1_")

        # Real repository files copied with identical content
        assert (workspace / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert (workspace / "README.md").read_text(encoding="utf-8") == "demo repo\n"

        # Excluded junk directories are not copied anywhere in the tree
        for dirpath, dirnames, _filenames in os.walk(workspace):
            assert not (set(dirnames) & EXCLUDED_JUNK_DIRS), f"excluded dir copied under {dirpath}"

        # The source's fake .git was not copied; a fresh baseline .git exists
        assert (workspace / ".git").is_dir()
        assert (workspace / ".git" / "objects").is_dir()
        assert not (workspace / ".git" / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"


# -------------------------------------------------------------------------
# 2. Modifying the workspace does not modify the source
# -------------------------------------------------------------------------
def test_modifying_workspace_does_not_modify_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = _make_source_repo(root)
        manager = WorkspaceManager(root_dir=root / "workspaces")

        before = _snapshot(source)
        workspace = manager.create_workspace(2, source)

        # Mutate the workspace aggressively
        (workspace / "src" / "app.py").write_text("VALUE = 999\n", encoding="utf-8")
        (workspace / "new_file.py").write_text("brand new\n", encoding="utf-8")
        (workspace / "README.md").unlink()

        after = _snapshot(source)
        assert after == before
        assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert not (source / "new_file.py").exists()


# -------------------------------------------------------------------------
# 3. Baseline Git repository works (single baseline commit + diffable)
# -------------------------------------------------------------------------
def test_baseline_git_repo_produces_diff():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = _make_source_repo(root)
        manager = WorkspaceManager(root_dir=root / "workspaces")

        workspace = manager.create_workspace(3, source)

        with git.Repo(workspace) as gr:
            commits = list(gr.iter_commits())
            assert len(commits) == 1
            assert "baseline" in commits[0].message.lower()
            tracked = gr.git.ls_files().splitlines()
            assert "src/app.py" in tracked
            assert "README.md" in tracked

        # Clean working tree right after creation
        assert GitService.get_changed_files(workspace) == []

        # Agent-style edit produces a clean unified diff against the baseline
        (workspace / "src" / "app.py").write_text("VALUE = 42\n", encoding="utf-8")

        diff = GitService.get_workspace_diff(workspace)
        assert "-VALUE = 1" in diff
        assert "+VALUE = 42" in diff
        assert GitService.get_changed_files(workspace) == ["src/app.py"]


# -------------------------------------------------------------------------
# 4. Cleanup works
# -------------------------------------------------------------------------
def test_cleanup_workspace_deletes_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = _make_source_repo(root)
        manager = WorkspaceManager(root_dir=root / "workspaces")

        workspace = manager.create_workspace(4, source)
        assert workspace.exists()

        assert manager.cleanup_workspace(workspace) is True
        assert not workspace.exists()
        # Second cleanup of a missing path is a safe no-op returning False
        assert manager.cleanup_workspace(workspace) is False

        # Refuses to delete anything outside the workspace root
        outside = root / "outside"
        outside.mkdir()
        with pytest.raises(ValueError):
            manager.cleanup_workspace(outside)


# -------------------------------------------------------------------------
# 5. Invalid source fails safely
# -------------------------------------------------------------------------
def test_invalid_source_fails_safely():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        workspaces = root / "workspaces"
        manager = WorkspaceManager(root_dir=workspaces)

        # Nonexistent path
        with pytest.raises(ValueError):
            manager.create_workspace(5, root / "does_not_exist")

        # A file, not a directory
        file_path = root / "just_a_file.txt"
        file_path.write_text("data\n", encoding="utf-8")
        with pytest.raises(ValueError):
            manager.create_workspace(6, file_path)

        # No partial workspace directories left behind
        if workspaces.exists():
            assert list(workspaces.iterdir()) == []


# -------------------------------------------------------------------------
# 6. Agent execution receives the isolated workspace
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_execution_receives_isolated_workspace():
    from app.api.v1.agent import CreateTaskRequest, create_and_run_task

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = _make_source_repo(root)
        repo_obj = Repository(id=1, name="source_repo", local_path=str(source))

        captured = {}

        async def fake_ainvoke(state):
            """Stand-in for the LangGraph loop: records state and edits 'the repo'."""
            captured.update(state)
            ws = Path(state["workspace_dir"])
            (ws / "src" / "app.py").write_text("VALUE = 42\n", encoding="utf-8")
            return {
                **state,
                "is_verified": True,
                "attempt_count": 1,
                "test_results": {"output": "1 passed"},
                "status": "verified",
            }

        exec_result = MagicMock()
        exec_result.scalar_one_or_none.return_value = repo_obj
        db = MagicMock()
        db.execute = AsyncMock(return_value=exec_result)
        db.add = MagicMock()
        db.commit = AsyncMock()

        async def fake_refresh(obj, *args, **kwargs):
            obj.id = 7

        db.refresh = AsyncMock(side_effect=fake_refresh)

        payload = CreateTaskRequest(
            repository_id=1,
            title="Fix VALUE",
            description="Set VALUE to 42",
        )

        with patch.object(settings, "workspace_dir", root / "workspaces"):
            with patch("app.api.v1.agent.agent_app", SimpleNamespace(ainvoke=fake_ainvoke)):
                result = await create_and_run_task(payload, db)

        # The agent received an isolated copy, not the live repository path
        assert captured["workspace_dir"] != str(source)
        workspace = Path(captured["workspace_dir"])
        assert workspace.is_dir()
        assert workspace.parent == (root / "workspaces").resolve()

        # The live repository was untouched by the agent's edit
        assert (source / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"

        # Task is pending review with the workspace path stored and diff captured
        assert result.status == "human_approval_required"
        assert result.workspace_path == str(workspace)
        assert result.patch_content is not None
        assert "+VALUE = 42" in result.patch_content


# -------------------------------------------------------------------------
# 7. Workspace .gitignore is created and committed in the baseline
# -------------------------------------------------------------------------
def test_workspace_gitignore_created_and_committed():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = _make_source_repo(root)
        manager = WorkspaceManager(root_dir=root / "workspaces")

        workspace = manager.create_workspace(7, source)

        gitignore = workspace / ".gitignore"
        assert gitignore.is_file()
        content = gitignore.read_text(encoding="utf-8")
        for pattern in ("__pycache__/", "*.pyc", ".pytest_cache/", ".coverage"):
            assert pattern in content

        # The .gitignore is tracked in the single baseline commit, so it never
        # shows up as a change in later task diffs.
        with git.Repo(workspace) as gr:
            commits = list(gr.iter_commits())
            assert len(commits) == 1
            assert ".gitignore" in gr.git.ls_files().splitlines()

        assert GitService.get_changed_files(workspace) == []


# -------------------------------------------------------------------------
# 8. Runtime artifacts from sandbox test runs cannot enter the review diff
# -------------------------------------------------------------------------
def test_runtime_artifacts_cannot_enter_review_diff():
    """Regression test for Task #6: pytest-generated bytecode/caches written to
    the workspace after creation must not leak into staged changes or diffs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = _make_source_repo(root)
        manager = WorkspaceManager(root_dir=root / "workspaces")

        workspace = manager.create_workspace(8, source)

        # Simulate a pytest run inside the workspace (local subprocess fallback)
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__" / "conftest.cpython-314-pytest-9.1.1.pyc").write_bytes(b"\x00\x01")
        (workspace / "src" / "__pycache__").mkdir()
        (workspace / "src" / "__pycache__" / "app.cpython-314.pyc").write_bytes(b"\x00\x02")
        (workspace / ".pytest_cache" / "v").mkdir(parents=True)
        (workspace / ".pytest_cache" / "v" / "lastfailed").write_text("{}\n", encoding="utf-8")
        (workspace / ".coverage").write_bytes(b"\x01\x02")

        # Legitimate agent fix alongside the artifacts
        (workspace / "src" / "app.py").write_text("VALUE = 42\n", encoding="utf-8")

        assert GitService.stage_all_changes(workspace) is True

        changed = GitService.get_changed_files(workspace)
        assert changed == ["src/app.py"]

        diff = GitService.get_workspace_diff(workspace)
        for banned in ("__pycache__", ".pyc", ".pytest_cache", ".coverage"):
            assert banned not in diff, f"artifact '{banned}' leaked into diff"
        assert "-VALUE = 1" in diff
        assert "+VALUE = 42" in diff
