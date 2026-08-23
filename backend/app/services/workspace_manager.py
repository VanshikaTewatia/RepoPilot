"""Isolated per-task workspace management.

Creates a private copy of a repository for each agent task so the live
repository is never modified during agent execution. Each workspace copy is
initialized as its own Git repository with a baseline commit, enabling
GitService to produce clean unified diffs of the agent's changes.
"""

import fnmatch
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Set, Union

import git

from app.core.config import settings
from app.core.logging import logger

# Directory names never copied into an isolated workspace (build artifacts,
# dependency trees, VCS metadata, caches, virtual environments).
EXCLUDED_DIR_NAMES: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "ENV",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".eggs",
    ".idea",
    ".vscode",
    ".cache",
    "htmlcov",
    "dist",
    "build",
}

# File name / glob patterns excluded from the copy.
EXCLUDED_FILE_PATTERNS: Set[str] = {
    "*.pyc",
    "*.pyo",
    "*.log",
    ".env",
    ".DS_Store",
}

# Patterns written into every workspace .gitignore before its baseline commit so
# runtime-generated Python/test artifacts (e.g. bytecode written by pytest inside
# the sandbox) can never be staged into the task's persisted review diff.
WORKSPACE_GITIGNORE_PATTERNS: List[str] = [
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".coverage",
]


def _make_copy_ignore_filter() -> Callable[[str, List[str]], Set[str]]:
    """Build a shutil.copytree ignore callable based on exclusion rules."""

    def ignore_func(directory: str, entries: List[str]) -> Set[str]:
        excluded: Set[str] = set()
        for entry in entries:
            if entry in EXCLUDED_DIR_NAMES:
                excluded.add(entry)
                continue
            if any(fnmatch.fnmatch(entry, pattern) for pattern in EXCLUDED_FILE_PATTERNS):
                excluded.add(entry)
        return excluded

    return ignore_func


class WorkspaceManager:
    """Creates, tracks, and cleans up isolated per-task workspace copies."""

    def __init__(self, root_dir: Optional[Union[Path, str]] = None):
        """Initialize manager with a workspace root (defaults to configured workspace_dir)."""
        self.root_dir = Path(root_dir) if root_dir else Path(settings.workspace_dir)

    # ---------------------------------------------------------------------
    # Creation
    # ---------------------------------------------------------------------
    def create_workspace(self, task_id: Union[int, str], source_path: Union[Path, str]) -> Path:
        """Create an isolated workspace copy of the source repository for a task.

        Copies repository contents (excluding VCS metadata, caches, virtual
        environments, and build artifacts), initializes a fresh Git repository
        in the copy, and creates a baseline commit so GitService can produce a
        diff after the agent edits files.

        Raises ValueError for invalid or nonexistent source repositories.
        """
        source = Path(source_path)
        if not source.exists():
            raise ValueError(f"Source repository does not exist: {source}")
        if not source.is_dir():
            raise ValueError(f"Source repository path is not a directory: {source}")

        source = source.resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

        workspace = (self.root_dir / f"task_{task_id}_{uuid.uuid4().hex[:8]}").resolve()
        if workspace.exists():
            raise ValueError(f"Workspace path already exists: {workspace}")

        logger.info(f"Creating isolated workspace for task {task_id}: {workspace}")
        try:
            shutil.copytree(
                source,
                workspace,
                ignore=_make_copy_ignore_filter(),
                dirs_exist_ok=False,
            )
            self._init_baseline_git_repo(workspace, task_id)
        except Exception as e:
            # Never leave a partially-copied workspace behind.
            self._force_remove(workspace)
            logger.error(f"Failed to create workspace for task {task_id}: {e}")
            raise

        logger.info(f"Isolated workspace ready for task {task_id}: {workspace}")
        return workspace

    @staticmethod
    def _init_baseline_git_repo(workspace: Path, task_id: Union[int, str]) -> None:
        """Initialize a Git repository in the workspace with a baseline commit."""
        # The .gitignore is written before the baseline commit so it is itself
        # tracked and keeps runtime artifacts out of every later diff.
        gitignore_path = workspace / ".gitignore"
        gitignore_path.write_text(
            "\n".join(WORKSPACE_GITIGNORE_PATTERNS) + "\n", encoding="utf-8"
        )

        repo = git.Repo.init(workspace)
        try:
            # Local identity so commits work without global git config.
            with repo.config_writer() as writer:
                writer.set_value("user", "name", "RepoPilot")
                writer.set_value("user", "email", "repopilot@localhost")

            repo.git.add(A=True)
            repo.index.commit(f"RepoPilot baseline commit for task {task_id}")
        finally:
            repo.close()

    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    def cleanup_workspace(self, workspace_path: Union[Path, str]) -> bool:
        """Delete an isolated workspace directory.

        Returns True if the workspace was deleted, False if it did not exist.
        Refuses to delete any path outside the configured workspace root.
        """
        workspace = Path(workspace_path).resolve()
        root = self.root_dir.resolve()

        if workspace != root and root not in workspace.parents:
            raise ValueError(
                f"Refusing to delete '{workspace}': outside workspace root '{root}'"
            )

        if not workspace.exists():
            return False

        logger.info(f"Cleaning up workspace: {workspace}")
        _rmtree_force(workspace)
        return True

    def list_workspaces(self) -> List[Path]:
        """List existing task workspace directories under the workspace root."""
        if not self.root_dir.exists():
            return []
        return sorted(p for p in self.root_dir.iterdir() if p.is_dir())

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _force_remove(path: Path) -> None:
        """Best-effort recursive delete that never raises."""
        try:
            if path.exists():
                _rmtree_force(path)
        except Exception as e:
            logger.warning(f"Could not remove path '{path}': {e}")


def _rmtree_force(path: Path) -> None:
    """Recursively delete a directory tree, clearing read-only attributes.

    Git writes loose objects read-only, which makes plain shutil.rmtree fail
    with PermissionError (WinError 5) on Windows.
    """
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                file_path = Path(dirpath) / name
                file_path.chmod(stat.S_IWRITE)
            except OSError:
                pass
    shutil.rmtree(path)
