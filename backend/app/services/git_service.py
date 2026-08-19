"""Git workspace and diff generation service."""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import git

from app.core.logging import logger


class GitService:
    """Manages repository status, workspace snapshots, and unified diff inspection."""

    @staticmethod
    def get_workspace_diff(workspace_dir: Path | str) -> str:
        """Generate unified git diff for all unstaged and staged changes in workspace."""
        workspace = Path(workspace_dir).resolve()
        if not (workspace / ".git").is_dir():
            # If not a git repo, return message or empty diff
            return ""

        try:
            repo = git.Repo(workspace)
            # Full diff against HEAD
            diff_text = repo.git.diff("HEAD")
            if not diff_text:
                # Check untracked files
                untracked = repo.untracked_files
                if untracked:
                    diff_text = f"# New untracked files:\n" + "\n".join(f"+ {f}" for f in untracked)
            return diff_text
        except Exception as e:
            logger.warning(f"Failed to generate git diff: {e}")
            try:
                # Subprocess fallback
                res = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                )
                return res.stdout
            except Exception:
                return ""

    @staticmethod
    def get_changed_files(workspace_dir: Path | str) -> List[str]:
        """List files modified, added, or deleted in the workspace."""
        workspace = Path(workspace_dir).resolve()
        if not (workspace / ".git").is_dir():
            return []

        try:
            repo = git.Repo(workspace)
            changed = [item.a_path for item in repo.index.diff(None)]
            changed.extend(repo.untracked_files)
            return list(set(changed))
        except Exception:
            return []
