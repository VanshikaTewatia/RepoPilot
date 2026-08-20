"""Git workspace and diff generation service."""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import git

from app.core.logging import logger


class GitService:
    """Manages repository status, workspace snapshots, and unified diff inspection."""

    @staticmethod
    def _get_repo_and_relative_path(workspace_dir: Path | str) -> Optional[Tuple[git.Repo, str]]:
        """Resolve Git repository and workspace path relative to repository root."""
        workspace = Path(workspace_dir).resolve()
        if not workspace.is_dir():
            return None
        try:
            repo = git.Repo(workspace, search_parent_directories=True)
            repo_root = Path(repo.working_tree_dir).resolve()
            rel_path = workspace.relative_to(repo_root)
            rel_path_str = str(rel_path).replace("\\", "/") if str(rel_path) != "." else ""
            return repo, rel_path_str
        except Exception:
            return None

    @staticmethod
    def _close_repo(repo: Optional[git.Repo]) -> None:
        """Close a GitPython repo, releasing any open file handles (Windows-safe)."""
        if repo is None:
            return
        try:
            repo.close()
        except Exception:
            pass

    @staticmethod
    def get_workspace_diff(workspace_dir: Path | str) -> str:
        """Generate unified git diff for all unstaged and staged changes in workspace."""
        workspace = Path(workspace_dir).resolve()
        res = GitService._get_repo_and_relative_path(workspace_dir)
        if not res:
            return ""

        repo, rel_path_str = res
        try:
            if rel_path_str:
                diff_text = repo.git.diff("HEAD", "--", rel_path_str)
            else:
                diff_text = repo.git.diff("HEAD")

            if not diff_text:
                # Check untracked files scoped to workspace
                untracked = repo.untracked_files
                scoped_untracked = []
                for f in untracked:
                    f_clean = f.replace("\\", "/")
                    if not rel_path_str or f_clean.startswith(rel_path_str.rstrip("/") + "/"):
                        rel_to_ws = f_clean[len(rel_path_str.rstrip("/")) + 1 :] if rel_path_str else f_clean
                        scoped_untracked.append(rel_to_ws)

                if scoped_untracked:
                    diff_text = f"# New untracked files:\n" + "\n".join(f"+ {f}" for f in scoped_untracked)

            return diff_text
        except Exception as e:
            logger.warning(f"Failed to generate git diff via GitPython: {e}")
            try:
                # Subprocess fallback scoped to workspace; paths are repo-root-relative,
                # matching the GitPython output above
                res_proc = subprocess.run(
                    ["git", "diff", "HEAD", "--", "."],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                )
                return res_proc.stdout
            except Exception:
                return ""
        finally:
            GitService._close_repo(repo)

    @staticmethod
    def get_changed_files(workspace_dir: Path | str) -> List[str]:
        """List files modified, added, or deleted in the workspace, relative to workspace."""
        res = GitService._get_repo_and_relative_path(workspace_dir)
        if not res:
            return []

        repo, rel_path_str = res
        prefix = (rel_path_str.rstrip("/") + "/") if rel_path_str else ""

        try:
            changed: List[str] = []
            # Check unstaged changes
            for item in repo.index.diff(None):
                path = (item.a_path or item.b_path or "").replace("\\", "/")
                if not prefix or path.startswith(prefix):
                    rel = path[len(prefix):] if prefix else path
                    changed.append(rel)

            # Check staged changes
            for item in repo.index.diff("HEAD"):
                path = (item.a_path or item.b_path or "").replace("\\", "/")
                if not prefix or path.startswith(prefix):
                    rel = path[len(prefix):] if prefix else path
                    changed.append(rel)

            # Check untracked files
            for f in repo.untracked_files:
                f_clean = f.replace("\\", "/")
                if not prefix or f_clean.startswith(prefix):
                    rel = f_clean[len(prefix):] if prefix else f_clean
                    changed.append(rel)

            return sorted(list(set(changed)))
        except Exception as e:
            logger.warning(f"Failed to get changed files via GitPython: {e}")
            try:
                proc = subprocess.run(
                    ["git", "status", "--porcelain", "--", "."],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                )
                files = []
                for line in proc.stdout.splitlines():
                    if len(line) < 4:
                        continue
                    path = line[3:].strip()
                    if " -> " in path:
                        path = path.split(" -> ", 1)[1]
                    path = path.replace("\\", "/")
                    if prefix:
                        if path.startswith(prefix):
                            files.append(path[len(prefix):])
                    else:
                        files.append(path)
                return sorted(list(set(files)))
            except Exception:
                return []
        finally:
            GitService._close_repo(repo)