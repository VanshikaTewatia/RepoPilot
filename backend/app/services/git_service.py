"""Git workspace and diff generation service."""

import fnmatch
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import git

from app.core.logging import logger

# Runtime-generated Python/test artifacts. These are pruned from a workspace
# before staging so they can never enter the persisted review diff, even for
# workspaces created before per-workspace .gitignore files existed.
ARTIFACT_DIR_NAMES = {"__pycache__", ".pytest_cache"}
ARTIFACT_FILE_PATTERNS = {"*.pyc", "*.pyo", ".coverage"}


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

    @staticmethod
    def _prune_runtime_artifacts(workspace_dir: Path) -> None:
        """Delete pytest/bytecode artifacts from the workspace on disk.

        Test runs generate __pycache__ directories, .pyc files, .pytest_cache,
        and .coverage inside the workspace. Pruning them before staging keeps
        generated artifacts out of `git add -A` and therefore out of the task
        diff, regardless of any .gitignore present.
        """
        root = Path(workspace_dir)
        if not root.is_dir():
            return

        for dirpath, dirnames, filenames in os.walk(root):
            for name in list(dirnames):
                if name in ARTIFACT_DIR_NAMES:
                    shutil.rmtree(Path(dirpath) / name, ignore_errors=True)
                    dirnames.remove(name)
            for name in filenames:
                if any(fnmatch.fnmatch(name, pattern) for pattern in ARTIFACT_FILE_PATTERNS):
                    try:
                        (Path(dirpath) / name).unlink()
                    except OSError as e:
                        logger.warning(f"Could not prune artifact '{dirpath / name}': {e}")

    @staticmethod
    def stage_all_changes(workspace_dir: Path | str) -> bool:
        """Stage all changes (including new files) in a workspace git repository.

        Staging makes new files appear as proper, applicable diff hunks in
        `git diff HEAD` output instead of an untracked-file listing. Generated
        runtime artifacts are pruned first so they are never staged.
        """
        workspace = Path(workspace_dir).resolve()
        if not workspace.is_dir():
            return False

        GitService._prune_runtime_artifacts(workspace)

        repo = None
        try:
            repo = git.Repo(workspace, search_parent_directories=True)
            repo.git.add(A=True)
            return True
        except Exception as e:
            logger.warning(f"Failed to stage changes in '{workspace}': {e}")
            return False
        finally:
            GitService._close_repo(repo)

    @staticmethod
    def apply_diff(target_dir: Path | str, diff_text: str) -> Tuple[bool, str]:
        """Safely apply a unified diff to a git repository working tree.

        Runs `git apply --check` first so the patch is only applied when it
        fits the current state of the target without overwriting unrelated
        changes. Never uses fuzzy matching.

        Returns (success, error_output).
        """
        if not diff_text or not diff_text.strip():
            return False, "Empty diff"

        target = Path(target_dir).resolve()
        if not target.is_dir():
            return False, f"Target directory does not exist: {target}"

        repo = None
        tmp_path: Optional[str] = None
        try:
            repo = git.Repo(target, search_parent_directories=True)

            # Write patch to a temp file with exact line endings; newline=""
            # prevents Windows CRLF translation from corrupting the patch.
            # A trailing newline is required or git apply reports the patch
            # as corrupt at its final line.
            patch_text = diff_text if diff_text.endswith("\n") else diff_text + "\n"
            with tempfile.NamedTemporaryFile(
                "w", suffix=".patch", delete=False, encoding="utf-8", newline=""
            ) as tmp:
                tmp.write(patch_text)
                tmp_path = tmp.name

            repo.git.apply("--check", tmp_path)
            repo.git.apply(tmp_path)
            return True, ""
        except git.GitCommandError as e:
            stderr = (e.stderr or "").strip() or str(e)
            logger.warning(f"git apply failed against '{target}': {stderr}")
            return False, stderr
        except Exception as e:
            logger.warning(f"Failed to apply diff to '{target}': {e}")
            return False, str(e)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
            GitService._close_repo(repo)