"""Git workspace and diff generation service."""

import fnmatch
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple
import git

from app.core.logging import logger

# Runtime-generated Python/test artifacts. These are pruned from a workspace
# before staging so they can never enter the persisted review diff, even for
# workspaces created before per-workspace .gitignore files existed.
#
# ARTIFACT_DIR_NAMES matches directory names exactly (`name in ARTIFACT_DIR_NAMES`).
# "build", "dist", and ".eggs" are always named exactly that -- same names
# app.services.workspace_manager.EXCLUDED_DIR_NAMES already treats as
# disposable when copying a repo into a fresh workspace, applied here to the
# artifacts a task's own execution (chiefly `pip install --target X .` in
# app.services.verification.engine._install_python_deps_isolated, which
# writes build/ and *.egg-info into the CURRENT directory as a normal
# side effect regardless of --target) generates afterward.
#
# *.egg-info is NOT a fixed name -- it's "<distribution-name>.egg-info", a
# different string per package -- so it cannot go in ARTIFACT_DIR_NAMES (a
# plain `in` check would never match it). It's matched by fnmatch against
# ARTIFACT_DIR_PATTERNS instead, mirroring how ARTIFACT_FILE_PATTERNS
# already fnmatch-matches file names below.
ARTIFACT_DIR_NAMES = {"__pycache__", ".pytest_cache", "build", "dist", ".eggs"}
ARTIFACT_DIR_PATTERNS = {"*.egg-info"}
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
    def _tracked_files_and_root(workspace_dir: Path) -> Tuple[Optional[Path], FrozenSet[str]]:
        """Resolve the git repository root and the set of tracked file paths
        (relative to that root, forward-slash-separated, via `git ls-files`)
        for the repo containing workspace_dir.

        Returns (None, frozenset()) if workspace_dir is not inside a git
        repository -- callers then have nothing to protect and fall back to
        the pre-existing unconditional-prune behavior for that case.
        """
        repo = None
        try:
            repo = git.Repo(workspace_dir, search_parent_directories=True)
            repo_root = Path(repo.working_tree_dir).resolve()
            output = repo.git.ls_files()
            tracked = frozenset(output.splitlines()) if output else frozenset()
            return repo_root, tracked
        except Exception:
            return None, frozenset()
        finally:
            GitService._close_repo(repo)

    @staticmethod
    def _is_tracked(rel_path: str, tracked: FrozenSet[str], is_dir: bool) -> bool:
        """True if `rel_path` (relative to the repo root, forward-slash
        style) is itself a tracked file, or -- for a directory candidate --
        git tracks at least one file underneath it. Used so a name merely
        matching an artifact pattern is never sufficient reason to delete
        it: only a path git doesn't know about is a generated artifact
        rather than legitimate, already-committed content."""
        if is_dir:
            prefix = rel_path.rstrip("/") + "/"
            return any(t == rel_path or t.startswith(prefix) for t in tracked)
        return rel_path in tracked

    @staticmethod
    def _prune_runtime_artifacts(workspace_dir: Path) -> None:
        """Delete generated Python/test artifacts from the workspace on disk.

        Test runs generate __pycache__ directories, .pyc files,
        .pytest_cache, and .coverage; installing a local package (`pip
        install --target X .`, see
        app.services.verification.engine._install_python_deps_isolated)
        generates build/, dist/, .eggs/, and *.egg-info inside the CURRENT
        directory as a normal side effect, regardless of --target. Pruning
        all of these before staging keeps them out of `git add -A` and
        therefore out of the task diff, regardless of any .gitignore
        present.

        Git-aware: a candidate is only ever deleted if git does not already
        track it (or, for a directory, does not track anything underneath
        it) -- a legitimately committed `build/` directory or `.coverage`
        file is never touched, no matter its name. If workspace_dir isn't
        inside a git repository at all, there is nothing tracked to protect
        and every match is pruned, matching the previous behavior.
        """
        root = Path(workspace_dir)
        if not root.is_dir():
            return

        repo_root, tracked = GitService._tracked_files_and_root(root)
        base = repo_root if repo_root is not None else root.resolve()

        for dirpath, dirnames, filenames in os.walk(root):
            for name in list(dirnames):
                if name in ARTIFACT_DIR_NAMES or any(
                    fnmatch.fnmatch(name, pattern) for pattern in ARTIFACT_DIR_PATTERNS
                ):
                    candidate = Path(dirpath) / name
                    rel = candidate.resolve().relative_to(base).as_posix()
                    if GitService._is_tracked(rel, tracked, is_dir=True):
                        continue
                    shutil.rmtree(candidate, ignore_errors=True)
                    dirnames.remove(name)
            for name in filenames:
                if any(fnmatch.fnmatch(name, pattern) for pattern in ARTIFACT_FILE_PATTERNS):
                    candidate = Path(dirpath) / name
                    rel = candidate.resolve().relative_to(base).as_posix()
                    if GitService._is_tracked(rel, tracked, is_dir=False):
                        continue
                    try:
                        candidate.unlink()
                    except OSError as e:
                        logger.warning(f"Could not prune artifact '{candidate}': {e}")

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
    def create_branch(repo_path: Path | str, branch_name: str, base_branch: str) -> None:
        """Create (or reset) a local branch off base_branch and check it out.

        If a branch with this name already exists (e.g. a retried approval),
        it is deleted and recreated from the current tip of base_branch so
        the branch always reflects the latest approved patch.
        """
        repo = None
        try:
            repo = git.Repo(Path(repo_path).resolve())
            repo.git.checkout(base_branch, force=True)
            repo.git.clean("-fd")
            if branch_name in [h.name for h in repo.heads]:
                repo.git.branch("-D", branch_name)
            repo.git.checkout("-b", branch_name)
        finally:
            GitService._close_repo(repo)

    @staticmethod
    def checkout(repo_path: Path | str, branch_name: str, force: bool = False) -> None:
        """Check out an existing local branch.

        force=True discards any uncommitted working-tree changes so the
        repository lands in a clean, deterministic state -- used when
        rolling back a task branch after a failed approval step.
        """
        repo = None
        try:
            repo = git.Repo(Path(repo_path).resolve())
            if force:
                repo.git.checkout(branch_name, force=True)
                repo.git.clean("-fd")
            else:
                repo.git.checkout(branch_name)
        finally:
            GitService._close_repo(repo)

    @staticmethod
    def delete_branch(repo_path: Path | str, branch_name: str) -> None:
        """Force-delete a local branch if it exists. Never raises."""
        repo = None
        try:
            repo = git.Repo(Path(repo_path).resolve())
            if branch_name in [h.name for h in repo.heads]:
                repo.git.branch("-D", branch_name)
        except Exception as e:
            logger.warning(f"Could not delete branch '{branch_name}' in '{repo_path}': {e}")
        finally:
            GitService._close_repo(repo)

    @staticmethod
    def commit_all(repo_path: Path | str, message: str) -> bool:
        """Stage and commit all changes (including new files) in the working tree.

        Prunes runtime test artifacts first, same as stage_all_changes.
        Returns False if there was nothing to commit.
        """
        workspace = Path(repo_path).resolve()
        GitService._prune_runtime_artifacts(workspace)

        repo = None
        try:
            repo = git.Repo(workspace)
            repo.git.add(A=True)
            if not repo.is_dirty(untracked_files=True) and not repo.index.diff("HEAD"):
                return False
            with repo.config_writer() as writer:
                writer.set_value("user", "name", "RepoPilot")
                writer.set_value("user", "email", "repopilot@localhost")
            repo.index.commit(message)
            return True
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