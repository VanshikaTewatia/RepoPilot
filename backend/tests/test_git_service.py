"""Unit tests for GitService workspace diffing and subdirectory scoping."""

import tempfile
from pathlib import Path
import git

from app.services.git_service import GitService


def test_non_git_directory_fails_safely():
    """Test that a non-git directory returns empty diff and empty changed files without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_git_path = Path(tmpdir) / "empty_dir"
        non_git_path.mkdir()

        diff = GitService.get_workspace_diff(non_git_path)
        assert diff == ""

        changed = GitService.get_changed_files(non_git_path)
        assert changed == []


def test_non_existent_path_fails_safely():
    """Test that a non-existent path returns empty diff and empty changed files without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "does_not_exist"

        diff = GitService.get_workspace_diff(missing)
        assert diff == ""

        changed = GitService.get_changed_files(missing)
        assert changed == []


def test_git_repository_root():
    """Test GitService against a git repository root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "test_repo"
        repo_dir.mkdir()
        with git.Repo.init(repo_dir) as repo:
            # Initial commit
            test_file = repo_dir / "file1.txt"
            test_file.write_text("initial content\n", encoding="utf-8")
            repo.index.add(["file1.txt"])
            repo.index.commit("Initial commit")

            # Make modification
            test_file.write_text("modified content\n", encoding="utf-8")

            diff = GitService.get_workspace_diff(repo_dir)
            assert "modified content" in diff
            assert "file1.txt" in diff

            changed = GitService.get_changed_files(repo_dir)
            assert "file1.txt" in changed


def test_git_subdirectory_scoping():
    """Test that GitService correctly scopes diff and changed files when workspace is a subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "parent_repo"
        repo_root.mkdir()
        with git.Repo.init(repo_root) as repo:
            # Subdirectories
            sub_a = repo_root / "service_a"
            sub_b = repo_root / "service_b"
            sub_a.mkdir()
            sub_b.mkdir()

            # Files in both subdirectories
            file_a = sub_a / "app.py"
            file_b = sub_b / "app.py"
            file_a.write_text("service a v1\n", encoding="utf-8")
            file_b.write_text("service b v1\n", encoding="utf-8")

            repo.index.add(["service_a/app.py", "service_b/app.py"])
            repo.index.commit("Commit services A and B")

            # Modify BOTH service A and service B
            file_a.write_text("service a v2 modified\n", encoding="utf-8")
            file_b.write_text("service b v2 modified\n", encoding="utf-8")

            # 1. Inspect Service A only
            diff_a = GitService.get_workspace_diff(sub_a)
            changed_a = GitService.get_changed_files(sub_a)

            assert "service a v2 modified" in diff_a
            assert "service b" not in diff_a
            assert "app.py" in changed_a
            assert "service_b/app.py" not in changed_a

            # 2. Inspect Service B only
            diff_b = GitService.get_workspace_diff(sub_b)
            changed_b = GitService.get_changed_files(sub_b)

            assert "service b v2 modified" in diff_b
            assert "service a" not in diff_b
            assert "app.py" in changed_b
            assert "service_a/app.py" not in changed_b


def test_subdirectory_untracked_files_scoping():
    """Test that untracked files are reported and scoped when the workspace is a subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "parent_repo"
        repo_root.mkdir()
        with git.Repo.init(repo_root) as repo:
            sub_a = repo_root / "service_a"
            sub_b = repo_root / "service_b"
            sub_a.mkdir()
            sub_b.mkdir()

            (sub_a / "tracked.py").write_text("tracked\n", encoding="utf-8")
            (sub_b / "tracked.py").write_text("tracked\n", encoding="utf-8")
            repo.index.add(["service_a/tracked.py", "service_b/tracked.py"])
            repo.index.commit("Commit tracked files")

            # Untracked file in service_a only
            (sub_a / "new_feature.py").write_text("new\n", encoding="utf-8")

            diff_a = GitService.get_workspace_diff(sub_a)
            changed_a = GitService.get_changed_files(sub_a)

            assert "new_feature.py" in changed_a
            assert "service_b" not in changed_a
            assert "new_feature.py" in diff_a
            assert "service_b" not in diff_a


def _make_committed_repo(root: Path) -> Path:
    """Build a minimal committed git repository simulating a pre-fix workspace (no .gitignore)."""
    repo_dir = root / "ws_repo"
    repo_dir.mkdir()
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    with git.Repo.init(repo_dir) as repo:
        repo.git.add(A=True)
        repo.index.commit("baseline")
    return repo_dir


def _create_pytest_artifacts(ws: Path) -> None:
    """Simulate the artifacts pytest generates inside a workspace during a test run."""
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "conftest.cpython-314-pytest-9.1.1.pyc").write_bytes(b"\x00\x01")
    (ws / "src" / "__pycache__").mkdir()
    (ws / "src" / "__pycache__" / "app.cpython-314.pyc").write_bytes(b"\x00\x02")
    (ws / ".pytest_cache" / "v").mkdir(parents=True)
    (ws / ".pytest_cache" / "v" / "lastfailed").write_text("{}\n", encoding="utf-8")
    (ws / ".coverage").write_bytes(b"\x01\x02")


def test_stage_all_changes_excludes_runtime_artifacts():
    """Regression test for Task #6: artifacts generated by a sandbox test run must
    never be staged or appear in the review diff, even without any .gitignore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = _make_committed_repo(Path(tmpdir))
        _create_pytest_artifacts(ws)

        # Legitimate agent changes alongside the artifacts
        (ws / "src" / "app.py").write_text("VALUE = 42\n", encoding="utf-8")
        (ws / "new_module.py").write_text("NEW = 1\n", encoding="utf-8")

        assert GitService.stage_all_changes(ws) is True

        changed = GitService.get_changed_files(ws)
        assert changed == ["new_module.py", "src/app.py"]

        diff = GitService.get_workspace_diff(ws)
        for banned in ("__pycache__", ".pyc", ".pytest_cache", ".coverage"):
            assert banned not in diff, f"artifact '{banned}' leaked into diff"
        assert "-VALUE = 1" in diff
        assert "+VALUE = 42" in diff
        assert "+NEW = 1" in diff

        # Artifacts were pruned from disk, so later staging passes stay clean too
        assert not (ws / "__pycache__").exists()
        assert not (ws / "src" / "__pycache__").exists()
        assert not (ws / ".pytest_cache").exists()
        assert not (ws / ".coverage").exists()


def test_stage_all_changes_preserves_legitimate_source_changes():
    """Pruning must only remove generated artifacts, never legitimate changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = _make_committed_repo(Path(tmpdir))
        (ws / ".coverage").write_bytes(b"\x01")  # name collision candidate
        (ws / "src" / "coverage_tool.py").write_text("TOOL = 1\n", encoding="utf-8")
        (ws / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

        assert GitService.stage_all_changes(ws) is True

        changed = GitService.get_changed_files(ws)
        assert "src/coverage_tool.py" in changed
        assert "src/app.py" in changed
        assert ".coverage" not in changed


# -------------------------------------------------------------------------
# Branch / commit helpers (used by the GitHub approval flow)
# -------------------------------------------------------------------------
def test_create_branch_forks_from_base_and_checks_it_out():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_committed_repo(Path(tmpdir))
        with git.Repo(repo_dir) as gr:
            base = gr.active_branch.name

        GitService.create_branch(repo_dir, "repopilot/task-1", base)

        with git.Repo(repo_dir) as gr:
            assert gr.active_branch.name == "repopilot/task-1"
            assert base in [h.name for h in gr.heads]


def test_create_branch_recreates_existing_branch_from_latest_base():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_committed_repo(Path(tmpdir))
        with git.Repo(repo_dir) as gr:
            base = gr.active_branch.name

        GitService.create_branch(repo_dir, "repopilot/task-1", base)
        # Simulate a leftover branch from a previous failed/retried approval
        # by advancing it, then re-running create_branch for the same task.
        (repo_dir / "src" / "leftover.py").write_text("X = 1\n", encoding="utf-8")
        with git.Repo(repo_dir) as gr:
            gr.git.add(A=True)
            gr.index.commit("stale attempt")

        GitService.create_branch(repo_dir, "repopilot/task-1", base)

        with git.Repo(repo_dir) as gr:
            assert gr.active_branch.name == "repopilot/task-1"
            # The stale commit is gone -- branch was rebuilt fresh from base.
            assert not (repo_dir / "src" / "leftover.py").exists()


def test_checkout_force_discards_uncommitted_changes():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_committed_repo(Path(tmpdir))
        with git.Repo(repo_dir) as gr:
            base = gr.active_branch.name

        GitService.create_branch(repo_dir, "repopilot/task-1", base)
        (repo_dir / "src" / "app.py").write_text("VALUE = 999\n", encoding="utf-8")

        GitService.checkout(repo_dir, base, force=True)

        with git.Repo(repo_dir) as gr:
            assert gr.active_branch.name == base
        assert (repo_dir / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_delete_branch_removes_local_ref_and_is_a_noop_if_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_committed_repo(Path(tmpdir))
        with git.Repo(repo_dir) as gr:
            base = gr.active_branch.name

        GitService.create_branch(repo_dir, "repopilot/task-1", base)
        GitService.checkout(repo_dir, base)
        GitService.delete_branch(repo_dir, "repopilot/task-1")

        with git.Repo(repo_dir) as gr:
            assert "repopilot/task-1" not in [h.name for h in gr.heads]

        # Deleting again (already gone) must not raise.
        GitService.delete_branch(repo_dir, "repopilot/task-1")


def test_commit_all_stages_and_commits_pending_changes():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_committed_repo(Path(tmpdir))
        (repo_dir / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
        (repo_dir / "src" / "new_file.py").write_text("NEW = True\n", encoding="utf-8")

        committed = GitService.commit_all(repo_dir, "apply fix")
        assert committed is True

        with git.Repo(repo_dir) as gr:
            assert not gr.is_dirty(untracked_files=True)
            assert gr.head.commit.message == "apply fix"


def test_commit_all_returns_false_when_nothing_to_commit():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_committed_repo(Path(tmpdir))
        assert GitService.commit_all(repo_dir, "no-op") is False