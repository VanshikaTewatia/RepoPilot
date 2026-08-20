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