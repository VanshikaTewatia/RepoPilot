"""Unit tests for agent tools and security containment."""

import tempfile
from pathlib import Path
import pytest

from app.services.agent.tools import (
    _resolve_safe_path,
    list_files,
    search_code,
    read_file,
    apply_patch,
)


def test_path_traversal_prevention():
    """Test that directory traversal attacks raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Attempt traversal outside workspace
        with pytest.raises(ValueError, match="Path traversal detected"):
            _resolve_safe_path(workspace, "../../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal detected"):
            _resolve_safe_path(workspace, "../outside.txt")


def test_agent_tools_workflow():
    """Test list_files, read_file, search_code, and apply_patch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = str(Path(tmpdir))

        # 1. Apply patch to create a new file
        res = apply_patch(
            workspace_dir=workspace,
            file_path="src/service.py",
            patch_or_content="line 1\nline 2\nline 3\n",
        )
        assert res["success"] is True

        # 2. List files
        list_res = list_files(workspace)
        assert list_res["success"] is True
        assert "src/service.py" in list_res["files"]

        # 3. Read file with line slice
        read_res = read_file(workspace, "src/service.py", start_line=2, end_line=3)
        assert read_res["success"] is True
        assert read_res["content"] == "line 2\nline 3\n"

        # 4. Search code
        search_res = search_code(workspace, query="line 2")
        assert search_res["success"] is True
        assert len(search_res["matches"]) >= 1
        assert search_res["matches"][0]["line"] == 2

        # 5. Replace targeted line range
        patch_res = apply_patch(
            workspace_dir=workspace,
            file_path="src/service.py",
            patch_or_content="modified line 2\n",
            start_line=2,
            end_line=2,
        )
        assert patch_res["success"] is True
        assert patch_res["action"] == "replaced_range"

        # Verify modification
        final_read = read_file(workspace, "src/service.py")
        assert "modified line 2" in final_read["content"]


# ---------------------------------------------------------------------------
# search_code: multi-language default (Task #15-style regression coverage --
# the agent's own investigate_node must never be blind to non-Python repos)
# ---------------------------------------------------------------------------
def _write(root: Path, rel_path: str, content: str) -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_search_code_default_finds_matches_across_languages():
    """With no file_pattern given, search_code() must search JS/TS/TSX/Go/
    Rust/Java/C#/Dart files, not just *.py -- the old default silently
    searched Python only, which is exactly what made investigate_node blind
    to non-Python repositories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(workspace, "src/auth.js", "function login() { return AUTH_TOKEN; }\n")
        _write(workspace, "src/Auth.tsx", "export function AuthProvider() { return AUTH_TOKEN; }\n")
        _write(workspace, "src/auth.go", "func Login() string { return AUTH_TOKEN }\n")
        _write(workspace, "src/Auth.rs", "fn login() -> &'static str { AUTH_TOKEN }\n")
        _write(workspace, "src/Auth.java", "class Auth { String token = AUTH_TOKEN; }\n")
        _write(workspace, "src/Auth.cs", "class Auth { string token = AUTH_TOKEN; }\n")
        _write(workspace, "lib/auth.dart", "String login() => AUTH_TOKEN;\n")

        res = search_code(str(workspace), query="AUTH_TOKEN")

        assert res["success"] is True
        matched_files = {m["file"] for m in res["matches"]}
        assert matched_files == {
            "src/auth.js",
            "src/Auth.tsx",
            "src/auth.go",
            "src/Auth.rs",
            "src/Auth.java",
            "src/Auth.cs",
            "lib/auth.dart",
        }


def test_search_code_explicit_single_pattern_narrows_to_one_language():
    """An explicit file_pattern still narrows the search exactly as before
    (e.g. restoring the old Python-only behavior on request)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(workspace, "a.py", "TARGET = 1\n")
        _write(workspace, "b.js", "const TARGET = 1;\n")

        res = search_code(str(workspace), query="TARGET", file_pattern="*.py")

        assert res["success"] is True
        assert {m["file"] for m in res["matches"]} == {"a.py"}


def test_search_code_explicit_list_of_patterns():
    """A list of globs matches the union of those extensions only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(workspace, "a.ts", "TARGET\n")
        _write(workspace, "a.tsx", "TARGET\n")
        _write(workspace, "a.py", "TARGET\n")

        res = search_code(str(workspace), query="TARGET", file_pattern=["*.ts", "*.tsx"])

        assert res["success"] is True
        assert {m["file"] for m in res["matches"]} == {"a.ts", "a.tsx"}
