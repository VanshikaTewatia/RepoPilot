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


# ---------------------------------------------------------------------------
# Phase 3C fix #4: containment must be path-component-wise (Path.parents),
# not a naive string-prefix check -- a sibling directory that merely shares
# the workspace directory's name as a text prefix is not actually contained
# within it.
# ---------------------------------------------------------------------------
def test_path_traversal_rejects_sibling_directory_with_prefix_matching_name():
    with tempfile.TemporaryDirectory() as parent:
        workspace = Path(parent) / "task_1_abc"
        workspace.mkdir()
        sibling = Path(parent) / "task_1_abcXYZ"
        sibling.mkdir()
        (sibling / "secret.txt").write_text("should not be reachable\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Path traversal detected"):
            _resolve_safe_path(workspace, "../task_1_abcXYZ/secret.txt")


def test_path_traversal_allows_legitimate_child_and_self_paths():
    """Normal child paths, and the base directory itself, must continue to
    resolve without raising."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "src").mkdir()
        (workspace / "src" / "file.py").write_text("x = 1\n", encoding="utf-8")

        resolved = _resolve_safe_path(workspace, "src/file.py")
        assert resolved == (workspace / "src" / "file.py").resolve()

        resolved_self = _resolve_safe_path(workspace, ".")
        assert resolved_self == workspace.resolve()


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


# ---------------------------------------------------------------------------
# search_code: relevance ranking (the Q5/Ecommerce_Frontend fix)
# ---------------------------------------------------------------------------
def test_search_code_prioritizes_filename_path_matches():
    """A query term appearing in the file's own path must rank that file's
    match(es) above an unrelated file that only happens to contain the
    term deep inside noisy content -- this is what lets 'auth' reliably
    surface AuthContext.jsx by name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(workspace, "src/context/AuthContext.jsx", "import React from 'react';\nexport const AuthProvider = () => null;\n")
        _write(workspace, "src/pages/ProductsPage.jsx", "// auth is mentioned once here, unrelated to the filename\nconst x = 1;\n")

        res = search_code(str(workspace), query="auth")

        assert res["success"] is True
        files_in_order = [m["file"] for m in res["matches"]]
        assert files_in_order[0] == "src/context/AuthContext.jsx"


def test_search_code_surfaces_path_match_even_without_content_match():
    """A file whose NAME matches the query must be returned even if no
    single line's content contains the query text at all."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(
            workspace, "src/pages/session/LoginPage.jsx",
            "import React from 'react';\nexport default function LoginPage() { return null; }\n",
        )

        res = search_code(str(workspace), query="session")

        assert res["success"] is True
        # "session" never appears in the file's content, only in its path.
        assert any(m["file"] == "src/pages/session/LoginPage.jsx" for m in res["matches"])


def test_search_code_prioritizes_exact_phrase_over_token_only_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(workspace, "src/exact.js", "// cart subtotal is computed right here\nconst x = 1;\n")
        _write(workspace, "src/scattered.js", "// the cart has a discount, subtotal shown separately\nconst y = 2;\n")

        res = search_code(str(workspace), query="cart subtotal")

        files_in_order = [m["file"] for m in res["matches"]]
        assert files_in_order[0] == "src/exact.js"


def test_search_code_finds_token_matches_when_no_exact_phrase_exists():
    """A multi-word query must still find content where the words appear
    separately, not only an exact phrase match (the old implementation only
    matched the literal, full query string)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(workspace, "src/cart.js", "// computes the cart item subtotal for checkout\nconst x = 1;\n")

        res = search_code(str(workspace), query="cart item component")

        assert any(m["file"] == "src/cart.js" for m in res["matches"])


def test_search_code_suppresses_package_lock_and_node_modules_noise():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(workspace, "package-lock.json", '{"name": "react", "dependencies": {"react": "auth-token-string"}}\n')
        _write(workspace, "node_modules/react/index.js", "module.exports = { auth: true };\n")
        _write(workspace, "src/AuthContext.jsx", "export const AuthContext = () => null; // auth\n")

        res = search_code(str(workspace), query="auth")

        matched_files = {m["file"] for m in res["matches"]}
        assert "package-lock.json" not in matched_files
        assert not any(f.startswith("node_modules/") for f in matched_files)
        assert "src/AuthContext.jsx" in matched_files


def test_search_code_ordering_is_deterministic():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        for i in range(5):
            _write(workspace, f"src/file_{i}.js", "const TARGET = 1;\n")

        first = search_code(str(workspace), query="TARGET")
        second = search_code(str(workspace), query="TARGET")

        assert [m["file"] for m in first["matches"]] == [m["file"] for m in second["matches"]]


def test_search_code_file_pattern_still_supported_with_relevance_ranking():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        _write(workspace, "src/AuthContext.jsx", "export const AuthContext = () => null;\n")
        _write(workspace, "src/AuthContext.py", "AUTH_CONTEXT = None\n")

        res = search_code(str(workspace), query="AuthContext", file_pattern="*.jsx")

        assert {m["file"] for m in res["matches"]} == {"src/AuthContext.jsx"}
