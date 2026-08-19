"""Unit tests for file discovery and language detection."""

import tempfile
from pathlib import Path
from app.services.indexing.file_discovery import (
    detect_language,
    discover_source_files,
    is_binary_file,
    should_ignore_file,
)


def test_detect_language():
    """Test language detection from file extensions."""
    assert detect_language("app/main.py") == "python"
    assert detect_language("index.ts") == "typescript"
    assert detect_language("component.jsx") == "javascriptreact"
    assert detect_language("server.go") == "go"
    assert detect_language("main.rs") == "rust"
    assert detect_language("unknown.xyz") is None


def test_is_binary_file():
    """Test binary file detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        text_file = Path(tmpdir) / "test.txt"
        text_file.write_text("Hello world text content", encoding="utf-8")
        assert is_binary_file(text_file) is False

        bin_file = Path(tmpdir) / "test.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03\x00")
        assert is_binary_file(bin_file) is True


def test_file_discovery_and_ignores():
    """Test discovering source files while ignoring .git, __pycache__, and node_modules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create valid source files
        (root / "main.py").write_text("print('hello')", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "utils.py").write_text("def util(): pass", encoding="utf-8")

        # Create ignored folders and files
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("git config", encoding="utf-8")

        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00\x00\x00")

        (root / "node_modules").mkdir()
        (root / "node_modules" / "package.json").write_text("{}", encoding="utf-8")

        # Discover files
        discovered = list(discover_source_files(root))
        rel_paths = [str(p.relative_to(root)).replace("\\", "/") for p in discovered]

        assert "main.py" in rel_paths
        assert "src/utils.py" in rel_paths
        assert not any(".git" in p for p in rel_paths)
        assert not any("__pycache__" in p for p in rel_paths)
        assert not any("node_modules" in p for p in rel_paths)
