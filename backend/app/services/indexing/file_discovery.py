"""File discovery and filtering service for repository ingestion."""

import os
from pathlib import Path
from typing import Generator, List, Optional, Set

# Directories always ignored during indexing
IGNORED_DIRS: Set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "coverage",
    "htmlcov",
    ".system_generated",
    "postgres_data",
}

# File extensions strictly ignored (binaries, caches, images, archives)
IGNORED_EXTENSIONS: Set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".lock",
    ".DS_Store",
}

# Extension to language mapping
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".sql": "sql",
}


def detect_language(file_path: Path | str) -> Optional[str]:
    """Detect programming language from file extension."""
    path = Path(file_path)
    ext = path.suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext)


def is_binary_file(file_path: Path, max_bytes: int = 1024) -> bool:
    """Check if a file appears to be binary by scanning for null bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(max_bytes)
            if b"\x00" in chunk:
                return True
        return False
    except (OSError, PermissionError):
        return True


def parse_gitignore_patterns(repo_root: Path) -> List[str]:
    """Load ignore patterns from repository .gitignore if it exists."""
    gitignore_file = repo_root / ".gitignore"
    patterns: List[str] = []
    if gitignore_file.is_file():
        try:
            with open(gitignore_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        # Normalize pattern
                        patterns.append(stripped.rstrip("/"))
        except Exception:
            pass
    return patterns


def should_ignore_file(file_path: Path, repo_root: Path, custom_patterns: Optional[List[str]] = None) -> bool:
    """Check whether a file should be skipped during indexing."""
    # Check parts against IGNORED_DIRS
    for part in file_path.relative_to(repo_root).parts:
        if part in IGNORED_DIRS:
            return True

    # Check extension
    if file_path.suffix.lower() in IGNORED_EXTENSIONS:
        return True

    # Check custom gitignore patterns
    if custom_patterns:
        rel_str = str(file_path.relative_to(repo_root)).replace("\\", "/")
        for pattern in custom_patterns:
            if pattern and (pattern in rel_str or rel_str.endswith(pattern)):
                return True

    # Check if binary
    if is_binary_file(file_path):
        return True

    return False


def discover_source_files(repo_path: Path | str) -> Generator[Path, None, None]:
    """Discover all parseable source code files in a repository directory."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        return

    gitignore_patterns = parse_gitignore_patterns(root)

    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)

        # Modify dirnames in-place to avoid descending into ignored directories
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not any(
                p for p in gitignore_patterns if p == d or d.startswith(".")
            )
        ]

        for filename in filenames:
            file_path = current_dir / filename
            if not should_ignore_file(file_path, root, gitignore_patterns):
                yield file_path
