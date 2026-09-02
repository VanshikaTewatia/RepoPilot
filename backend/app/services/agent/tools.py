"""Secure tools for LangGraph Agent repository inspection and editing."""

from pathlib import Path
import fnmatch
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from app.core.logging import logger
from app.services.indexing.file_discovery import IGNORED_DIRS
from app.services.verification.engine import VerificationEngine

# Filenames excluded from search_code() regardless of extension/glob match --
# real source-adjacent text (unlike e.g. package.json) but pure dependency-
# manifest noise for code investigation. Files ending in ".lock" are already
# excluded via file_discovery.IGNORED_EXTENSIONS' convention; these don't
# use that extension so need an explicit name match.
_SEARCH_NOISE_FILENAMES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
}

# Tokens shorter than this are too generic to score a relevance match on
# their own (matches _question_terms' >3-character convention in
# app.services.qa.investigator).
_MIN_TOKEN_LENGTH = 3

_PATH_EXACT_PHRASE_SCORE = 100
_PATH_TOKEN_SCORE = 50
_CONTENT_EXACT_PHRASE_SCORE = 20
_CONTENT_ALL_TOKENS_SCORE = 10
_CONTENT_ANY_TOKEN_SCORE = 5

# Source/config extensions searched by default when search_code() is given no
# explicit file_pattern. Covers every ecosystem the verification adapters
# recognize (JS/JSX/TS/TSX, Java/Kotlin, Go, Rust, C#, Dart, Python) plus
# common config/text files -- so a caller that doesn't specify a pattern
# never silently searches Python only, as happened previously (the old
# default was "*.py", which made the agent's own investigate_node blind to
# every non-Python repository).
_DEFAULT_SEARCH_GLOBS: Tuple[str, ...] = (
    "*.py",
    "*.js", "*.jsx", "*.ts", "*.tsx",
    "*.java", "*.kt",
    "*.go",
    "*.rs",
    "*.cs",
    "*.dart",
    "*.rb", "*.php",
    "*.c", "*.h", "*.cpp", "*.hpp",
    "*.json", "*.yaml", "*.yml", "*.toml",
    "*.md",
)


def _normalize_search_patterns(file_pattern: Optional[Union[str, Sequence[str]]]) -> Tuple[str, ...]:
    """Resolve the ``file_pattern`` argument into a concrete glob tuple.

    ``None`` (the default) means "search every common source/config
    extension" (``_DEFAULT_SEARCH_GLOBS``). A single string narrows to one
    glob (e.g. ``"*.py"`` for the old Python-only behavior, or ``"*.tsx"``).
    A sequence narrows to exactly those globs (e.g. ``["*.go"]`` or
    ``["*.ts", "*.tsx"]``).
    """
    if file_pattern is None:
        return _DEFAULT_SEARCH_GLOBS
    if isinstance(file_pattern, str):
        return (file_pattern,)
    return tuple(file_pattern)


def _resolve_safe_path(base_dir: Path | str, target_rel_path: str) -> Path:
    """Resolve and validate path within base directory, strictly preventing path traversal."""
    base = Path(base_dir).resolve()
    # Normalize path separators
    clean_rel = os.path.normpath(target_rel_path.strip().lstrip("/\\"))
    resolved = (base / clean_rel).resolve()

    if not str(resolved).startswith(str(base)):
        raise ValueError(
            f"Path traversal detected: '{target_rel_path}' is outside the authorized workspace."
        )
    return resolved


def list_files(workspace_dir: str, sub_dir: str = ".") -> Dict[str, Any]:
    """List source files in the workspace directory safely."""
    try:
        target_dir = _resolve_safe_path(workspace_dir, sub_dir)
        if not target_dir.is_dir():
            return {"success": False, "error": f"Directory not found: {sub_dir}", "files": []}

        files_found: List[str] = []
        base_path = Path(workspace_dir).resolve()

        for root, dirs, files in os.walk(target_dir):
            # Ignore hidden and build folders
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", "venv", ".venv")]
            for f in files:
                full = Path(root) / f
                rel = str(full.relative_to(base_path)).replace("\\", "/")
                files_found.append(rel)

        return {"success": True, "files": sorted(files_found), "total": len(files_found)}
    except Exception as e:
        return {"success": False, "error": str(e), "files": []}


def _search_tokens(text: str) -> List[str]:
    """Split free text into lowercase word tokens, dropping anything too
    short to be a meaningful relevance signal on its own."""
    return [t for t in re.split(r"\W+", text.lower()) if len(t) >= _MIN_TOKEN_LENGTH]


def _content_relevance(line_lower: str, query_lower: str, tokens: List[str]) -> int:
    if query_lower and query_lower in line_lower:
        return _CONTENT_EXACT_PHRASE_SCORE
    if tokens:
        hits = sum(1 for t in tokens if t in line_lower)
        if hits == len(tokens):
            return _CONTENT_ALL_TOKENS_SCORE
        if hits > 0:
            return _CONTENT_ANY_TOKEN_SCORE
    return 0


def _path_relevance(rel_path_lower: str, query_lower: str, tokens: List[str]) -> int:
    if query_lower and query_lower in rel_path_lower:
        return _PATH_EXACT_PHRASE_SCORE
    if tokens and any(t in rel_path_lower for t in tokens):
        return _PATH_TOKEN_SCORE
    return 0


def search_code(
    workspace_dir: str,
    query: str,
    file_pattern: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Search for text (case-insensitive) in workspace files, ranked by
    relevance rather than raw filesystem-walk order.

    ``file_pattern`` accepts a single glob (``"*.tsx"``), a list of globs
    (``["*.go", "*.mod"]``), or ``None`` (default) to search across every
    ecosystem's common source extensions plus common config/text files --
    see ``_DEFAULT_SEARCH_GLOBS``. Pass an explicit pattern to narrow the
    search to one language/ecosystem; never assume the repository is Python.

    Relevance ranking (highest first, deterministic ties broken by file
    path then line number -- never an LLM call, just a fixed scoring rule):
      1. The file's own path contains the exact query phrase.
      2. The file's own path contains one of the query's word tokens.
      3. A line's content contains the exact query phrase.
      4. A line's content contains every one of the query's word tokens.
      5. A line's content contains at least one query token.
    A file whose path matches (1)/(2) is still returned even if no single
    line's content matches at all (e.g. searching "auth" should surface
    ``AuthContext.jsx`` by name, not just by grep hits inside it).

    Dependency-manifest noise (node_modules, build/dist output,
    package-lock.json, and the other directories/extensions
    app.services.indexing.file_discovery already excludes from indexing)
    is skipped so common terms don't get crowded out by thousands of lock-
    file lines.
    """
    try:
        patterns = _normalize_search_patterns(file_pattern)
        base_path = Path(workspace_dir).resolve()
        query_stripped = query.strip()
        query_lower = query_stripped.lower()
        tokens = _search_tokens(query_stripped)

        # (score, file, line, content) -- sorted once at the end for
        # deterministic, relevance-first ordering.
        candidates: List[Tuple[int, str, int, str]] = []

        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
            for f in files:
                if f in _SEARCH_NOISE_FILENAMES:
                    continue
                if not any(fnmatch.fnmatch(f, p) for p in patterns):
                    continue

                file_path = Path(root) / f
                rel_path = str(file_path.relative_to(base_path)).replace("\\", "/")
                path_score = _path_relevance(rel_path.lower(), query_lower, tokens)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                        file_lines = fp.readlines()
                except Exception:
                    continue

                found_content_match = False
                for idx, line in enumerate(file_lines, start=1):
                    content_score = _content_relevance(line.lower(), query_lower, tokens)
                    if content_score > 0:
                        found_content_match = True
                        candidates.append((path_score + content_score, rel_path, idx, line.rstrip()))

                if not found_content_match and path_score > 0:
                    first_line = file_lines[0].rstrip() if file_lines else ""
                    candidates.append((path_score, rel_path, 1, first_line))

        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        matches = [{"file": f, "line": ln, "content": c} for _, f, ln, c in candidates[:50]]

        return {"success": True, "query": query, "matches": matches, "total_matches": len(candidates)}
    except Exception as e:
        return {"success": False, "error": str(e), "matches": []}


def read_file(
    workspace_dir: str,
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Read full content or targeted line slice of a workspace file."""
    try:
        target = _resolve_safe_path(workspace_dir, file_path)
        if not target.is_file():
            return {"success": False, "error": f"File not found: {file_path}", "content": ""}

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        s_line = max(1, start_line) if start_line is not None else 1
        e_line = min(total_lines, end_line) if end_line is not None else total_lines

        selected_lines = lines[s_line - 1 : e_line]
        content = "".join(selected_lines)

        return {
            "success": True,
            "file_path": file_path,
            "start_line": s_line,
            "end_line": e_line,
            "total_lines": total_lines,
            "content": content,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "content": ""}


def apply_patch(
    workspace_dir: str,
    file_path: str,
    patch_or_content: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply a targeted code patch or replace a line range safely."""
    try:
        target = _resolve_safe_path(workspace_dir, file_path)
        if not target.is_file():
            # Create file if it doesn't exist
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(patch_or_content)
            return {"success": True, "file_path": file_path, "action": "created"}

        with open(target, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line is not None and end_line is not None:
            # Validate line bounds
            if start_line < 1 or end_line < start_line or start_line > len(lines) + 1:
                return {
                    "success": False,
                    "error": f"Invalid line range {start_line}-{end_line} for file '{file_path}' containing {len(lines)} lines.",
                    "file_path": file_path,
                }

            # Replace line range
            s_idx = max(0, start_line - 1)
            e_idx = min(len(lines), end_line)
            replacement_lines = [l if l.endswith("\n") else l + "\n" for l in patch_or_content.splitlines(keepends=True)]
            new_lines = lines[:s_idx] + replacement_lines + lines[e_idx:]
            with open(target, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            return {
                "success": True,
                "file_path": file_path,
                "action": "replaced_range",
                "start_line": start_line,
                "end_line": end_line,
            }
        else:
            # Complete overwrite if no line range specified
            with open(target, "w", encoding="utf-8") as f:
                f.write(patch_or_content)
            return {"success": True, "file_path": file_path, "action": "overwritten"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_tests(workspace_dir: str, test_path: Optional[str] = None) -> Dict[str, Any]:
    """Detect the repository's project ecosystem and run its verification command
    in Docker sandbox or secure local runner. Never assumes pytest for a
    non-Python project -- see app.services.verification for the detection and
    per-ecosystem adapter logic."""
    engine = VerificationEngine()
    return engine.verify(workspace_path=workspace_dir, test_path=test_path)


def run_tests_for_task(
    workspace_dir: str,
    task_description: str = "",
    keyword_matches: Optional[List[Dict[str, Any]]] = None,
    test_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Repository- and task-aware verification.

    Detects every project actually present in the workspace and runs the
    one(s) relevant to this task, instead of assuming the workspace root is
    a single project -- see VerificationEngine.verify_repository and
    app.services.verification.project_analyzer for the multi-project
    detection and evidence-based selection logic. A workspace that resolves
    to a single project behaves identically to run_tests().
    """
    engine = VerificationEngine()
    return engine.verify_repository(
        workspace_path=workspace_dir,
        task_description=task_description,
        keyword_matches=keyword_matches,
        test_path=test_path,
    )
