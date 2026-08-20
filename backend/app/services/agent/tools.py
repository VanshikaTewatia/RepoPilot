"""Secure tools for LangGraph Agent repository inspection and editing."""

from pathlib import Path
import fnmatch
import os
import re
from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.services.sandbox.docker_runner import DockerTestRunner


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


def search_code(workspace_dir: str, query: str, file_pattern: str = "*.py") -> Dict[str, Any]:
    """Search for literal text or regex patterns in workspace files."""
    try:
        base_path = Path(workspace_dir).resolve()
        matches: List[Dict[str, Any]] = []

        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", "venv", ".venv")]
            for f in files:
                if fnmatch.fnmatch(f, file_pattern):
                    file_path = Path(root) / f
                    rel_path = str(file_path.relative_to(base_path)).replace("\\", "/")
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                            for idx, line in enumerate(fp, start=1):
                                if query.lower() in line.lower():
                                    matches.append({
                                        "file": rel_path,
                                        "line": idx,
                                        "content": line.rstrip(),
                                    })
                    except Exception:
                        continue

        return {"success": True, "query": query, "matches": matches[:50], "total_matches": len(matches)}
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
    """Execute pytest suite in Docker sandbox or secure local runner."""
    runner = DockerTestRunner()
    return runner.run_tests(workspace_path=workspace_dir, test_path=test_path)
