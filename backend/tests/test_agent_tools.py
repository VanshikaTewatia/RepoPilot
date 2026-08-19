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
