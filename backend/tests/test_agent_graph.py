"""Unit tests for LangGraph state machine execution, patch generation, validation, and retry loop."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.services.agent.graph import (
    agent_app,
    plan_node,
    edit_node,
    investigate_node,
    retrieve_node,
    validate_patch,
    parse_and_validate_patches,
    _generate_patches_with_gemini,
)


def test_validate_patch_and_malformed_rejection():
    """Test patch validation logic rejects malformed inputs safely."""
    # Valid whole-file replacement
    p1 = {"file_path": "src/code.py", "code": "print('hello')\n"}
    v1 = validate_patch(p1)
    assert v1 is not None
    assert v1["file_path"] == "src/code.py"
    assert v1["start_line"] is None

    # Valid line range replacement
    p2 = {"file_path": "src/code.py", "code": "new_line\n", "start_line": 5, "end_line": 10}
    v2 = validate_patch(p2)
    assert v2 is not None
    assert v2["start_line"] == 5
    assert v2["end_line"] == 10

    # Malformed: not a dict
    assert validate_patch("not a dict") is None

    # Malformed: empty file_path or missing code
    assert validate_patch({"file_path": "", "code": "abc"}) is None
    assert validate_patch({"file_path": "src/code.py"}) is None

    # Malformed: path traversal
    assert validate_patch({"file_path": "../outside.py", "code": "abc"}) is None
    assert validate_patch({"file_path": "/etc/passwd", "code": "abc"}) is None
    assert validate_patch({"file_path": "src/../../outside.py", "code": "abc"}) is None

    # Malformed: invalid line range (start > end or start < 1)
    assert validate_patch({"file_path": "a.py", "code": "c", "start_line": 10, "end_line": 5}) is None
    assert validate_patch({"file_path": "a.py", "code": "c", "start_line": 0, "end_line": 5}) is None
    assert validate_patch({"file_path": "a.py", "code": "c", "start_line": "5", "end_line": 10}) is None


def test_parse_and_validate_patches_safe_parsing():
    """Test parsing markdown fences and malformed JSON safely."""
    # Valid markdown JSON block
    raw_markdown = "```json\n[{\"file_path\": \"src/app.py\", \"code\": \"x = 1\\n\", \"start_line\": 1, \"end_line\": 2}]\n```"
    res = parse_and_validate_patches(raw_markdown)
    assert len(res) == 1
    assert res[0]["file_path"] == "src/app.py"

    # Completely invalid non-JSON output
    bad_output = "I cannot fulfill this request because of an error."
    res_bad = parse_and_validate_patches(bad_output)
    assert res_bad == []

    # Mixed valid and invalid items
    mixed = json.dumps([
        {"file_path": "valid.py", "code": "a = 1"},
        {"file_path": "../evil.py", "code": "evil()"},
        {"file_path": "bad_lines.py", "code": "b = 2", "start_line": 10, "end_line": 2},
    ])
    res_mixed = parse_and_validate_patches(mixed)
    assert len(res_mixed) == 1
    assert res_mixed[0]["file_path"] == "valid.py"


def test_plan_node_generates_patches_from_mocked_gemini():
    """Test plan_node calls Gemini and stores valid proposed_patches in state."""
    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {
            "file_path": "src/order_service.py",
            "code": "        return order.subtotal * 0.15\n",
            "start_line": 40,
            "end_line": 42,
        }
    ])

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    state = {
        "workspace_dir": "/tmp/test_ws",
        "task_description": "Fix VIP discount calculation",
        "retrieved_context": [
            {"file_path": "src/order_service.py", "content": "def calculate_order_total():\n    pass", "total_lines": 50}
        ],
        "error_analysis": None,
    }

    with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"):
        with patch("google.genai.Client", return_value=mock_client):
            out = plan_node(state)

    assert out["status"] == "planning"
    assert len(out["proposed_patches"]) == 1
    patch_item = out["proposed_patches"][0]
    assert patch_item["file_path"] == "src/order_service.py"
    assert patch_item["start_line"] == 40
    assert patch_item["end_line"] == 42
    assert "Generated 1 patch(es)" in out["repair_plan"]


def test_edit_node_applies_proposed_patches():
    """Test edit_node applies proposed patches to workspace files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target_file = workspace / "sample.py"
        target_file.write_text("line 1\nold line 2\nline 3\n", encoding="utf-8")

        state = {
            "workspace_dir": str(workspace),
            "attempt_count": 0,
            "proposed_patches": [
                {
                    "file_path": "sample.py",
                    "code": "new line 2\n",
                    "start_line": 2,
                    "end_line": 2,
                }
            ],
        }

        res = edit_node(state)
        assert res["status"] == "edited"
        assert res["attempt_count"] == 1

        content = target_file.read_text(encoding="utf-8")
        assert "new line 2" in content
        assert "old line 2" not in content


@pytest.mark.asyncio
async def test_agent_graph_success_flow():
    """Test full LangGraph state machine run with test passing on first attempt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Create a passing test file in workspace
        (workspace / "test_sample.py").write_text(
            "def test_ok():\n    assert 1 == 1\n",
            encoding="utf-8",
        )

        initial_state = {
            "task_id": 1,
            "repository_id": 1,
            "workspace_dir": str(workspace),
            "task_description": "Verify test suite works",
            "status": "pending",
            "attempt_count": 0,
            "max_attempts": 3,
            "investigation_findings": "",
            "retrieved_context": [],
            "repair_plan": "",
            "proposed_patches": [],
            "test_results": None,
            "error_analysis": None,
            "is_verified": False,
            "messages": [],
        }

        final_state = await agent_app.ainvoke(initial_state)

        assert final_state["attempt_count"] == 1
        assert final_state["is_verified"] is True
        assert final_state["status"] == "verified"


@pytest.mark.asyncio
async def test_agent_graph_retry_path_analyze_failure_to_plan():
    """Test that retry loop passes through analyze_failure -> plan -> edit and succeeds on 2nd attempt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        code_file = workspace / "math_lib.py"
        code_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        test_file = workspace / "test_math.py"
        test_file.write_text("from math_lib import add\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")

        # Attempt 1: Gemini produces empty/bad patch -> test fails
        # Attempt 2: Gemini produces correct patch with error_analysis feedback -> test passes
        response_attempt_1 = MagicMock()
        response_attempt_1.text = "[]"

        response_attempt_2 = MagicMock()
        response_attempt_2.text = json.dumps([
            {
                "file_path": "math_lib.py",
                "code": "def add(a, b):\n    return a + b\n",
                "start_line": 1,
                "end_line": 2,
            }
        ])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [response_attempt_1, response_attempt_2]

        initial_state = {
            "task_id": 10,
            "repository_id": 1,
            "workspace_dir": str(workspace),
            "task_description": "Fix addition function in math_lib",
            "status": "pending",
            "attempt_count": 0,
            "max_attempts": 3,
            "investigation_findings": "",
            "retrieved_context": [],
            "repair_plan": "",
            "proposed_patches": [],
            "test_results": None,
            "error_analysis": None,
            "is_verified": False,
            "messages": [],
        }

        with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"):
            with patch("google.genai.Client", return_value=mock_client):
                final_state = await agent_app.ainvoke(initial_state)

        assert final_state["is_verified"] is True
        assert final_state["attempt_count"] == 2
        assert final_state["status"] == "verified"
        assert "return a + b" in code_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_agent_graph_failure_loop_stops_at_max_attempts():
    """Test that LangGraph stops at max_attempts when tests persistently fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "test_fail.py").write_text(
            "def test_bad():\n    assert 1 == 2\n",
            encoding="utf-8",
        )

        initial_state = {
            "task_id": 2,
            "repository_id": 1,
            "workspace_dir": str(workspace),
            "task_description": "Failing test task",
            "status": "pending",
            "attempt_count": 0,
            "max_attempts": 3,
            "investigation_findings": "",
            "retrieved_context": [],
            "repair_plan": "",
            "proposed_patches": [],
            "test_results": None,
            "error_analysis": None,
            "is_verified": False,
            "messages": [],
        }

        final_state = await agent_app.ainvoke(initial_state)

        assert final_state["is_verified"] is False
        assert final_state["attempt_count"] == 3


@pytest.mark.asyncio
async def test_retry_causes_fresh_retrieval_and_context():
    """Test that retry loop re-reads modified files from disk so plan_node sees fresh lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        code_file = workspace / "counter.py"
        code_file.write_text("count = 0\n", encoding="utf-8")

        test_file = workspace / "test_counter.py"
        test_file.write_text("import counter\ndef test_c():\n    assert counter.count == 5\n", encoding="utf-8")

        # Attempt 1: Gemini produces count = 1 (wrong) -> fails test
        # Attempt 2: Gemini produces count = 5 (correct) based on fresh file content -> passes test
        resp1 = MagicMock()
        resp1.text = json.dumps([{"file_path": "counter.py", "code": "count = 1\n", "start_line": 1, "end_line": 1}])
        resp2 = MagicMock()
        resp2.text = json.dumps([{"file_path": "counter.py", "code": "count = 5\n", "start_line": 1, "end_line": 1}])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [resp1, resp2]

        initial_state = {
            "task_id": 20,
            "repository_id": 1,
            "workspace_dir": str(workspace),
            "task_description": "Set count to 5 in counter.py",
            "status": "pending",
            "attempt_count": 0,
            "max_attempts": 3,
            "investigation_findings": "",
            "retrieved_context": [],
            "repair_plan": "",
            "proposed_patches": [],
            "test_results": None,
            "error_analysis": None,
            "is_verified": False,
            "messages": [],
        }

        with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"):
            with patch("google.genai.Client", return_value=mock_client):
                final_state = await agent_app.ainvoke(initial_state)

        assert final_state["is_verified"] is True
        assert final_state["attempt_count"] == 2
        assert "count = 5" in code_file.read_text(encoding="utf-8")


def test_stale_line_ranges_rejected_safely():
    """Test that out-of-bounds line ranges for disk files are rejected safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        sample = workspace / "short.py"
        sample.write_text("line 1\nline 2\n", encoding="utf-8")

        # Stale patch with line range beyond file length (e.g. lines 10-15)
        stale_patch = {"file_path": "short.py", "code": "pass\n", "start_line": 10, "end_line": 15}
        val = validate_patch(stale_patch, workspace_dir=str(workspace))
        assert val is None

        # Verify apply_patch also refuses out of bounds
        from app.services.agent.tools import apply_patch
        res = apply_patch(str(workspace), "short.py", "pass\n", start_line=10, end_line=15)
        assert res["success"] is False
        assert "Invalid line range" in res["error"]
        # Ensure file content was not corrupted
        assert sample.read_text(encoding="utf-8") == "line 1\nline 2\n"


@pytest.mark.asyncio
async def test_task_specific_test_target_isolation():
    """Test that specifying test_target runs and verifies only the targeted test, ignoring unrelated failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Passing test
        (workspace / "test_target_feature.py").write_text(
            "def test_target():\n    assert True\n",
            encoding="utf-8",
        )
        # Failing unrelated test in the same workspace
        (workspace / "test_unrelated_feature.py").write_text(
            "def test_unrelated():\n    assert False\n",
            encoding="utf-8",
        )

        initial_state = {
            "task_id": 30,
            "repository_id": 1,
            "workspace_dir": str(workspace),
            "task_description": "Verify target feature works",
            "test_target": "test_target_feature.py",
            "status": "pending",
            "attempt_count": 0,
            "max_attempts": 3,
            "investigation_findings": "",
            "retrieved_context": [],
            "repair_plan": "",
            "proposed_patches": [],
            "test_results": None,
            "error_analysis": None,
            "is_verified": False,
            "messages": [],
        }

        final_state = await agent_app.ainvoke(initial_state)

        # Because test_target was scoped to test_target_feature.py, it passes
        assert final_state["is_verified"] is True
        assert final_state["attempt_count"] == 1
        assert final_state["status"] == "verified"


# -------------------------------------------------------------------------
# Focused retrieval ranking (Task #6 follow-up)
# -------------------------------------------------------------------------
def _make_ranking_workspace(tmpdir: str, files: dict) -> Path:
    workspace = Path(tmpdir)
    for name, content in files.items():
        (workspace / name).write_text(content, encoding="utf-8")
    return workspace


def test_investigate_persists_keyword_matches():
    """investigate_node must expose its keyword matches for retrieval ranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_ranking_workspace(tmpdir, {"zzz_vip.py": "discount = 0.15\n"})
        state = {"workspace_dir": str(workspace), "task_description": "Apply discount logic"}

        out = investigate_node(state)

        assert len(out["keyword_matches"]) == 1
        assert out["keyword_matches"][0]["file"] == "zzz_vip.py"


def test_retrieval_ranks_relevant_files_first():
    """A relevant file must outrank alphabetically earlier irrelevant fillers."""
    files = {
        "a_alpha.py": "filler value\n",
        "b_beta.py": "filler value\n",
        "c_gamma.py": "filler value\n",
        "d_delta.py": "filler value\n",
        "e_epsilon.py": "filler value\n",
        "zzz_vip.py": "discount = 0.15\n# cart discount applied\n",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_ranking_workspace(tmpdir, files)
        state = {"workspace_dir": str(workspace), "task_description": "Apply discount to cart totals"}

        inv = investigate_node(state)
        assert inv["keyword_matches"], "expected investigation matches"

        out = retrieve_node({**state, **inv})
        retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]

        # Limit is still 5; the relevant file ranks first even though it sorts last
        assert retrieved_paths[0] == "zzz_vip.py"
        assert len(retrieved_paths) == 5
        assert "e_epsilon.py" not in retrieved_paths


def test_retrieval_ties_break_alphabetically():
    """Equal match counts keep deterministic alphabetical ordering."""
    files = {
        "m_mid.py": "plain content\n",
        "aaa_z.py": "widget here\n",
        "zzz_a.py": "widget too\n",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_ranking_workspace(tmpdir, files)
        state = {"workspace_dir": str(workspace), "task_description": "Add widget support"}

        inv = investigate_node(state)
        out = retrieve_node({**state, **inv})
        retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]

        assert retrieved_paths == ["aaa_z.py", "zzz_a.py", "m_mid.py"]


def test_retrieval_without_matches_keeps_alphabetical_order():
    """No investigation data falls back to the previous alphabetical slice."""
    files = {
        "b_beta.py": "one\n",
        "a_alpha.py": "two\n",
        "c_gamma.py": "three\n",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_ranking_workspace(tmpdir, files)
        state = {"workspace_dir": str(workspace)}  # no task_description/keyword_matches

        out = retrieve_node(state)
        retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]

        assert retrieved_paths == ["a_alpha.py", "b_beta.py", "c_gamma.py"]


# -------------------------------------------------------------------------
# Gemini minimality prompt requirements (Task #6 follow-up)
# -------------------------------------------------------------------------
def test_gemini_system_prompt_requires_minimality():
    """The patch-generation prompt must enforce minimal, on-task changes."""
    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {
            "file_path": "src/order_service.py",
            "code": "        return order.subtotal * 0.15\n",
            "start_line": 40,
            "end_line": 42,
        }
    ])

    captured = {}

    def capture_generate_content(*args, **kwargs):
        captured.update(kwargs)
        return mock_response

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = capture_generate_content

    with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"):
        with patch("google.genai.Client", return_value=mock_client):
            patches = _generate_patches_with_gemini(
                task_description="Fix VIP discount calculation",
                retrieved_context=[
                    {"file_path": "src/order_service.py", "content": "def f():\n    pass\n", "total_lines": 50}
                ],
            )

    instruction = captured["config"]["system_instruction"]
    assert "Modify only files that are necessary" in instruction
    assert "smallest possible code change" in instruction
    assert "Do not refactor unrelated code." in instruction
    assert "Do not rewrite docstrings or comments unless the task requires it." in instruction
    assert "Do not add unrelated improvements" in instruction
    assert "Do not modify behavior unrelated to the task." in instruction
    assert "Prefer targeted line replacements" in instruction
    assert "return an empty array: []" in instruction

    # Existing parsing/validation behavior is intact
    assert len(patches) == 1
    assert patches[0]["file_path"] == "src/order_service.py"
    assert patches[0]["start_line"] == 40
