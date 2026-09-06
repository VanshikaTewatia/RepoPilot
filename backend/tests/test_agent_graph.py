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


def _not_applicable_baseline_response() -> MagicMock:
    """The full graph now runs a real (mocked, per-test) baseline
    reproduction planning call between investigate and retrieve -- see
    app.services.agent.graph.baseline_node. Tests below that patch
    genai.Client with a scripted call queue for patch generation must
    account for this extra, earlier call; a genuine NOT_APPLICABLE response
    keeps it a no-op for tests that aren't exercising baseline behavior
    itself (see test_agent_outcomes.py for tests that do)."""
    response = MagicMock()
    response.text = json.dumps({
        "applicable": False,
        "reason": "No repository evidence supports a specific reproduction for this task.",
        "reproduction_type": "not_applicable",
        "commands": [],
    })
    return response


def _no_evidence_diagnosis_response() -> MagicMock:
    """The full graph now also runs a real (mocked, per-test) diagnosis
    Gemini call on every pass between retrieve and plan -- see
    app.services.agent.graph.diagnose_node. Tests below that patch
    genai.Client with a scripted call queue for patch generation must
    account for this extra call before each patch-generation call; a
    genuine no-evidence-shaped response keeps it advisory-inert for tests
    that aren't exercising diagnosis behavior itself (see
    test_agent_diagnosis_integration.py for tests that do).

    Note: this still parses into a DIAGNOSED diagnosis (with empty
    hypotheses and confidence "no_evidence") -- diagnoser.py always
    produces DIAGNOSED for any successfully-parsed response; only a
    parse/network failure produces DIAGNOSIS_FAILED. So patch_plan_node
    (Phase 6C) still attempts its own Gemini call after this -- see
    _planned_patch_plan_response() below.
    """
    response = MagicMock()
    response.text = json.dumps({"summary": "", "hypotheses": [], "confidence": "no_evidence"})
    return response


def _planned_patch_plan_response() -> MagicMock:
    """The full graph now also runs a real (mocked, per-test) patch-planning
    Gemini call on every pass between diagnose and plan -- see
    app.services.agent.graph.patch_plan_node. Tests below that patch
    genai.Client with a scripted call queue for patch generation must
    account for this extra call before each patch-generation call. A
    genuine, minimal PLANNED response opens plan_node's allow-list gate so
    the subsequent patch-generation call is still reached, for tests that
    aren't exercising patch-planning behavior itself (see
    test_agent_patch_plan_integration.py for tests that do)."""
    response = MagicMock()
    response.text = json.dumps({
        "applicable": True,
        "summary": "Apply the targeted fix.",
        "changes": [
            {
                "file_path": "placeholder.py",
                "change_type": "modify",
                "description": "Apply the fix.",
                "rationale": "Addresses the reported issue.",
                "citations": [],
                "symbols_affected": [],
            }
        ],
        "diagnosis_alignment": "Addresses the reported issue.",
        "confidence": "inferred",
    })
    return response


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


# ===========================================================================
# Phase 6E: deterministic same-file patch conflict resolution.
#
# edit_node applies proposed_patches in list order, mutating the file on
# disk after each apply_patch call -- but validate_patch checks every
# patch's line range against the file's ON-DISK state at validation time,
# before any patch this attempt has been applied. Two patches to the same
# file with different ranges can otherwise silently corrupt the file (the
# second patch's line numbers, computed against the pre-edit file, no
# longer point at the right content once the first patch has shifted it).
# parse_and_validate_patches now resolves this deterministically before
# proposed_patches is ever set -- see
# app.services.agent.graph._resolve_same_file_patch_conflicts.
# ===========================================================================
def test_parse_and_validate_patches_same_file_non_overlapping_survive_ordered_bottom_to_top():
    """Two non-overlapping same-file patches must both survive, returned
    descending by start_line so edit_node applies bottom-to-top -- an
    earlier (higher-line-number) edit never invalidates a not-yet-applied
    lower-line-number patch's line numbers."""
    raw = json.dumps([
        {"file_path": "a.py", "code": "top\n", "start_line": 2, "end_line": 3},
        {"file_path": "a.py", "code": "bottom\n", "start_line": 8, "end_line": 9},
    ])
    res = parse_and_validate_patches(raw)
    assert len(res) == 2
    assert res[0]["start_line"] == 8
    assert res[1]["start_line"] == 2


def test_parse_and_validate_patches_same_file_overlapping_keeps_first_drops_later():
    raw = json.dumps([
        {"file_path": "a.py", "code": "first\n", "start_line": 5, "end_line": 10},
        {"file_path": "a.py", "code": "second\n", "start_line": 8, "end_line": 12},
    ])
    res = parse_and_validate_patches(raw)
    assert len(res) == 1
    assert res[0]["code"] == "first\n"
    assert res[0]["start_line"] == 5


def test_parse_and_validate_patches_whole_file_plus_range_keeps_only_first():
    raw = json.dumps([
        {"file_path": "a.py", "code": "whole file content\n"},
        {"file_path": "a.py", "code": "range patch\n", "start_line": 3, "end_line": 4},
    ])
    res = parse_and_validate_patches(raw)
    assert len(res) == 1
    assert res[0]["code"] == "whole file content\n"
    assert res[0]["start_line"] is None

    # "First" means first by original list order, regardless of which of
    # the two is the whole-file replacement.
    raw_reversed = json.dumps([
        {"file_path": "b.py", "code": "range patch\n", "start_line": 3, "end_line": 4},
        {"file_path": "b.py", "code": "whole file content\n"},
    ])
    res_reversed = parse_and_validate_patches(raw_reversed)
    assert len(res_reversed) == 1
    assert res_reversed[0]["code"] == "range patch\n"
    assert res_reversed[0]["start_line"] == 3


def test_parse_and_validate_patches_multiple_files_all_preserved_and_unreordered():
    """Patches to different files must never be dropped and must retain
    their original first-appearance file order."""
    raw = json.dumps([
        {"file_path": "a.py", "code": "a1\n", "start_line": 1, "end_line": 1},
        {"file_path": "b.py", "code": "b1\n", "start_line": 5, "end_line": 5},
        {"file_path": "c.py", "code": "c1\n", "start_line": 9, "end_line": 9},
    ])
    res = parse_and_validate_patches(raw)
    assert [p["file_path"] for p in res] == ["a.py", "b.py", "c.py"]
    assert [p["code"] for p in res] == ["a1\n", "b1\n", "c1\n"]


def test_parse_and_validate_patches_single_patch_per_file_unchanged():
    """Regression guard: the overwhelmingly common single-patch-per-file
    case must be byte-identical to pre-Phase-6E behavior -- no reordering,
    no dropping, no logging."""
    raw = json.dumps([
        {"file_path": "src/app.py", "code": "x = 1\n", "start_line": 1, "end_line": 2},
        {"file_path": "src/other.py", "code": "y = 2\n"},
    ])
    with patch("app.services.agent.graph.logger") as mock_logger:
        res = parse_and_validate_patches(raw)

    assert res == [
        {"file_path": "src/app.py", "code": "x = 1\n", "start_line": 1, "end_line": 2},
        {"file_path": "src/other.py", "code": "y = 2\n", "start_line": None, "end_line": None},
    ]
    mock_logger.warning.assert_not_called()


def test_parse_and_validate_patches_logs_warning_for_each_dropped_conflict():
    """Every dropped patch must be logged with enough information (file
    path and both patches' line ranges) to diagnose the conflict."""
    raw = json.dumps([
        {"file_path": "a.py", "code": "first\n", "start_line": 5, "end_line": 10},
        {"file_path": "a.py", "code": "second\n", "start_line": 8, "end_line": 12},
    ])
    with patch("app.services.agent.graph.logger") as mock_logger:
        res = parse_and_validate_patches(raw)

    assert len(res) == 1
    mock_logger.warning.assert_called_once()
    warning_message = mock_logger.warning.call_args[0][0]
    assert "a.py" in warning_message
    assert "8" in warning_message and "12" in warning_message
    assert "5" in warning_message and "10" in warning_message


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
        # Phase 6C: plan_node only calls Gemini for patches when patch
        # planning already produced a validated PLANNED plan -- this test
        # is specifically about plan_node's OWN patch-generation call, not
        # patch planning itself, so the gate is opened directly.
        "patch_plan_status": "PLANNED",
        "patch_plan": None,
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


def test_edit_node_applies_same_file_patches_correctly_after_conflict_resolution():
    """End-to-end proof the Phase 6E fix actually prevents corruption, not
    just that both patches are "kept": one patch's replacement changes the
    file's line count (a 1-line region becomes 3 lines), so applying the
    patches in the WRONG order would shift the second patch's target lines
    and corrupt the file. Routes raw Gemini-shaped JSON through
    parse_and_validate_patches (which reorders same-file patches
    descending by start_line) and then through edit_node against a real
    temporary file, asserting the ACTUAL final file content is correct."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target_file = workspace / "sample.py"
        target_file.write_text(
            "line 1\nold line 2\nline 3\nline 4\nline 5\nline 6\nold line 7\nline 8\n",
            encoding="utf-8",
        )

        # Emitted in top-to-bottom order, as an LLM naturally would --
        # applying them in THIS order would corrupt the file once the
        # first (line-count-changing) patch shifts everything below it.
        raw = json.dumps([
            {
                "file_path": "sample.py",
                "code": "fixed 2a\nfixed 2b\nfixed 2c\n",
                "start_line": 2,
                "end_line": 2,
            },
            {
                "file_path": "sample.py",
                "code": "fixed line 7\n",
                "start_line": 7,
                "end_line": 7,
            },
        ])
        resolved_patches = parse_and_validate_patches(raw, workspace_dir=str(workspace))
        # Reordered descending by start_line -- the line-7 patch applies
        # first, against the still-untouched file.
        assert [p["start_line"] for p in resolved_patches] == [7, 2]

        res = edit_node({
            "workspace_dir": str(workspace),
            "attempt_count": 0,
            "proposed_patches": resolved_patches,
        })
        assert res["applied_patch_count"] == 2

        lines = target_file.read_text(encoding="utf-8").splitlines()
        assert lines == [
            "line 1",
            "fixed 2a",
            "fixed 2b",
            "fixed 2c",
            "line 3",
            "line 4",
            "line 5",
            "line 6",
            "fixed line 7",
            "line 8",
        ]


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
        mock_client.models.generate_content.side_effect = [
            _not_applicable_baseline_response(),
            _no_evidence_diagnosis_response(),
            _planned_patch_plan_response(),
            response_attempt_1,
            _no_evidence_diagnosis_response(),
            _planned_patch_plan_response(),
            response_attempt_2,
        ]

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
        mock_client.models.generate_content.side_effect = [
            _not_applicable_baseline_response(),
            _no_evidence_diagnosis_response(),
            _planned_patch_plan_response(),
            resp1,
            _no_evidence_diagnosis_response(),
            _planned_patch_plan_response(),
            resp2,
        ]

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


@pytest.mark.asyncio
async def test_retrieval_ranks_relevant_files_first():
    """A relevant file must outrank alphabetically earlier irrelevant fillers.

    No repository_id is set in state, so Phase 6B's semantic retrieval
    short-circuits before ever touching a DB/retriever -- this proves
    ranking stays byte-identical to the pre-Phase-6B keyword-only
    behavior when semantic retrieval is inapplicable.
    """
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

        out = await retrieve_node({**state, **inv})
        retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]

        # Limit is still 5; the relevant file ranks first even though it sorts last
        assert retrieved_paths[0] == "zzz_vip.py"
        assert len(retrieved_paths) == 5
        assert "e_epsilon.py" not in retrieved_paths


@pytest.mark.asyncio
async def test_retrieval_ties_break_alphabetically():
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
        out = await retrieve_node({**state, **inv})
        retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]

        assert retrieved_paths == ["aaa_z.py", "zzz_a.py", "m_mid.py"]


@pytest.mark.asyncio
async def test_retrieval_without_matches_keeps_alphabetical_order():
    """No investigation data falls back to the previous alphabetical slice."""
    files = {
        "b_beta.py": "one\n",
        "a_alpha.py": "two\n",
        "c_gamma.py": "three\n",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_ranking_workspace(tmpdir, files)
        state = {"workspace_dir": str(workspace)}  # no task_description/keyword_matches

        out = await retrieve_node(state)
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


# ---------------------------------------------------------------------------
# Phase 3C fix #2: retrieved-context fences must match the file's real
# language, not be hardcoded to python for every file.
# ---------------------------------------------------------------------------
def test_patch_prompt_uses_language_aware_fence_for_non_python_file():
    mock_response = MagicMock()
    mock_response.text = json.dumps([])

    captured = {}

    def capture_generate_content(*args, **kwargs):
        captured.update(kwargs)
        return mock_response

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = capture_generate_content

    with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"):
        with patch("google.genai.Client", return_value=mock_client):
            _generate_patches_with_gemini(
                task_description="Fix the cart total display",
                retrieved_context=[
                    {
                        "file_path": "src/components/Cart.jsx",
                        "content": "function Cart() {\n  return null;\n}\n",
                        "total_lines": 3,
                    }
                ],
            )

    prompt = captured["contents"]
    assert "```javascript" in prompt
    assert "```python" not in prompt


def test_patch_prompt_fence_language_covers_multiple_ecosystems_and_unknown_fallback():
    mock_response = MagicMock()
    mock_response.text = json.dumps([])

    captured = {}

    def capture_generate_content(*args, **kwargs):
        captured.update(kwargs)
        return mock_response

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = capture_generate_content

    with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"):
        with patch("google.genai.Client", return_value=mock_client):
            _generate_patches_with_gemini(
                task_description="Fix the build",
                retrieved_context=[
                    {"file_path": "main.go", "content": "package main\n", "total_lines": 1},
                    {"file_path": "lib.rs", "content": "fn main() {}\n", "total_lines": 1},
                    {"file_path": "App.tsx", "content": "export default App;\n", "total_lines": 1},
                    {"file_path": "Makefile", "content": "build:\n\tgo build\n", "total_lines": 2},
                ],
            )

    prompt = captured["contents"]
    assert "### File: main.go (1 lines total)\n```go\n" in prompt
    assert "### File: lib.rs (1 lines total)\n```rust\n" in prompt
    assert "### File: App.tsx (1 lines total)\n```typescript\n" in prompt
    # No mapping entry for an extensionless file -- safe generic fallback,
    # never the raw (attacker-influenced) filename/extension text itself.
    assert "### File: Makefile (2 lines total)\n```text\n" in prompt
