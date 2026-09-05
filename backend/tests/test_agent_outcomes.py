"""Tests for the agent's real-world outcome classification: FIXED,
NO_CHANGE_NEEDED, UNABLE_TO_VERIFY, and FAILED. These are distinct from the
mechanical `is_verified`/`status` retry-loop fields -- see
app.services.agent.graph.finalize_node.

The user's task description is a hypothesis, never ground truth: several
tests here deliberately describe a bug that doesn't exist, or name the wrong
file, and assert the agent does not fabricate a patch just to satisfy the
workflow.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.agent.graph import agent_app, edit_node, finalize_node


def _not_applicable_baseline_response() -> MagicMock:
    """The full graph now runs a real (mocked, per-test) baseline
    reproduction planning call between investigate and retrieve -- see
    app.services.agent.graph.baseline_node. A genuine NOT_APPLICABLE
    response keeps it a no-op for outcome tests that aren't specifically
    exercising baseline gating (see test_baseline_gating_*.py-style tests
    below that DO)."""
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
    app.services.agent.graph.diagnose_node. A genuine no-evidence-shaped
    response keeps it advisory-inert for outcome tests that aren't
    specifically exercising diagnosis behavior itself.

    Note: this still parses into a DIAGNOSED diagnosis (empty hypotheses,
    confidence "no_evidence") -- only a parse/network failure produces
    DIAGNOSIS_FAILED. So patch_plan_node (Phase 6C) still attempts its own
    Gemini call after this -- see _planned_patch_plan_response() below.
    """
    response = MagicMock()
    response.text = json.dumps({"summary": "", "hypotheses": [], "confidence": "no_evidence"})
    return response


def _planned_patch_plan_response() -> MagicMock:
    """The full graph now also runs a real (mocked, per-test) patch-planning
    Gemini call on every pass between diagnose and plan -- see
    app.services.agent.graph.patch_plan_node. A genuine, minimal PLANNED
    response opens plan_node's allow-list gate so the subsequent patch-
    generation call is still reached, for outcome tests that aren't
    specifically exercising patch-planning behavior itself."""
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


def _base_state(workspace: Path, task_description: str, **overrides) -> dict:
    state = {
        "task_id": 1,
        "repository_id": 1,
        "workspace_dir": str(workspace),
        "task_description": task_description,
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
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# finalize_node unit tests (fast, no graph execution)
# ---------------------------------------------------------------------------
def test_finalize_node_unable_to_verify_when_verification_tooling_unavailable():
    state = {
        "test_results": {"available": False, "detail": "gradle not found"},
        "proposed_patches": [],
        "is_verified": False,
    }
    out = finalize_node(state)
    assert out["outcome"] == "UNABLE_TO_VERIFY"
    assert "gradle not found" in out["outcome_detail"]


def test_finalize_node_no_change_needed_when_verified_without_patches():
    state = {
        "test_results": {"available": True, "success": True},
        "proposed_patches": [],
        "is_verified": True,
    }
    out = finalize_node(state)
    assert out["outcome"] == "NO_CHANGE_NEEDED"


def test_finalize_node_fixed_when_verified_with_patches():
    state = {
        "test_results": {"available": True, "success": True},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "is_verified": True,
    }
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


def test_finalize_node_failed_when_not_verified_and_tooling_available():
    state = {
        "test_results": {"available": True, "success": False},
        "proposed_patches": [],
        "is_verified": False,
    }
    out = finalize_node(state)
    assert out["outcome"] == "FAILED"


def test_finalize_node_unable_to_verify_takes_priority_over_verified_flag():
    """A missing-tool signal must never be shadowed by any other field --
    it is not evidence the behavior is correct or incorrect."""
    state = {
        "test_results": {"available": False, "detail": "flutter not found"},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "is_verified": False,
    }
    out = finalize_node(state)
    assert out["outcome"] == "UNABLE_TO_VERIFY"


# ---------------------------------------------------------------------------
# Full graph: reported bug already fixed / cannot be substantiated -> NO_CHANGE_NEEDED
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outcome_no_change_needed_when_claimed_bug_already_fixed():
    """The task claims a bug that doesn't actually exist in the code; the
    test-environment Gemini key (`test_gemini_key_123`) short-circuits patch
    generation to an empty list, exactly like the model concluding the
    behavior is already correct. No patch must be fabricated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "cart.py").write_text(
            "def total(items):\n    return sum(item.price for item in items)\n",
            encoding="utf-8",
        )
        (workspace / "test_cart.py").write_text(
            "from cart import total\n"
            "class Item:\n"
            "    def __init__(self, price):\n"
            "        self.price = price\n"
            "def test_total():\n"
            "    assert total([Item(2), Item(3)]) == 5\n",
            encoding="utf-8",
        )

        initial_state = _base_state(
            workspace,
            "The cart total is calculating incorrectly, please fix the total() function.",
        )

        final_state = await agent_app.ainvoke(initial_state)

        assert final_state["outcome"] == "NO_CHANGE_NEEDED"
        assert final_state["is_verified"] is True
        assert final_state["proposed_patches"] == []


@pytest.mark.asyncio
async def test_outcome_no_change_needed_when_user_names_wrong_file():
    """Task names a file that isn't actually responsible for the behavior;
    the real implementation is correct, so no patch should be invented
    against the (incorrectly identified) file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "pricing.py").write_text(
            "def discount(price):\n    return price * 0.9\n",
            encoding="utf-8",
        )
        (workspace / "ProductCard.py").write_text("# unrelated presentational stub\n", encoding="utf-8")
        (workspace / "test_pricing.py").write_text(
            "from pricing import discount\n"
            "def test_discount():\n"
            "    assert discount(100) == 90\n",
            encoding="utf-8",
        )

        initial_state = _base_state(
            workspace,
            "Fix the discount bug in ProductCard.py, it shows the wrong price.",
        )

        final_state = await agent_app.ainvoke(initial_state)

        assert final_state["outcome"] == "NO_CHANGE_NEEDED"
        assert final_state["proposed_patches"] == []
        # The misidentified file must not have been touched.
        assert (
            (workspace / "ProductCard.py").read_text(encoding="utf-8")
            == "# unrelated presentational stub\n"
        )


# ---------------------------------------------------------------------------
# Full graph: actual bug confirmed and fixed -> FIXED
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outcome_fixed_when_bug_confirmed_and_patch_verified():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        code_file = workspace / "math_lib.py"
        code_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (workspace / "test_math.py").write_text(
            "from math_lib import add\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        mock_response = MagicMock()
        mock_response.text = json.dumps([
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
            mock_response,
        ]

        initial_state = _base_state(workspace, "The add() function in math_lib.py returns the wrong result.")

        with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"):
            with patch("google.genai.Client", return_value=mock_client):
                final_state = await agent_app.ainvoke(initial_state)

        assert final_state["outcome"] == "FIXED"
        assert final_state["is_verified"] is True
        assert len(final_state["proposed_patches"]) == 1
        assert "return a + b" in code_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Full graph: exhausted retry budget on a real, unfixed bug -> FAILED
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outcome_failed_when_attempts_exhausted_on_real_bug():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "test_fail.py").write_text(
            "def test_bad():\n    assert 1 == 2\n",
            encoding="utf-8",
        )

        initial_state = _base_state(workspace, "Failing test task")

        final_state = await agent_app.ainvoke(initial_state)

        assert final_state["outcome"] == "FAILED"
        assert final_state["is_verified"] is False
        assert final_state["attempt_count"] == 3


# ---------------------------------------------------------------------------
# Full graph: unsupported ecosystem -> UNABLE_TO_VERIFY (never a false pass)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outcome_unable_to_verify_for_unsupported_ecosystem():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "README.md").write_text("Just documentation, no build system here.\n", encoding="utf-8")
        (workspace / "notes.txt").write_text("nothing recognizable\n", encoding="utf-8")

        initial_state = _base_state(workspace, "Fix the formatting in the docs")

        final_state = await agent_app.ainvoke(initial_state)

        assert final_state["outcome"] == "UNABLE_TO_VERIFY"
        assert final_state["test_results"]["available"] is False
        # Must never be reported as a confirmed pass or fail.
        assert final_state["is_verified"] is False


# ---------------------------------------------------------------------------
# Full graph: detected ecosystem, but required tool missing -> UNABLE_TO_VERIFY
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outcome_unable_to_verify_when_required_tool_missing():
    """'gradle not found' must be UNABLE_TO_VERIFY, never 'bug does not exist'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")

        initial_state = _base_state(workspace, "Fix the failing Gradle unit test")

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("[Errno 2] No such file or directory: 'gradle'")

        with patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            final_state = await agent_app.ainvoke(initial_state)

        assert final_state["outcome"] == "UNABLE_TO_VERIFY"
        assert final_state["test_results"]["ecosystem"] == "java-gradle"
        assert final_state["is_verified"] is False


# ---------------------------------------------------------------------------
# Phase 3C fix #3: zero-applied-patch information must be surfaced, never
# silently discarded -- a task where every generated patch fails to apply
# must never be reported identically to one where real edits were made and
# verification simply failed, and must never be reported as FIXED.
# ---------------------------------------------------------------------------
def test_edit_node_reports_zero_applied_patch_count_when_all_apply_patch_calls_fail():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")

        state = _base_state(
            workspace,
            "Fix a.py",
            proposed_patches=[
                # Way beyond the 1-line file -- tools.apply_patch's own
                # bounds check rejects this, so it never applies.
                {"file_path": "a.py", "code": "x = 2\n", "start_line": 50, "end_line": 51},
            ],
        )

        out = edit_node(state)

        assert out["applied_patch_count"] == 0


def test_finalize_node_verified_with_zero_applied_patches_is_no_change_needed_not_fixed():
    """Patches were generated and verification passed, but none of the
    patches actually applied -- the code was never really changed, so this
    must never be reported as FIXED."""
    state = {
        "test_results": {"available": True, "success": True},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 0,
        "is_verified": True,
    }
    out = finalize_node(state)

    assert out["outcome"] == "NO_CHANGE_NEEDED"
    assert out["outcome"] != "FIXED"
    assert "none could be applied" in out["outcome_detail"]


def test_finalize_node_failed_detail_distinguishes_zero_applied_from_real_edit_failure():
    """A FAILED outcome where nothing was ever actually applied must read
    differently from a FAILED outcome where real edits were made and
    verification still failed -- otherwise the two are indistinguishable to
    a human reviewer."""
    zero_applied_state = {
        "test_results": {"available": True, "success": False},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 2\n"}],
        "applied_patch_count": 0,
        "is_verified": False,
    }
    real_edit_failed_state = {
        "test_results": {"available": True, "success": False},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 2\n"}],
        "applied_patch_count": 1,
        "is_verified": False,
    }

    zero_applied_out = finalize_node(zero_applied_state)
    real_edit_out = finalize_node(real_edit_failed_state)

    assert zero_applied_out["outcome"] == "FAILED"
    assert real_edit_out["outcome"] == "FAILED"
    assert zero_applied_out["outcome_detail"] != real_edit_out["outcome_detail"]
    assert "none could be applied" in zero_applied_out["outcome_detail"]
    assert "none could be applied" not in real_edit_out["outcome_detail"]


def test_finalize_node_applied_patch_count_absent_preserves_prior_behavior():
    """A caller (e.g. a hand-constructed state, matching every finalize_node
    test that predates this field) that never sets applied_patch_count at
    all must see byte-identical behavior to before this fix -- only an
    explicit 0 downgrades an otherwise-FIXED result."""
    state = {
        "test_results": {"available": True, "success": True},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "is_verified": True,
    }
    out = finalize_node(state)

    assert out["outcome"] == "FIXED"
