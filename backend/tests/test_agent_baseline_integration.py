"""Integration tests for Phase 4B-3: evidence-driven baseline reproduction
wired into the agent's LangGraph workflow (app.services.agent.graph.
baseline_node, and finalize_node's baseline-aware outcome gating).

No real Gemini/Docker/GitHub calls anywhere -- app.services.baseline's
planner/bridge/executor entry points (plan_reproduction, build_
reproduction_input, reproduce) are patched directly at their
app.services.agent.graph import bindings, since this suite is about the
INTEGRATION contract (how baseline_status feeds finalize_node, when the
Phase 4A executor is/isn't invoked), not about planner/bridge/executor
internals -- those already have their own dedicated test suites
(test_baseline_planner.py, test_baseline_bridge.py, test_baseline.py),
which this suite does not duplicate or weaken.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.agent.graph import (
    agent_app,
    baseline_node,
    finalize_node,
)
from app.services.baseline import (
    BaselineResult,
    BaselineStatus,
    BridgeOutcome,
    ExitCodeSemantics,
    PlanBridgeResult,
    ReproductionExpectation,
    ReproductionInput,
    ReproductionPlan,
    ReproductionType,
)


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
        "keyword_matches": [],
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


def _applicable_plan(**overrides) -> ReproductionPlan:
    defaults = dict(
        applicable=True,
        reason="tests/test_cart.py::test_subtotal fails today because of this bug",
        reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["python", "repro.py"]],
        expected_observation="The cart subtotal test fails.",
        exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED,
    )
    defaults.update(overrides)
    return ReproductionPlan(**defaults)


def _executable_bridge_result(workspace: str) -> PlanBridgeResult:
    return PlanBridgeResult(
        outcome=BridgeOutcome.EXECUTABLE,
        reproduction_input=ReproductionInput(
            workspace_path=workspace,
            commands=[["python", "repro.py"]],
            expectation=ReproductionExpectation(exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED),
        ),
        detail="ok",
    )


def _patched_baseline(plan_reproduction_return=None, bridge_return=None, reproduce_return=None):
    """Patch the three Phase 4B entry points exactly as baseline_node
    imports them, as a context-manager tuple. `plan_reproduction` is
    `async def`, so `unittest.mock.patch` auto-creates an `AsyncMock` for
    it -- `return_value` must be the plain resolved value, not a coroutine
    (AsyncMock already makes `await mock(...)` resolve to `return_value`
    directly)."""
    return (
        patch("app.services.agent.graph.plan_reproduction", return_value=plan_reproduction_return),
        patch("app.services.agent.graph.build_reproduction_input", return_value=bridge_return),
        patch("app.services.agent.graph.reproduce", return_value=reproduce_return),
    )


# ===========================================================================
# 1. Strong evidence + valid plan + successful reproduction -> REPRODUCED
# ===========================================================================
@pytest.mark.asyncio
async def test_baseline_node_reproduced():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _base_state(workspace, "The cart subtotal is wrong for VIP customers")
        plan = _applicable_plan()
        bridge_result = _executable_bridge_result(str(workspace))
        reproduce_result = BaselineResult(
            status=BaselineStatus.REPRODUCED,
            detail="The reproduction pattern was observed.",
            exit_code=1,
        )

        p1, p2, p3 = _patched_baseline(plan, bridge_result, reproduce_result)
        with p1, p2, p3 as mock_reproduce:
            result = await baseline_node(state)

        assert result["baseline_status"] == "REPRODUCED"
        assert result["baseline_result"]["status"] == "REPRODUCED"
        mock_reproduce.assert_called_once()


# ===========================================================================
# 2. Valid plan executes but expected failure absent -> NOT_REPRODUCED
# ===========================================================================
@pytest.mark.asyncio
async def test_baseline_node_not_reproduced():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _base_state(workspace, "The cart subtotal is wrong for VIP customers")
        plan = _applicable_plan()
        bridge_result = _executable_bridge_result(str(workspace))
        reproduce_result = BaselineResult(
            status=BaselineStatus.NOT_REPRODUCED,
            detail="The reproduction executed successfully; no evidence of the reported bug.",
            exit_code=0,
        )

        p1, p2, p3 = _patched_baseline(plan, bridge_result, reproduce_result)
        with p1, p2, p3:
            result = await baseline_node(state)

        assert result["baseline_status"] == "NOT_REPRODUCED"
        assert result["baseline_result"]["status"] == "NOT_REPRODUCED"


# ===========================================================================
# 3. Planner planning_failed=True -> UNABLE_TO_REPRODUCE, Phase 4A executor
# (reproduce()) is not called.
# ===========================================================================
@pytest.mark.asyncio
async def test_baseline_node_planning_failed_is_unable_to_reproduce_and_skips_executor():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _base_state(workspace, "Some bug report")
        plan = ReproductionPlan(
            applicable=False,
            reason="Gemini timed out",
            reproduction_type=ReproductionType.NOT_APPLICABLE,
            planning_failed=True,
            failure_reason="Gemini timed out",
        )

        p1, p2, p3 = _patched_baseline(plan, None, None)
        with p1, p2, p3 as mock_reproduce:
            result = await baseline_node(state)

        assert result["baseline_status"] == "UNABLE_TO_REPRODUCE"
        assert result["baseline_result"] is None
        assert "Gemini timed out" in result["baseline_detail"]
        mock_reproduce.assert_not_called()


# ===========================================================================
# 4. Genuine NOT_APPLICABLE -> NOT_APPLICABLE, executor not called.
# ===========================================================================
@pytest.mark.asyncio
async def test_baseline_node_genuine_not_applicable_skips_executor():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _base_state(workspace, "Some vague request with no reproducible behavior")
        plan = ReproductionPlan(
            applicable=False,
            reason="No repository evidence supports any reproduction.",
            reproduction_type=ReproductionType.NOT_APPLICABLE,
            planning_failed=False,
        )

        p1, p2, p3 = _patched_baseline(plan, None, None)
        with p1, p2, p3 as mock_reproduce:
            result = await baseline_node(state)

        assert result["baseline_status"] == "NOT_APPLICABLE"
        assert result["baseline_result"] is None
        mock_reproduce.assert_not_called()


# ===========================================================================
# 5. Bridge PLANNING_FAILED -> UNABLE_TO_REPRODUCE, executor not called.
# ===========================================================================
@pytest.mark.asyncio
async def test_baseline_node_bridge_planning_failed_is_unable_to_reproduce_and_skips_executor():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _base_state(workspace, "Some bug report")
        plan = _applicable_plan()
        bridge_result = PlanBridgeResult(
            outcome=BridgeOutcome.PLANNING_FAILED,
            detail="working_dir escapes the workspace",
        )

        p1, p2, p3 = _patched_baseline(plan, bridge_result, None)
        with p1, p2, p3 as mock_reproduce:
            result = await baseline_node(state)

        assert result["baseline_status"] == "UNABLE_TO_REPRODUCE"
        assert result["baseline_result"] is None
        assert "working_dir" in result["baseline_detail"]
        mock_reproduce.assert_not_called()


# ===========================================================================
# 6. Executor/tool/environment failure -> UNABLE_TO_REPRODUCE, never
# converted to NOT_REPRODUCED or NOT_APPLICABLE.
# ===========================================================================
@pytest.mark.asyncio
async def test_baseline_node_executor_environment_failure_is_unable_to_reproduce():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _base_state(workspace, "Some bug report")
        plan = _applicable_plan()
        bridge_result = _executable_bridge_result(str(workspace))
        reproduce_result = BaselineResult(
            status=BaselineStatus.UNABLE_TO_REPRODUCE,
            detail="Required tool 'npm' is not available in this environment.",
        )

        p1, p2, p3 = _patched_baseline(plan, bridge_result, reproduce_result)
        with p1, p2, p3:
            result = await baseline_node(state)

        assert result["baseline_status"] == "UNABLE_TO_REPRODUCE"
        assert result["baseline_status"] != "NOT_REPRODUCED"
        assert result["baseline_status"] != "NOT_APPLICABLE"


# ===========================================================================
# 7. User wording conflicts with repository detection -> repository
# evidence/detection remains authoritative (RepositoryEvidence is built from
# REAL detected projects regardless of what the task description claims).
# ===========================================================================
@pytest.mark.asyncio
async def test_repository_evidence_reflects_real_detection_not_user_wording():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Real repository evidence: a Python project (pyproject.toml).
        (workspace / "pyproject.toml").write_text(
            "[project]\nname = 'demo'\n", encoding="utf-8"
        )
        (workspace / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

        # User claims React/Node -- must not override real detection.
        state = _base_state(workspace, "This React button component is broken, please fix the React state bug.")

        captured_evidence = {}

        async def _capture_plan(task_description, evidence):
            captured_evidence["evidence"] = evidence
            return ReproductionPlan(
                applicable=False,
                reason="no reproduction needed for this test",
                reproduction_type=ReproductionType.NOT_APPLICABLE,
            )

        with patch("app.services.agent.graph.plan_reproduction", side_effect=_capture_plan):
            await baseline_node(state)

        evidence = captured_evidence["evidence"]
        ecosystems = {p.ecosystem for p in evidence.detected_projects}
        assert "python" in ecosystems
        assert "node" not in ecosystems
        assert "react" not in {e.lower() for e in ecosystems}


# ===========================================================================
# 8. Existing task workflow still reaches normal fix/verification behavior
# when baseline is REPRODUCED (full graph run).
# ===========================================================================
@pytest.mark.asyncio
async def test_full_graph_reaches_fixed_when_baseline_reproduced_and_fix_verified():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        code_file = workspace / "math_lib.py"
        code_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (workspace / "test_math.py").write_text(
            "from math_lib import add\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        # Phase 6C: diagnose_node and patch_plan_node each also run a real
        # (mocked) Gemini call between retrieve and plan_node's own patch-
        # generation call -- genuine, parseable responses for both are
        # required so plan_node's PLANNED-only allow-list gate actually
        # opens and the real fix below is still generated.
        diagnosis_response = MagicMock()
        diagnosis_response.text = json.dumps({
            "summary": "add() subtracts instead of adding.",
            "hypotheses": [
                {
                    "rank": 1,
                    "description": "Wrong operator in add().",
                    "citations": [{"file_path": "math_lib.py", "start_line": 1, "end_line": 2, "symbol_name": "add"}],
                }
            ],
            "confidence": "inferred",
        })
        patch_plan_response = MagicMock()
        patch_plan_response.text = json.dumps({
            "applicable": True,
            "summary": "Fix the addition operator.",
            "changes": [
                {
                    "file_path": "math_lib.py",
                    "change_type": "modify",
                    "description": "Change the subtraction operator to addition.",
                    "rationale": "Matches the diagnosed cause.",
                    "citations": [{"file_path": "math_lib.py", "start_line": 1, "end_line": 2, "symbol_name": "add"}],
                    "symbols_affected": ["add"],
                }
            ],
            "diagnosis_alignment": "Directly addresses the diagnosed operator bug.",
            "confidence": "inferred",
        })
        patch_response = MagicMock()
        patch_response.text = json.dumps([
            {"file_path": "math_lib.py", "code": "def add(a, b):\n    return a + b\n", "start_line": 1, "end_line": 2}
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [diagnosis_response, patch_plan_response, patch_response]

        plan = _applicable_plan()
        bridge_result = _executable_bridge_result(str(workspace))
        # Phase 5: reproduce() is called twice -- once by baseline_node
        # (pre-fix, REPRODUCED) and once by post_fix_reproduction_node
        # (post-fix, must confirm the failure is gone) -- side_effect
        # supplies the two calls in that exact order.
        baseline_reproduce_result = BaselineResult(status=BaselineStatus.REPRODUCED, detail="reproduced", exit_code=1)
        post_fix_reproduce_result = BaselineResult(
            status=BaselineStatus.NOT_REPRODUCED, detail="no longer reproduces", exit_code=0
        )

        initial_state = _base_state(workspace, "The add() function in math_lib.py returns the wrong result.")

        with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"), patch(
            "google.genai.Client", return_value=mock_client
        ), patch("app.services.agent.graph.plan_reproduction", return_value=plan), patch(
            "app.services.agent.graph.build_reproduction_input", return_value=bridge_result
        ), patch(
            "app.services.agent.graph.reproduce",
            side_effect=[baseline_reproduce_result, post_fix_reproduce_result],
        ):
            final_state = await agent_app.ainvoke(initial_state)

        assert final_state["baseline_status"] == "REPRODUCED"
        assert final_state["post_fix_reproduction_status"] == "NOT_REPRODUCED"
        assert final_state["outcome"] == "FIXED"
        assert "return a + b" in code_file.read_text(encoding="utf-8")


# ===========================================================================
# 9. Baseline failure cannot produce a false FIXED/NO_CHANGE_NEEDED outcome
# (direct finalize_node tests -- a patch was generated AND applied, and full
# verification passed, exactly the conditions that would otherwise say
# FIXED).
# ===========================================================================
def _verified_with_applied_patch_state(**overrides) -> dict:
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": True,
    }
    state.update(overrides)
    return state


def test_finalize_node_unable_to_reproduce_baseline_never_yields_fixed_or_no_change_needed():
    state = _verified_with_applied_patch_state(
        baseline_status="UNABLE_TO_REPRODUCE",
        baseline_detail="Gemini was unavailable.",
    )
    out = finalize_node(state)
    assert out["outcome"] != "FIXED"
    assert out["outcome"] != "NO_CHANGE_NEEDED"
    assert out["outcome"] == "UNABLE_TO_VERIFY"


def test_unable_to_reproduce_and_not_reproduced_share_outcome_but_stay_distinguishable_in_detail():
    """Both baseline_status values conservatively map to the same outcome
    category, but the two must remain distinguishable through
    baseline_status/outcome_detail -- never silently merged into one
    indistinguishable signal."""
    unable_state = _verified_with_applied_patch_state(
        baseline_status="UNABLE_TO_REPRODUCE", baseline_detail="planner failure detail"
    )
    not_reproduced_state = _verified_with_applied_patch_state(
        baseline_status="NOT_REPRODUCED", baseline_detail="no evidence observed detail"
    )

    unable_out = finalize_node(unable_state)
    not_reproduced_out = finalize_node(not_reproduced_state)

    assert unable_out["outcome"] == not_reproduced_out["outcome"] == "UNABLE_TO_VERIFY"
    assert unable_out["outcome_detail"] != not_reproduced_out["outcome_detail"]
    assert "planner failure detail" in unable_out["outcome_detail"]
    assert "no evidence observed detail" in not_reproduced_out["outcome_detail"]


def test_finalize_node_not_reproduced_baseline_never_yields_fixed_or_no_change_needed():
    """NOT_REPRODUCED means the reproduction executed successfully and did
    not observe the reported behavior -- it does NOT mean "verified as
    already correct", and NO_CHANGE_NEEDED elsewhere in finalize_node
    specifically means "no code change was actually made", which is false
    here (a patch WAS applied). UNABLE_TO_VERIFY is the correct, most
    conservative existing outcome: it claims neither that the fix is
    correct nor that no fix was needed."""
    state = _verified_with_applied_patch_state(
        baseline_status="NOT_REPRODUCED",
        baseline_detail="No evidence of the reported bug was observed.",
    )
    out = finalize_node(state)
    assert out["outcome"] != "FIXED"
    assert out["outcome"] != "NO_CHANGE_NEEDED"
    assert out["outcome"] == "UNABLE_TO_VERIFY"


def test_finalize_node_reproduced_baseline_allows_fixed():
    """Phase 5: a REPRODUCED baseline additionally requires the targeted
    post-fix reproduction to have positively confirmed the failure is gone
    before FIXED can be claimed."""
    state = _verified_with_applied_patch_state(
        baseline_status="REPRODUCED",
        post_fix_reproduction_status="NOT_REPRODUCED",
    )
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


def test_finalize_node_reproduced_baseline_without_post_fix_confirmation_never_yields_fixed():
    """A REPRODUCED baseline with NO post-fix confirmation at all (e.g.
    post_fix_reproduction_node never ran for this attempt) must not default
    to FIXED just because the general test suite passed."""
    state = _verified_with_applied_patch_state(baseline_status="REPRODUCED")
    out = finalize_node(state)
    assert out["outcome"] != "FIXED"
    assert out["outcome"] == "UNABLE_TO_VERIFY"


def test_finalize_node_not_applicable_baseline_allows_fixed():
    """NOT_APPLICABLE is not the same thing as UNABLE_TO_REPRODUCE -- when
    there was genuinely no meaningful reproduction to attempt, existing
    fix/verification behavior is unaffected."""
    state = _verified_with_applied_patch_state(baseline_status="NOT_APPLICABLE")
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


def test_finalize_node_absent_baseline_status_preserves_prior_behavior():
    """A hand-constructed state with no baseline_status at all (e.g. a task
    predating this integration) behaves exactly as before Phase 4B-3."""
    state = _verified_with_applied_patch_state()
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


def test_finalize_node_baseline_gate_does_not_affect_no_change_needed_branch():
    """A baseline failure must not need to be checked at all when the
    existing logic already wouldn't claim FIXED (zero patches applied)."""
    state = _verified_with_applied_patch_state(
        proposed_patches=[{"file_path": "a.py", "code": "x = 1\n"}],
        applied_patch_count=0,
        baseline_status="UNABLE_TO_REPRODUCE",
    )
    out = finalize_node(state)
    assert out["outcome"] == "NO_CHANGE_NEEDED"


def test_finalize_node_baseline_gate_does_not_affect_failed_branch():
    state = {
        "test_results": {"available": True, "success": False, "failed": 1},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": False,
        "baseline_status": "UNABLE_TO_REPRODUCE",
    }
    out = finalize_node(state)
    assert out["outcome"] == "FAILED"


# ===========================================================================
# Phase 6D: a REPRODUCED baseline is positive evidence the reported bug is
# real -- it must never be silently overridden by the "no real code change"
# NO_CHANGE_NEEDED branches, which would otherwise claim the opposite of
# what baseline reproduction just established.
# ===========================================================================
def test_finalize_node_reproduced_baseline_with_zero_patches_is_failed_not_no_change_needed():
    """No patches were generated this attempt, but baseline independently
    proved the reported bug is real -- NO_CHANGE_NEEDED would falsely claim
    the behavior was already correct or unsubstantiated. FAILED is correct:
    the issue is real and was not fixed this attempt."""
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [],
        "is_verified": True,
        "baseline_status": "REPRODUCED",
    }
    out = finalize_node(state)
    assert out["outcome"] == "FAILED"
    assert out["outcome"] != "NO_CHANGE_NEEDED"
    assert out["outcome"] != "FIXED"
    assert "no patch was generated" in out["outcome_detail"]


def test_finalize_node_reproduced_baseline_with_zero_applied_patches_is_failed_not_no_change_needed():
    """Patches were generated but none actually applied -- same masking bug
    as the zero-patches case, via the OTHER "no real code change" branch."""
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 0,
        "is_verified": True,
        "baseline_status": "REPRODUCED",
    }
    out = finalize_node(state)
    assert out["outcome"] == "FAILED"
    assert out["outcome"] != "NO_CHANGE_NEEDED"
    assert out["outcome"] != "FIXED"
    assert "none could be applied" in out["outcome_detail"]


def test_finalize_node_reproduced_baseline_zero_patches_with_post_fix_reproduced_is_failed():
    """The realistic full scenario: baseline REPRODUCED, no patch generated,
    the general test suite still passes, and post-fix reproduction reruns
    the identical check against the unchanged workspace and reconfirms the
    bug is still there. post_fix_reproduction_status must not need to be
    consulted for this to correctly resolve to FAILED -- there is no
    applied change for it to be evidence about."""
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [],
        "is_verified": True,
        "baseline_status": "REPRODUCED",
        "post_fix_reproduction_status": "REPRODUCED",
        "post_fix_reproduction_detail": "the same failure was observed again",
    }
    out = finalize_node(state)
    assert out["outcome"] == "FAILED"


def test_finalize_node_applied_patch_count_none_with_baseline_reproduced_unaffected_by_new_branch():
    """Regression guard: applied_patch_count absent (unknown, not zero) with
    a REPRODUCED baseline and a real, confirmed fix must still reach FIXED
    -- the new Phase 6D branch's zero_applied/not-patches condition must
    never fire just because applied_patch_count is unset."""
    state = _verified_with_applied_patch_state(
        baseline_status="REPRODUCED",
        post_fix_reproduction_status="NOT_REPRODUCED",
        applied_patch_count=None,
    )
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"
