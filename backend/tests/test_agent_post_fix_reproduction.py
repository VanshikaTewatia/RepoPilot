"""Integration tests for Phase 5: targeted post-fix reproduction
(app.services.agent.graph.post_fix_reproduction_node, and finalize_node's
further REPRODUCED-baseline gating).

No real Gemini/Docker/GitHub calls -- app.services.baseline's planner/
bridge/executor entry points are patched directly at their
app.services.agent.graph import bindings, exactly as
test_agent_baseline_integration.py already does. This suite is about the
Phase 5 integration contract specifically (reusing the exact reproduction
spec after an edit, never re-planning, never a second workspace/executor)
and does not duplicate Phase 4A/4B-1/4B-2's own dedicated suites.
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
    post_fix_reproduction_node,
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
        commands=[["python", "repro.py", "--check"]],
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
            commands=[["python", "repro.py", "--check"]],
            working_dir="backend",
            timeout_seconds=45,
            image="python:3.11-slim",
            expectation=ReproductionExpectation(
                exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED,
                reproduced_output_pattern=r"BUG: .*",
                not_reproduced_output_pattern=None,
            ),
            task_context="The cart subtotal test fails.",
        ),
        detail="ok",
    )


def _reproduced_state(workspace: Path, **overrides) -> dict:
    """A state as it would look right after baseline_node ran with a
    REPRODUCED result and captured reproduction_spec, then edit_node/
    test_node/verify_node ran and this attempt's own verification passed."""
    bridge_result = _executable_bridge_result(str(workspace))
    from app.services.agent.graph import _reproduction_input_to_state_dict

    state = _base_state(
        workspace,
        "The cart subtotal is wrong for VIP customers",
        baseline_status="REPRODUCED",
        reproduction_spec=_reproduction_input_to_state_dict(bridge_result.reproduction_input),
        proposed_patches=[{"file_path": "a.py", "code": "x = 1\n"}],
        applied_patch_count=1,
        is_verified=True,
        test_results={"available": True, "success": True, "failed": 0},
    )
    state.update(overrides)
    return state


async def _run_post_fix(state, reproduce_return=None):
    with patch("app.services.agent.graph.reproduce", return_value=reproduce_return) as mock_reproduce:
        result = await post_fix_reproduction_node(state)
    return result, mock_reproduce


# ===========================================================================
# 1. Post-fix no longer reproduces -> recorded, FIXED remains possible
# ===========================================================================
@pytest.mark.asyncio
async def test_post_fix_not_reproduced_is_recorded_and_allows_fixed_after_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace)
        reproduce_result = BaselineResult(status=BaselineStatus.NOT_REPRODUCED, detail="no longer reproduces", exit_code=0)

        result, mock_reproduce = await _run_post_fix(state, reproduce_result)

        assert result["post_fix_reproduction_status"] == "NOT_REPRODUCED"
        mock_reproduce.assert_called_once()

        final_state = {**state, **result}
        out = finalize_node(final_state)
        assert out["outcome"] == "FIXED"


# ===========================================================================
# 2. Post-fix still reproduces -> never FIXED
# ===========================================================================
@pytest.mark.asyncio
async def test_post_fix_still_reproduced_never_yields_fixed():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace)
        reproduce_result = BaselineResult(status=BaselineStatus.REPRODUCED, detail="still failing", exit_code=1)

        result, _ = await _run_post_fix(state, reproduce_result)
        assert result["post_fix_reproduction_status"] == "REPRODUCED"

        final_state = {**state, **result}
        out = finalize_node(final_state)
        assert out["outcome"] != "FIXED"
        assert out["outcome"] == "FAILED"


# ===========================================================================
# 3. Post-fix execution failure -> never FIXED
# ===========================================================================
@pytest.mark.asyncio
async def test_post_fix_execution_failure_never_yields_fixed():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace)
        reproduce_result = BaselineResult(status=BaselineStatus.UNABLE_TO_REPRODUCE, detail="toolchain missing")

        result, _ = await _run_post_fix(state, reproduce_result)
        assert result["post_fix_reproduction_status"] == "UNABLE_TO_REPRODUCE"

        final_state = {**state, **result}
        out = finalize_node(final_state)
        assert out["outcome"] != "FIXED"
        assert out["outcome"] == "UNABLE_TO_VERIFY"


# ===========================================================================
# 4/5/6/7. No post-fix reproduction for any baseline status other than
# REPRODUCED (including planning_failed, which baseline_node already
# surfaces as baseline_status == "UNABLE_TO_REPRODUCE").
# ===========================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["NOT_REPRODUCED", "UNABLE_TO_REPRODUCE", "NOT_APPLICABLE"])
async def test_no_post_fix_reproduction_for_non_reproduced_baseline(status):
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace, baseline_status=status)

        result, mock_reproduce = await _run_post_fix(state, None)

        assert result == {}
        mock_reproduce.assert_not_called()


@pytest.mark.asyncio
async def test_no_post_fix_reproduction_when_this_attempts_verification_failed():
    """Even with a REPRODUCED baseline and a captured spec, if THIS
    attempt's own test/verify did not pass, post-fix reproduction must not
    run -- the attempt is already headed back to analyze_failure (or a
    FAILED finalize) regardless."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace, is_verified=False)

        result, mock_reproduce = await _run_post_fix(state, None)

        assert result == {}
        mock_reproduce.assert_not_called()


@pytest.mark.asyncio
async def test_no_post_fix_reproduction_when_reproduction_spec_absent():
    """A REPRODUCED baseline_status with no reproduction_spec at all (should
    not happen in practice, but defensively) must not attempt anything."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace, reproduction_spec=None)

        result, mock_reproduce = await _run_post_fix(state, None)

        assert result == {}
        mock_reproduce.assert_not_called()


# ===========================================================================
# 8. Exact same reproduction specification is reused: same commands, order,
# working_dir, image, expectation, timeout.
# ===========================================================================
@pytest.mark.asyncio
async def test_post_fix_reuses_the_exact_same_reproduction_specification():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace)
        reproduce_result = BaselineResult(status=BaselineStatus.NOT_REPRODUCED, detail="ok", exit_code=0)

        with patch("app.services.agent.graph.reproduce", return_value=reproduce_result) as mock_reproduce:
            await post_fix_reproduction_node(state)

        mock_reproduce.assert_called_once()
        (rerun_input,), _kwargs = mock_reproduce.call_args
        assert rerun_input.workspace_path == str(workspace)
        assert rerun_input.commands == [["python", "repro.py", "--check"]]
        assert rerun_input.working_dir == "backend"
        assert rerun_input.image == "python:3.11-slim"
        assert rerun_input.timeout_seconds == 45
        assert rerun_input.expectation.exit_code_semantics == ExitCodeSemantics.NONZERO_IS_REPRODUCED
        assert rerun_input.expectation.reproduced_output_pattern == r"BUG: .*"
        assert rerun_input.expectation.not_reproduced_output_pattern is None
        assert rerun_input.task_context == "The cart subtotal test fails."


# ===========================================================================
# 9. No new planner call occurs after editing.
# ===========================================================================
@pytest.mark.asyncio
async def test_no_new_planner_call_during_post_fix_reproduction():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace)
        reproduce_result = BaselineResult(status=BaselineStatus.NOT_REPRODUCED, detail="ok", exit_code=0)

        with patch("app.services.agent.graph.reproduce", return_value=reproduce_result), patch(
            "app.services.agent.graph.plan_reproduction"
        ) as mock_plan, patch("app.services.agent.graph.build_reproduction_input") as mock_bridge:
            await post_fix_reproduction_node(state)

        mock_plan.assert_not_called()
        mock_bridge.assert_not_called()


# ===========================================================================
# 10/11. Post-fix reproduction uses the existing Phase 4A service (reproduce())
# against the SAME workspace_path -- no second workspace, no new mechanism.
# ===========================================================================
@pytest.mark.asyncio
async def test_post_fix_reproduction_uses_existing_reproduce_service_and_same_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        state = _reproduced_state(workspace)
        reproduce_result = BaselineResult(status=BaselineStatus.NOT_REPRODUCED, detail="ok", exit_code=0)

        with patch("app.services.agent.graph.reproduce", return_value=reproduce_result) as mock_reproduce:
            await post_fix_reproduction_node(state)

        mock_reproduce.assert_called_once()
        (rerun_input,), _ = mock_reproduce.call_args
        # Exact same workspace_path as state["workspace_dir"] -- never a
        # second/different workspace.
        assert rerun_input.workspace_path == state["workspace_dir"]


def test_graph_module_never_imports_workspace_manager_or_docker():
    """Static proof (not a mock assertion) that post_fix_reproduction_node
    cannot create a second workspace or call Docker directly: the module
    that defines it never imports either mechanism at all."""
    import app.services.agent.graph as graph_module

    with open(graph_module.__file__, "r", encoding="utf-8") as f:
        contents = f.read()
    assert "WorkspaceManager" not in contents
    assert "import docker" not in contents
    assert "containers.run" not in contents
    assert "docker.from_env" not in contents


# ===========================================================================
# 12. Retry behavior does not rerun baseline: a multi-attempt run (attempt 1
# fails verification, attempt 2 passes + post-fix confirms fixed) still
# only ever plans/executes baseline reproduction exactly once.
# ===========================================================================
@pytest.mark.asyncio
async def test_full_graph_retry_never_reruns_baseline_or_creates_a_second_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        code_file = workspace / "math_lib.py"
        code_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (workspace / "test_math.py").write_text(
            "from math_lib import add\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        # Attempt 1: empty patch -> test fails. Attempt 2: correct patch -> passes.
        empty_response = MagicMock()
        empty_response.text = "[]"
        fix_response = MagicMock()
        fix_response.text = json.dumps([
            {"file_path": "math_lib.py", "code": "def add(a, b):\n    return a + b\n", "start_line": 1, "end_line": 2}
        ])
        # diagnose_node also runs a real (mocked) Gemini call on every pass
        # between retrieve and plan -- see app.services.agent.graph.
        # diagnose_node. A genuine no-evidence-shaped response before each
        # patch-generation response keeps it advisory-inert here, since
        # this test isn't exercising diagnosis behavior itself.
        no_evidence_diagnosis_response = MagicMock()
        no_evidence_diagnosis_response.text = json.dumps({"summary": "", "hypotheses": [], "confidence": "no_evidence"})
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            no_evidence_diagnosis_response,
            empty_response,
            no_evidence_diagnosis_response,
            fix_response,
        ]

        plan = _applicable_plan()
        bridge_result = _executable_bridge_result(str(workspace))
        baseline_reproduce_result = BaselineResult(status=BaselineStatus.REPRODUCED, detail="reproduced", exit_code=1)
        post_fix_reproduce_result = BaselineResult(
            status=BaselineStatus.NOT_REPRODUCED, detail="no longer reproduces", exit_code=0
        )

        initial_state = _base_state(workspace, "The add() function in math_lib.py returns the wrong result.")

        with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"), patch(
            "google.genai.Client", return_value=mock_client
        ), patch("app.services.agent.graph.plan_reproduction", return_value=plan) as mock_plan_reproduction, patch(
            "app.services.agent.graph.build_reproduction_input", return_value=bridge_result
        ) as mock_build_input, patch(
            "app.services.agent.graph.reproduce",
            side_effect=[baseline_reproduce_result, post_fix_reproduce_result],
        ) as mock_reproduce:
            final_state = await agent_app.ainvoke(initial_state)

        assert final_state["attempt_count"] == 2
        assert final_state["baseline_status"] == "REPRODUCED"
        assert final_state["post_fix_reproduction_status"] == "NOT_REPRODUCED"
        assert final_state["outcome"] == "FIXED"

        # Baseline planning/bridging happened exactly once despite 2 attempts.
        mock_plan_reproduction.assert_called_once()
        mock_build_input.assert_called_once()
        # reproduce() called exactly twice total: once pre-fix (baseline),
        # once post-fix -- never once per attempt, never re-running baseline.
        assert mock_reproduce.call_count == 2


# ===========================================================================
# 13. A post-fix reproduction failure cannot become FIXED merely because the
# general full test suite passes (direct finalize_node proof, complementing
# test 2 above).
# ===========================================================================
def test_finalize_node_general_suite_pass_alone_cannot_override_still_reproducing_post_fix():
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": True,
        "baseline_status": "REPRODUCED",
        "post_fix_reproduction_status": "REPRODUCED",
        "post_fix_reproduction_detail": "the same failure was observed again",
    }
    out = finalize_node(state)
    assert out["outcome"] == "FAILED"
    assert "not been fixed" in out["outcome_detail"].lower() or "not fixed" in out["outcome_detail"].lower()


# ===========================================================================
# 14. GitHub approval/PR behavior is untouched by this phase (static
# inspection -- no GitHub calls are made anywhere in this suite).
# ===========================================================================
def test_no_github_integration_module_is_touched_by_post_fix_reproduction():
    import app.services.agent.graph as graph_module

    source = graph_module.__file__
    with open(source, "r", encoding="utf-8") as f:
        contents = f.read()
    assert "github" not in contents.lower()
