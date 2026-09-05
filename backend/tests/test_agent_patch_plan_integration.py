"""Integration tests for Phase 6C: validated patch planning wired into the
agent's LangGraph workflow (app.services.agent.graph.patch_plan_node, its
graph placement between diagnose_node and plan_node, and plan_node's
PLANNED-only allow-list gate on Gemini patch generation).

No real Gemini/Docker/GitHub calls anywhere -- app.services.patch_plan.
plan_patches is patched directly at its app.services.agent.graph import
binding for most tests (this suite is about the INTEGRATION contract, not
planner internals -- those already have their own dedicated suite,
test_patch_plan_planner.py, which this suite does not duplicate or
weaken).

Critical invariant asserted repeatedly below: patch planning is advisory
guidance for Gemini's patch-generation prompt, but a HARD, ALLOW-LIST gate
on whether that Gemini call happens at all -- ONLY patch_plan_status ==
"PLANNED" may result in a patch-generation call. INSUFFICIENT_DIAGNOSIS,
DIAGNOSIS_FAILED (folded into INSUFFICIENT_DIAGNOSIS by the planner
itself), PLANNING_FAILED, and NOT_APPLICABLE must all result in
proposed_patches == [] with NO Gemini call for patches -- never fall
through to "current behavior". diagnose_node/diagnoser.py, finalize_node,
and should_continue must remain completely untouched.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent.graph import (
    _generate_patches_with_gemini,
    agent_app,
    diagnose_node,
    finalize_node,
    patch_plan_node,
    plan_node,
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
from app.services.patch_plan.models import PatchPlan, PatchPlanStatus, PlannedChange
from app.services.qa.models import CitationRef


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
        reason="tests/test_math.py::test_add fails today because of this bug",
        reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["python", "repro.py"]],
        expected_observation="The add() test fails.",
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


def _diagnosed_state_diagnosis() -> dict:
    return {
        "status": "DIAGNOSED",
        "summary": "add() subtracts instead of adding.",
        "hypotheses": [
            {
                "rank": 1,
                "description": "Wrong operator in add().",
                "citations": [{"file_path": "math_lib.py", "start_line": 1, "end_line": 2, "symbol_name": "add"}],
                "suggested_fix_approach": None,
            }
        ],
        "confidence": "inferred",
        "failure_reason": None,
    }


def _planned(**overrides) -> PatchPlan:
    defaults = dict(
        status=PatchPlanStatus.PLANNED,
        summary="Fix the addition operator.",
        changes=[
            PlannedChange(
                file_path="math_lib.py",
                change_type="modify",
                description="Swap - for +.",
                rationale="Matches the diagnosed cause.",
                citations=[CitationRef(file_path="math_lib.py", start_line=1, end_line=2)],
            )
        ],
        diagnosis_alignment="Directly addresses the diagnosed operator bug.",
        confidence="inferred",
    )
    defaults.update(overrides)
    return PatchPlan(**defaults)


# ===========================================================================
# 1. patch_plan_node: never invokes reproduction/Docker/GitHub functionality
# ===========================================================================
@pytest.mark.asyncio
async def test_patch_plan_node_never_invokes_reproduction_or_docker():
    state = _base_state(
        Path("/nonexistent"), "Fix subtotal",
        diagnosis=_diagnosed_state_diagnosis(),
        retrieved_context=[{"file_path": "math_lib.py", "content": "x=1", "total_lines": 1}],
    )
    with patch(
        "app.services.agent.graph.plan_patches",
        AsyncMock(return_value=_planned()),
    ), patch("app.services.agent.graph.plan_reproduction") as mock_plan_repro, patch(
        "app.services.agent.graph.build_reproduction_input"
    ) as mock_build_input, patch(
        "app.services.agent.graph.reproduce"
    ) as mock_reproduce:
        out = await patch_plan_node(state)

    mock_plan_repro.assert_not_called()
    mock_build_input.assert_not_called()
    mock_reproduce.assert_not_called()
    assert out["patch_plan_status"] == "PLANNED"


# ===========================================================================
# 2. patch_plan_node: sets state fields from a PLANNED / failed result
# ===========================================================================
@pytest.mark.asyncio
async def test_patch_plan_node_sets_planned_state_fields():
    state = _base_state(
        Path("/nonexistent"), "Fix subtotal",
        diagnosis=_diagnosed_state_diagnosis(),
        retrieved_context=[{"file_path": "math_lib.py", "content": "x=1", "total_lines": 1}],
    )
    with patch("app.services.agent.graph.plan_patches", AsyncMock(return_value=_planned())):
        out = await patch_plan_node(state)

    assert out["patch_plan_status"] == "PLANNED"
    assert out["patch_plan"]["status"] == "PLANNED"
    assert out["patch_plan"]["changes"][0]["file_path"] == "math_lib.py"
    assert out["patch_plan_detail"] == "Fix the addition operator."


@pytest.mark.asyncio
async def test_patch_plan_node_planning_failed_never_raises_and_sets_failed_status():
    state = _base_state(Path("/nonexistent"), "Fix subtotal", diagnosis=None, retrieved_context=[])
    failed = PatchPlan(status=PatchPlanStatus.PLANNING_FAILED, confidence="no_evidence", failure_reason="quota exceeded")
    with patch("app.services.agent.graph.plan_patches", AsyncMock(return_value=failed)):
        out = await patch_plan_node(state)

    assert out["patch_plan_status"] == "PLANNING_FAILED"
    assert "quota exceeded" in out["patch_plan_detail"]


@pytest.mark.asyncio
async def test_patch_plan_node_unexpected_exception_degrades_to_planning_failed():
    """A crash inside the planner must never propagate out of
    patch_plan_node and must never crash the task."""
    state = _base_state(
        Path("/nonexistent"), "Fix subtotal",
        diagnosis=_diagnosed_state_diagnosis(),
        retrieved_context=[{"file_path": "a.py", "content": "x", "total_lines": 1}],
    )
    with patch("app.services.agent.graph.plan_patches", AsyncMock(side_effect=RuntimeError("boom"))):
        out = await patch_plan_node(state)

    assert out["patch_plan_status"] == "PLANNING_FAILED"
    assert out["patch_plan"] is None
    assert "boom" in out["patch_plan_detail"]


# ===========================================================================
# 3. patch_plan_node: refreshed on retry -- uses the CURRENT diagnosis and
# error_analysis each time it runs, never stale values from a prior pass
# ===========================================================================
@pytest.mark.asyncio
async def test_patch_plan_node_uses_fresh_diagnosis_and_error_analysis_on_each_call():
    mock_plan_patches = AsyncMock(return_value=_planned())
    with patch("app.services.agent.graph.plan_patches", mock_plan_patches):
        first_state = _base_state(
            Path("/nonexistent"), "Fix subtotal",
            diagnosis=_diagnosed_state_diagnosis(),
            retrieved_context=[{"file_path": "math_lib.py", "content": "v1", "total_lines": 1}],
            error_analysis=None,
        )
        await patch_plan_node(first_state)

        second_diagnosis = _diagnosed_state_diagnosis()
        second_diagnosis["summary"] = "still wrong on retry"
        second_state = _base_state(
            Path("/nonexistent"), "Fix subtotal",
            diagnosis=second_diagnosis,
            retrieved_context=[{"file_path": "math_lib.py", "content": "v2", "total_lines": 1}],
            error_analysis="AssertionError: add(2, 3) == -1",
        )
        await patch_plan_node(second_state)

    assert mock_plan_patches.await_count == 2
    first_call_kwargs = mock_plan_patches.await_args_list[0].kwargs
    second_call_kwargs = mock_plan_patches.await_args_list[1].kwargs
    assert first_call_kwargs["diagnosis"]["summary"] != second_call_kwargs["diagnosis"]["summary"]
    assert first_call_kwargs["error_analysis"] is None
    assert second_call_kwargs["error_analysis"] == "AssertionError: add(2, 3) == -1"
    assert first_call_kwargs["retrieved_context"][0]["content"] == "v1"
    assert second_call_kwargs["retrieved_context"][0]["content"] == "v2"


# ===========================================================================
# 4. plan_node: the PLANNED-only allow-list gate
# ===========================================================================
@pytest.mark.asyncio
async def test_plan_node_calls_gemini_when_patch_plan_status_is_planned():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "math_lib.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        state = _base_state(
            workspace, "Fix subtotal",
            retrieved_context=[{"file_path": "math_lib.py", "content": "x", "total_lines": 1}],
            patch_plan_status="PLANNED",
            patch_plan=_planned().to_dict(),
        )

        with patch("app.services.agent.graph._generate_patches_with_gemini", return_value=[]) as mock_gen:
            plan_node(state)

        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs["patch_plan"] == _planned().to_dict()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch_plan_status",
    [None, "NOT_APPLICABLE", "INSUFFICIENT_DIAGNOSIS", "PLANNING_FAILED"],
)
async def test_plan_node_never_calls_gemini_when_patch_plan_status_is_not_planned(patch_plan_status):
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "math_lib.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        state = _base_state(
            workspace, "Fix subtotal",
            retrieved_context=[{"file_path": "math_lib.py", "content": "x", "total_lines": 1}],
            patch_plan_status=patch_plan_status,
            patch_plan=None,
            patch_plan_detail="some detail",
        )

        with patch("app.services.agent.graph._generate_patches_with_gemini") as mock_gen:
            out = plan_node(state)

        mock_gen.assert_not_called()
        assert out["proposed_patches"] == []
        assert str(patch_plan_status or "absent") in out["repair_plan"]


@pytest.mark.asyncio
async def test_plan_node_never_reuses_stale_patches_when_gate_is_closed():
    """A prior attempt's proposed_patches must never be silently carried
    forward when this attempt's patch_plan_status closes the gate --
    every attempt's proposed_patches is governed by ITS OWN status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "math_lib.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        state = _base_state(
            workspace, "Fix subtotal",
            retrieved_context=[{"file_path": "math_lib.py", "content": "x", "total_lines": 1}],
            proposed_patches=[{"file_path": "math_lib.py", "code": "stale", "start_line": 1, "end_line": 1}],
            patch_plan_status="INSUFFICIENT_DIAGNOSIS",
            patch_plan=None,
        )

        out = plan_node(state)

    assert out["proposed_patches"] == []


# ===========================================================================
# 5. _generate_patches_with_gemini: byte-for-byte prompt equivalence when
# patch_plan is absent/not-planned (function-level contract, independent
# of plan_node's own gating)
# ===========================================================================
def _capture_prompt(**kwargs) -> str:
    mock_response = MagicMock()
    mock_response.text = "[]"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"), patch(
        "google.genai.Client", return_value=mock_client
    ):
        _generate_patches_with_gemini(**kwargs)
    return mock_client.models.generate_content.call_args.kwargs["contents"]


def test_no_patch_plan_prompt_is_byte_for_byte_identical_to_pre_phase_6c():
    common_kwargs = dict(
        task_description="Fix subtotal",
        retrieved_context=[{"file_path": "math_lib.py", "content": "def add(a, b): return a - b", "total_lines": 1}],
        error_analysis=None,
        workspace_dir=None,
    )
    prompt_without_param = _capture_prompt(**common_kwargs)
    prompt_with_none = _capture_prompt(**common_kwargs, patch_plan=None)
    prompt_with_not_applicable = _capture_prompt(
        **common_kwargs,
        patch_plan={"status": "NOT_APPLICABLE", "summary": "x", "changes": [], "diagnosis_alignment": None, "confidence": "no_evidence", "failure_reason": None},
    )
    prompt_with_insufficient = _capture_prompt(
        **common_kwargs,
        patch_plan={"status": "INSUFFICIENT_DIAGNOSIS", "summary": "", "changes": [], "diagnosis_alignment": None, "confidence": "no_evidence", "failure_reason": "x"},
    )
    prompt_with_failed = _capture_prompt(
        **common_kwargs,
        patch_plan={"status": "PLANNING_FAILED", "summary": "", "changes": [], "diagnosis_alignment": None, "confidence": "no_evidence", "failure_reason": "x"},
    )

    assert prompt_without_param == prompt_with_none
    assert prompt_without_param == prompt_with_not_applicable
    assert prompt_without_param == prompt_with_insufficient
    assert prompt_without_param == prompt_with_failed
    assert "Proposed Patch Plan" not in prompt_without_param


def test_planned_patch_plan_prompt_includes_advisory_section_and_differs_from_baseline():
    common_kwargs = dict(
        task_description="Fix subtotal",
        retrieved_context=[{"file_path": "math_lib.py", "content": "def add(a, b): return a - b", "total_lines": 1}],
        error_analysis=None,
        workspace_dir=None,
    )
    prompt_without_plan = _capture_prompt(**common_kwargs)
    prompt_with_plan = _capture_prompt(**common_kwargs, patch_plan=_planned().to_dict())

    assert prompt_with_plan != prompt_without_plan
    assert "Proposed Patch Plan" in prompt_with_plan
    assert "Swap - for +." in prompt_with_plan
    assert "verify against the actual code above" in prompt_with_plan
    assert prompt_with_plan.startswith(prompt_without_plan.split("Generate the JSON patch array now:")[0])
    assert prompt_with_plan.endswith("Generate the JSON patch array now:")


# ===========================================================================
# 6. CRITICAL REGRESSION: diagnosis failure cannot accidentally produce a
# fabricated patch plan, and neither diagnosis nor patch-plan failure can
# ever affect finalize_node's outcome.
# ===========================================================================
def test_diagnosis_failed_alongside_reproduced_baseline_still_yields_fixed():
    """Unchanged from Phase 6A: diagnosis/patch-plan status is never read
    by finalize_node."""
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": True,
        "baseline_status": "REPRODUCED",
        "post_fix_reproduction_status": "NOT_REPRODUCED",
        "diagnosis_status": "DIAGNOSIS_FAILED",
        "diagnosis": None,
        "patch_plan_status": "INSUFFICIENT_DIAGNOSIS",
        "patch_plan": None,
    }
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


def test_patch_plan_failed_never_affects_finalize_outcome():
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": True,
        "patch_plan_status": "PLANNING_FAILED",
    }
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


def test_finalize_node_absent_patch_plan_fields_preserves_prior_behavior():
    """A hand-constructed state with no patch_plan fields at all (e.g. a
    task predating Phase 6C) behaves exactly as before."""
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": True,
    }
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


@pytest.mark.asyncio
async def test_diagnosis_failed_never_reaches_gemini_for_planning_or_patch_generation():
    """End-to-end regression for the fabrication-gate fix: a DIAGNOSIS_FAILED
    diagnosis must result in patch_plan_status == INSUFFICIENT_DIAGNOSIS
    (never PLANNED), and plan_node must never call Gemini for patches."""
    state = _base_state(
        Path("/nonexistent"), "Fix subtotal",
        diagnosis_status="DIAGNOSIS_FAILED",
        diagnosis=None,
        retrieved_context=[{"file_path": "a.py", "content": "x", "total_lines": 1}],
    )
    out = await patch_plan_node(state)
    assert out["patch_plan_status"] == "INSUFFICIENT_DIAGNOSIS"

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
        plan_state = _base_state(
            workspace, "Fix subtotal",
            retrieved_context=[{"file_path": "a.py", "content": "x", "total_lines": 1}],
            **out,
        )
        with patch("app.services.agent.graph._generate_patches_with_gemini") as mock_gen:
            plan_out = plan_node(plan_state)

    mock_gen.assert_not_called()
    assert plan_out["proposed_patches"] == []


# ===========================================================================
# 7. Full graph wiring: patch_plan sits between diagnose and plan
# ===========================================================================
def test_build_agent_graph_includes_patch_plan_between_diagnose_and_plan():
    graph = agent_app.get_graph()
    node_names = set(graph.nodes.keys())
    assert "patch_plan" in node_names

    edges = {(e.source, e.target) for e in graph.edges}
    assert ("diagnose", "patch_plan") in edges
    assert ("patch_plan", "plan") in edges
    assert ("diagnose", "plan") not in edges


# ===========================================================================
# 8. Full end-to-end graph run (mock-mode planner, no patched `plan_patches`)
# -- confirms the real patch_plan_node wiring produces a plan and never
# breaks the existing FIXED outcome path.
# ===========================================================================
@pytest.mark.asyncio
async def test_full_graph_runs_patch_planning_and_still_reaches_fixed():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        code_file = workspace / "math_lib.py"
        code_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (workspace / "test_math.py").write_text(
            "from math_lib import add\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        diagnosis_response = MagicMock()
        diagnosis_response.text = json.dumps(_diagnosed_state_diagnosis())
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

        assert final_state["patch_plan_status"] == "PLANNED"
        assert final_state["outcome"] == "FIXED"
        assert "return a + b" in code_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_full_graph_diagnosis_failure_skips_patch_generation_and_does_not_falsely_fix():
    """The direct behavioral consequence of the revised Phase 6C gate: when
    diagnosis genuinely fails for real (malformed Gemini output), patch
    generation is skipped entirely this attempt -- no patch is applied, and
    the task does not reach a false FIXED."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        code_file = workspace / "math_lib.py"
        code_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (workspace / "test_math.py").write_text(
            "from math_lib import add\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        # A malformed (non-JSON-object) response for every Gemini call --
        # diagnosis fails to parse, so patch_plan_node never even attempts
        # its own Gemini call, and plan_node's gate stays closed.
        malformed_response = MagicMock()
        malformed_response.text = json.dumps(["not", "an", "object"])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = malformed_response

        initial_state = _base_state(workspace, "The add() function in math_lib.py returns the wrong result.")

        with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"), patch(
            "google.genai.Client", return_value=mock_client
        ):
            final_state = await agent_app.ainvoke(initial_state)

        assert final_state["diagnosis_status"] == "DIAGNOSIS_FAILED"
        assert final_state["patch_plan_status"] == "INSUFFICIENT_DIAGNOSIS"
        assert final_state["proposed_patches"] == []
        assert final_state["outcome"] != "FIXED"
        # the file is genuinely untouched -- no fabricated fix was ever applied
        assert code_file.read_text(encoding="utf-8") == "def add(a, b):\n    return a - b\n"
