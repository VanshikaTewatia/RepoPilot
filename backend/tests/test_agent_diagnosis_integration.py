"""Integration tests for Phase 6A: evidence-driven root-cause diagnosis
wired into the agent's LangGraph workflow (app.services.agent.graph.
diagnose_node, its graph placement between retrieve_node and plan_node,
and plan_node's advisory use of a DIAGNOSED result).

No real Gemini/Docker/GitHub calls anywhere -- app.services.diagnosis.diagnose
is patched directly at its app.services.agent.graph import binding for most
tests (this suite is about the INTEGRATION contract, not diagnoser
internals -- those already have their own dedicated suite,
test_diagnosis_diagnoser.py, which this suite does not duplicate or
weaken). A couple of full-graph tests exercise the real (mock-mode)
diagnoser end to end, exactly like test_agent_baseline_integration.py's
own full-graph test does for baseline reproduction.

Critical invariant asserted repeatedly below: diagnosis is advisory only.
It must never call reproduction/Docker functionality, and a
DIAGNOSIS_FAILED/INSUFFICIENT_EVIDENCE/absent diagnosis must never block,
alter, or gate finalize_node's/should_continue's existing behavior.
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
from app.services.diagnosis.models import Diagnosis, DiagnosisStatus, RootCauseHypothesis
from app.services.qa.models import CitationRef


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


def _diagnosed(**overrides) -> Diagnosis:
    defaults = dict(
        status=DiagnosisStatus.DIAGNOSED,
        summary="subtotal() subtracts instead of adding.",
        hypotheses=[
            RootCauseHypothesis(
                rank=1,
                description="Wrong operator in subtotal().",
                citations=[CitationRef(file_path="src/cart.py", start_line=1, end_line=2)],
            )
        ],
        confidence="inferred",
    )
    defaults.update(overrides)
    return Diagnosis(**defaults)


# ===========================================================================
# 1. diagnose_node: never invokes reproduction/Docker/GitHub functionality
# ===========================================================================
@pytest.mark.asyncio
async def test_diagnose_node_never_invokes_reproduction_or_docker():
    state = _base_state(
        Path("/nonexistent"),
        "Fix subtotal",
        retrieved_context=[{"file_path": "src/cart.py", "content": "x=1", "total_lines": 1}],
    )
    with patch(
        "app.services.agent.graph.diagnose",
        AsyncMock(return_value=_diagnosed()),
    ), patch("app.services.agent.graph.plan_reproduction") as mock_plan_repro, patch(
        "app.services.agent.graph.build_reproduction_input"
    ) as mock_build_input, patch(
        "app.services.agent.graph.reproduce"
    ) as mock_reproduce:
        out = await diagnose_node(state)

    mock_plan_repro.assert_not_called()
    mock_build_input.assert_not_called()
    mock_reproduce.assert_not_called()
    assert out["diagnosis_status"] == "DIAGNOSED"


# ===========================================================================
# 2. diagnose_node: sets state fields from a DIAGNOSED / failed result
# ===========================================================================
@pytest.mark.asyncio
async def test_diagnose_node_sets_diagnosed_state_fields():
    state = _base_state(
        Path("/nonexistent"), "Fix subtotal",
        retrieved_context=[{"file_path": "src/cart.py", "content": "x=1", "total_lines": 1}],
    )
    with patch("app.services.agent.graph.diagnose", AsyncMock(return_value=_diagnosed())):
        out = await diagnose_node(state)

    assert out["diagnosis_status"] == "DIAGNOSED"
    assert out["diagnosis"]["status"] == "DIAGNOSED"
    assert out["diagnosis"]["hypotheses"][0]["description"] == "Wrong operator in subtotal()."
    assert out["diagnosis_detail"] == "subtotal() subtracts instead of adding."


@pytest.mark.asyncio
async def test_diagnose_node_diagnosis_failed_never_raises_and_sets_failed_status():
    state = _base_state(Path("/nonexistent"), "Fix subtotal", retrieved_context=[])
    failed = Diagnosis(status=DiagnosisStatus.DIAGNOSIS_FAILED, confidence="no_evidence", failure_reason="quota exceeded")
    with patch("app.services.agent.graph.diagnose", AsyncMock(return_value=failed)):
        out = await diagnose_node(state)

    assert out["diagnosis_status"] == "DIAGNOSIS_FAILED"
    assert "quota exceeded" in out["diagnosis_detail"]


@pytest.mark.asyncio
async def test_diagnose_node_unexpected_exception_degrades_to_diagnosis_failed():
    """A crash inside the diagnoser must never propagate out of
    diagnose_node and must never crash the task."""
    state = _base_state(Path("/nonexistent"), "Fix subtotal", retrieved_context=[{"file_path": "a.py", "content": "x", "total_lines": 1}])
    with patch("app.services.agent.graph.diagnose", AsyncMock(side_effect=RuntimeError("boom"))):
        out = await diagnose_node(state)

    assert out["diagnosis_status"] == "DIAGNOSIS_FAILED"
    assert out["diagnosis"] is None
    assert "boom" in out["diagnosis_detail"]


# ===========================================================================
# 3. diagnose_node: refreshed on retry -- uses the CURRENT retrieved_context
# and error_analysis each time it runs, never stale values from a prior pass
# ===========================================================================
@pytest.mark.asyncio
async def test_diagnose_node_uses_fresh_context_and_error_analysis_on_each_call():
    mock_diagnose = AsyncMock(return_value=_diagnosed())
    with patch("app.services.agent.graph.diagnose", mock_diagnose):
        first_state = _base_state(
            Path("/nonexistent"), "Fix subtotal",
            retrieved_context=[{"file_path": "src/cart.py", "content": "v1", "total_lines": 1}],
            error_analysis=None,
        )
        await diagnose_node(first_state)

        second_state = _base_state(
            Path("/nonexistent"), "Fix subtotal",
            retrieved_context=[{"file_path": "src/cart.py", "content": "v2", "total_lines": 1}],
            error_analysis="AssertionError: subtotal() returned -1",
        )
        await diagnose_node(second_state)

    assert mock_diagnose.await_count == 2
    first_call_kwargs = mock_diagnose.await_args_list[0].kwargs
    second_call_kwargs = mock_diagnose.await_args_list[1].kwargs
    assert first_call_kwargs["retrieved_context"][0]["content"] == "v1"
    assert first_call_kwargs["error_analysis"] is None
    assert second_call_kwargs["retrieved_context"][0]["content"] == "v2"
    assert second_call_kwargs["error_analysis"] == "AssertionError: subtotal() returned -1"


# ===========================================================================
# 4. plan_node: a valid DIAGNOSED result informs the Gemini prompt
# ===========================================================================
@pytest.mark.asyncio
async def test_plan_node_passes_diagnosis_through_to_patch_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "cart.py").write_text("def subtotal():\n    return 1\n", encoding="utf-8")

        state = _base_state(
            workspace, "Fix subtotal",
            retrieved_context=[{"file_path": "cart.py", "content": "x", "total_lines": 1}],
            diagnosis_status="DIAGNOSED",
            diagnosis=_diagnosed().to_dict(),
        )

        with patch("app.services.agent.graph._generate_patches_with_gemini", return_value=[]) as mock_gen:
            plan_node(state)

        assert mock_gen.call_args.kwargs["diagnosis"] == _diagnosed().to_dict()


# ===========================================================================
# 5. _generate_patches_with_gemini: byte-for-byte prompt equivalence when
# diagnosis is absent/failed/insufficient
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


def test_no_diagnosis_prompt_is_byte_for_byte_identical_to_pre_phase_6a():
    common_kwargs = dict(
        task_description="Fix subtotal",
        retrieved_context=[{"file_path": "cart.py", "content": "def subtotal(): return 1", "total_lines": 1}],
        error_analysis=None,
        workspace_dir=None,
    )
    prompt_without_param = _capture_prompt(**common_kwargs)
    prompt_with_none = _capture_prompt(**common_kwargs, diagnosis=None)
    prompt_with_failed = _capture_prompt(
        **common_kwargs,
        diagnosis={"status": "DIAGNOSIS_FAILED", "summary": "", "hypotheses": [], "confidence": "no_evidence", "failure_reason": "x"},
    )
    prompt_with_insufficient = _capture_prompt(
        **common_kwargs,
        diagnosis={"status": "INSUFFICIENT_EVIDENCE", "summary": "x", "hypotheses": [], "confidence": "no_evidence", "failure_reason": None},
    )

    assert prompt_without_param == prompt_with_none
    assert prompt_without_param == prompt_with_failed
    assert prompt_without_param == prompt_with_insufficient
    assert "Root Cause Diagnosis" not in prompt_without_param


def test_diagnosed_prompt_includes_advisory_section_and_differs_from_baseline():
    common_kwargs = dict(
        task_description="Fix subtotal",
        retrieved_context=[{"file_path": "cart.py", "content": "def subtotal(): return 1", "total_lines": 1}],
        error_analysis=None,
        workspace_dir=None,
    )
    prompt_without_diagnosis = _capture_prompt(**common_kwargs)
    prompt_with_diagnosis = _capture_prompt(**common_kwargs, diagnosis=_diagnosed().to_dict())

    assert prompt_with_diagnosis != prompt_without_diagnosis
    assert "Root Cause Diagnosis" in prompt_with_diagnosis
    assert "Wrong operator in subtotal()" in prompt_with_diagnosis
    assert prompt_with_diagnosis.startswith(prompt_without_diagnosis.split("Generate the JSON patch array now:")[0])
    assert prompt_with_diagnosis.endswith("Generate the JSON patch array now:")


# ===========================================================================
# 6. CRITICAL REGRESSION: diagnosis must never affect finalize_node's
# outcome, even a DIAGNOSIS_FAILED diagnosis alongside a REPRODUCED baseline
# and a confirmed post-fix -- this exact combination still yields FIXED.
# ===========================================================================
def test_diagnosis_failed_alongside_reproduced_baseline_still_yields_fixed():
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": True,
        "baseline_status": "REPRODUCED",
        "post_fix_reproduction_status": "NOT_REPRODUCED",
        "diagnosis_status": "DIAGNOSIS_FAILED",
        "diagnosis": None,
        "diagnosis_detail": "Gemini quota exceeded",
    }
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


def test_diagnosis_insufficient_evidence_never_affects_finalize_outcome():
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": True,
        "diagnosis_status": "INSUFFICIENT_EVIDENCE",
    }
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


def test_finalize_node_absent_diagnosis_fields_preserves_prior_behavior():
    """A hand-constructed state with no diagnosis fields at all (e.g. a
    task predating Phase 6A) behaves exactly as before."""
    state = {
        "test_results": {"available": True, "success": True, "failed": 0},
        "proposed_patches": [{"file_path": "a.py", "code": "x = 1\n"}],
        "applied_patch_count": 1,
        "is_verified": True,
    }
    out = finalize_node(state)
    assert out["outcome"] == "FIXED"


# ===========================================================================
# 7. Full graph wiring: diagnose sits between retrieve and plan
# ===========================================================================
def test_build_agent_graph_includes_diagnose_between_retrieve_and_plan():
    graph = agent_app.get_graph()
    node_names = set(graph.nodes.keys())
    assert "diagnose" in node_names

    edges = {(e.source, e.target) for e in graph.edges}
    assert ("retrieve", "diagnose") in edges
    assert ("diagnose", "plan") in edges
    assert ("retrieve", "plan") not in edges


# ===========================================================================
# 8. Full end-to-end graph run (mock-mode diagnoser, no patched `diagnose`)
# -- confirms the real diagnose_node wiring produces a diagnosis and never
# breaks the existing FIXED outcome path.
# ===========================================================================
@pytest.mark.asyncio
async def test_full_graph_runs_diagnosis_and_still_reaches_fixed():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        code_file = workspace / "math_lib.py"
        code_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (workspace / "test_math.py").write_text(
            "from math_lib import add\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        patch_response = MagicMock()
        patch_response.text = json.dumps([
            {"file_path": "math_lib.py", "code": "def add(a, b):\n    return a + b\n", "start_line": 1, "end_line": 2}
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = patch_response

        plan = _applicable_plan()
        bridge_result = _executable_bridge_result(str(workspace))
        baseline_reproduce_result = BaselineResult(status=BaselineStatus.REPRODUCED, detail="reproduced", exit_code=1)
        post_fix_reproduce_result = BaselineResult(
            status=BaselineStatus.NOT_REPRODUCED, detail="no longer reproduces", exit_code=0
        )

        initial_state = _base_state(workspace, "The add() function in math_lib.py returns the wrong result.")

        # Baseline planning/execution is patched directly (exactly like
        # test_agent_baseline_integration.py's own full-graph test) so this
        # test isolates diagnosis's real wiring. The single mocked Gemini
        # client is shared by diagnosis and patch generation and only ever
        # returns the patch-array JSON above -- so diagnosis genuinely
        # fails to parse it as a diagnosis object (a real DIAGNOSIS_FAILED,
        # exercising the actual diagnose_node wiring end to end) while
        # patch generation, which expects exactly this shape, succeeds.
        # This is itself the regression under test: a real diagnosis
        # failure alongside a real, confirmed fix must still reach FIXED.
        with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"), patch(
            "google.genai.Client", return_value=mock_client
        ), patch("app.services.agent.graph.plan_reproduction", return_value=plan), patch(
            "app.services.agent.graph.build_reproduction_input", return_value=bridge_result
        ), patch(
            "app.services.agent.graph.reproduce",
            side_effect=[baseline_reproduce_result, post_fix_reproduce_result],
        ):
            final_state = await agent_app.ainvoke(initial_state)

        assert final_state["diagnosis_status"] in ("DIAGNOSED", "DIAGNOSIS_FAILED", "INSUFFICIENT_EVIDENCE")
        assert final_state["outcome"] == "FIXED"
        assert "return a + b" in code_file.read_text(encoding="utf-8")
