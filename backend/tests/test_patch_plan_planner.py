"""Unit tests for app.services.patch_plan.planner.plan_patches.

Gemini is always mocked -- no real API calls. Mirrors
test_diagnosis_diagnoser.py's own conventions (real_looking_key fixture,
_mock_gemini_response helper) since planner.py is a direct structural
analogue of diagnosis/diagnoser.py, one layer up.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.services.patch_plan.models import PatchPlanStatus
from app.services.patch_plan.planner import insufficient_diagnosis_plan, plan_patches


@pytest.fixture
def real_looking_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaFakeRealLookingKey123")


def _context(file_path="src/math_lib.py", content="def add(a, b): return a - b", total_lines=1):
    return {"file_path": file_path, "content": content, "total_lines": total_lines}


def _diagnosed(**overrides):
    defaults = dict(
        status="DIAGNOSED",
        summary="add() subtracts instead of adding.",
        hypotheses=[
            {
                "rank": 1,
                "description": "Wrong operator in add().",
                "citations": [{"file_path": "src/math_lib.py", "start_line": 1, "end_line": 1, "symbol_name": "add"}],
                "suggested_fix_approach": None,
            }
        ],
        confidence="inferred",
        failure_reason=None,
    )
    defaults.update(overrides)
    return defaults


def _mock_gemini_response(payload: dict):
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return patch("app.services.patch_plan.planner.genai.Client", return_value=mock_client)


# ---------------------------------------------------------------------------
# No usable diagnosis: deterministic, never calls Gemini
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_patches_no_diagnosis_is_deterministic_and_skips_gemini(real_looking_key):
    with patch("app.services.patch_plan.planner.genai") as mock_genai:
        result = await plan_patches("Fix subtotal", diagnosis=None, retrieved_context=[_context()])

    assert result.status == PatchPlanStatus.INSUFFICIENT_DIAGNOSIS
    assert result.failure_reason is not None
    mock_genai.Client.assert_not_called()


@pytest.mark.asyncio
async def test_plan_patches_diagnosis_failed_is_deterministic_and_skips_gemini(real_looking_key):
    diagnosis = {"status": "DIAGNOSIS_FAILED", "failure_reason": "quota exceeded"}
    with patch("app.services.patch_plan.planner.genai") as mock_genai:
        result = await plan_patches("Fix subtotal", diagnosis=diagnosis, retrieved_context=[_context()])

    assert result.status == PatchPlanStatus.INSUFFICIENT_DIAGNOSIS
    mock_genai.Client.assert_not_called()


@pytest.mark.asyncio
async def test_plan_patches_insufficient_evidence_diagnosis_is_deterministic_and_skips_gemini(real_looking_key):
    diagnosis = {"status": "INSUFFICIENT_EVIDENCE"}
    with patch("app.services.patch_plan.planner.genai") as mock_genai:
        result = await plan_patches("Fix subtotal", diagnosis=diagnosis, retrieved_context=[_context()])

    assert result.status == PatchPlanStatus.INSUFFICIENT_DIAGNOSIS
    mock_genai.Client.assert_not_called()


@pytest.mark.asyncio
async def test_plan_patches_empty_retrieved_context_is_deterministic_and_skips_gemini(real_looking_key):
    with patch("app.services.patch_plan.planner.genai") as mock_genai:
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[])

    assert result.status == PatchPlanStatus.INSUFFICIENT_DIAGNOSIS
    mock_genai.Client.assert_not_called()


def test_insufficient_diagnosis_plan_helper_matches_required_status():
    result = insufficient_diagnosis_plan("no diagnosis")
    assert result.status == PatchPlanStatus.INSUFFICIENT_DIAGNOSIS
    assert result.failure_reason == "no diagnosis"
    assert result.changes == []


# ---------------------------------------------------------------------------
# Valid diagnosis -> structured parsing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_patches_parses_structured_response(real_looking_key):
    payload = {
        "applicable": True,
        "summary": "Fix the operator.",
        "changes": [
            {
                "file_path": "src/math_lib.py",
                "change_type": "modify",
                "description": "Swap - for +.",
                "rationale": "Matches the diagnosed cause.",
                "citations": [{"file_path": "src/math_lib.py", "start_line": 1, "end_line": 1, "symbol_name": "add"}],
                "symbols_affected": ["add"],
            }
        ],
        "diagnosis_alignment": "Directly addresses the diagnosed cause.",
        "confidence": "direct_evidence",
    }
    with _mock_gemini_response(payload):
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context(), _context("src/other.py", "x=1", 1)])

    assert result.status == PatchPlanStatus.PLANNED
    assert result.summary == "Fix the operator."
    assert len(result.changes) == 1
    assert result.changes[0].file_path == "src/math_lib.py"
    assert result.changes[0].citations[0].citation == "src/math_lib.py:1-1"


@pytest.mark.asyncio
async def test_plan_patches_not_applicable_response(real_looking_key):
    payload = {"applicable": False, "summary": "Behavior is already correct.", "confidence": "inferred"}
    with _mock_gemini_response(payload):
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context()])

    assert result.status == PatchPlanStatus.NOT_APPLICABLE
    assert result.changes == []


@pytest.mark.asyncio
async def test_diagnosis_and_error_analysis_are_included_in_prompt(real_looking_key):
    payload = {"applicable": True, "summary": "x", "changes": [], "confidence": "inferred"}
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.patch_plan.planner.genai.Client", return_value=mock_client):
        await plan_patches(
            "Fix subtotal",
            diagnosis=_diagnosed(),
            retrieved_context=[_context()],
            error_analysis="AssertionError: add(2, 3) == -1",
        )

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    prompt = call_kwargs["contents"]
    assert "Wrong operator in add()" in prompt
    assert "src/math_lib.py" in prompt
    assert "AssertionError: add(2, 3) == -1" in prompt

    system_instruction = call_kwargs["config"]["system_instruction"]
    assert "PATCH PLAN" in system_instruction
    assert "modify" in system_instruction and "create" in system_instruction


# ---------------------------------------------------------------------------
# Hallucinated citation filtering / confidence sanity-check (end-to-end,
# via validate_patch_plan)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hallucinated_citation_is_filtered_out_end_to_end(real_looking_key):
    payload = {
        "applicable": True,
        "summary": "x",
        "changes": [
            {
                "file_path": "src/math_lib.py",
                "change_type": "modify",
                "description": "d",
                "rationale": "r",
                "citations": [
                    {"file_path": "src/math_lib.py", "start_line": 1, "end_line": 1},
                    {"file_path": "src/made_up_file.py", "start_line": 1, "end_line": 2},
                ],
            }
        ],
        "confidence": "inferred",
    }
    with _mock_gemini_response(payload):
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context()])

    files = {c.file_path for h in result.changes for c in h.citations}
    assert files == {"src/math_lib.py"}


@pytest.mark.asyncio
async def test_thin_evidence_direct_evidence_confidence_is_downgraded_end_to_end(real_looking_key):
    payload = {
        "applicable": True,
        "summary": "x",
        "changes": [{"file_path": "src/math_lib.py", "change_type": "modify", "description": "d", "rationale": "r"}],
        "confidence": "direct_evidence",
    }
    with _mock_gemini_response(payload):
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context()])

    assert result.confidence == "inferred"


# ---------------------------------------------------------------------------
# Diagnosis conflict: plan targets files the diagnosis never cited
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_diagnosis_conflicting_plan_is_not_rejected_but_flagged(real_looking_key):
    payload = {
        "applicable": True,
        "summary": "x",
        "changes": [
            {
                "file_path": "src/unrelated.py",
                "change_type": "modify",
                "description": "d",
                "rationale": "r",
                "citations": [
                    {"file_path": "src/unrelated.py", "start_line": 1, "end_line": 1},
                    {"file_path": "src/unrelated.py", "start_line": 2, "end_line": 3},
                ],
            }
        ],
        "confidence": "direct_evidence",
    }
    with _mock_gemini_response(payload):
        result = await plan_patches(
            "Fix subtotal",
            diagnosis=_diagnosed(),
            retrieved_context=[_context(), _context("src/unrelated.py", "x=1", 1)],
        )

    # Not rejected wholesale (advisory-only, filter/repair philosophy) --
    # but confidence is downgraded since the plan never touches the
    # diagnosis's own cited file.
    assert result.status == PatchPlanStatus.PLANNED
    assert result.confidence == "inferred"


# ---------------------------------------------------------------------------
# Malformed Gemini output / exception -> PLANNING_FAILED, never a verdict
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_malformed_gemini_output_yields_planning_failed(real_looking_key):
    mock_response = MagicMock()
    mock_response.text = "not json at all {{{"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.patch_plan.planner.genai.Client", return_value=mock_client):
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context()])

    assert result.status == PatchPlanStatus.PLANNING_FAILED
    assert result.failure_reason is not None


@pytest.mark.asyncio
async def test_gemini_exception_yields_planning_failed(real_looking_key):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

    with patch("app.services.patch_plan.planner.genai.Client", return_value=mock_client):
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context()])

    assert result.status == PatchPlanStatus.PLANNING_FAILED
    assert "quota exceeded" in result.failure_reason


@pytest.mark.asyncio
async def test_non_dict_gemini_json_yields_planning_failed(real_looking_key):
    mock_response = MagicMock()
    mock_response.text = json.dumps(["not", "a", "dict"])
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.patch_plan.planner.genai.Client", return_value=mock_client):
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context()])

    assert result.status == PatchPlanStatus.PLANNING_FAILED


@pytest.mark.asyncio
async def test_all_changes_rejected_by_validation_yields_planning_failed(real_looking_key):
    payload = {
        "applicable": True,
        "summary": "x",
        "changes": [{"file_path": "/etc/passwd", "change_type": "modify", "description": "d", "rationale": "r"}],
        "confidence": "inferred",
    }
    with _mock_gemini_response(payload):
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context()])

    assert result.status == PatchPlanStatus.PLANNING_FAILED
    assert result.changes == []


# ---------------------------------------------------------------------------
# Test/mock environment (no real Gemini key) -- deterministic, no network
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_plan_patches_without_gemini_key_uses_deterministic_mock():
    """Default test settings.gemini_api_key ("test_gemini_key_123", set in
    conftest.py) must never trigger a real network call, and the mock plan
    must be grounded in the diagnosis's own citation, never invented."""
    with patch("app.services.patch_plan.planner.genai") as mock_genai:
        result = await plan_patches("Fix subtotal", diagnosis=_diagnosed(), retrieved_context=[_context()])

    mock_genai.Client.assert_not_called()
    assert result.status == PatchPlanStatus.PLANNED
    assert len(result.changes) == 1
    assert result.changes[0].file_path == "src/math_lib.py"
    assert result.changes[0].citations[0].file_path == "src/math_lib.py"


@pytest.mark.asyncio
async def test_plan_patches_without_gemini_key_and_no_hypotheses_is_not_applicable():
    diagnosis = _diagnosed(hypotheses=[])
    with patch("app.services.patch_plan.planner.genai") as mock_genai:
        result = await plan_patches("Fix subtotal", diagnosis=diagnosis, retrieved_context=[_context()])

    mock_genai.Client.assert_not_called()
    assert result.status == PatchPlanStatus.NOT_APPLICABLE
