"""Unit tests for app.services.diagnosis.diagnoser.diagnose.

Gemini is always mocked -- no real API calls. Mirrors test_qa_answerer.py's
own conventions (real_looking_key fixture, _mock_gemini_response helper)
since diagnoser.py is a direct structural analogue of qa/answerer.py.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.services.diagnosis.diagnoser import NO_EVIDENCE_SUMMARY, diagnose, insufficient_evidence_diagnosis
from app.services.diagnosis.models import DiagnosisStatus


@pytest.fixture
def real_looking_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaFakeRealLookingKey123")


def _context(file_path="src/cart.py", content="def subtotal(): return 1", total_lines=1):
    return {"file_path": file_path, "content": content, "total_lines": total_lines}


def _mock_gemini_response(payload: dict):
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return patch("app.services.diagnosis.diagnoser.genai.Client", return_value=mock_client)


# ---------------------------------------------------------------------------
# No evidence: deterministic, never calls Gemini
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_diagnose_no_context_is_deterministic_and_skips_gemini(real_looking_key):
    with patch("app.services.diagnosis.diagnoser.genai") as mock_genai:
        result = await diagnose("Fix the bug", retrieved_context=[])

    assert result.status == DiagnosisStatus.INSUFFICIENT_EVIDENCE
    assert result.summary == NO_EVIDENCE_SUMMARY
    assert result.confidence == "no_evidence"
    mock_genai.Client.assert_not_called()


def test_insufficient_evidence_diagnosis_helper_matches_required_status():
    result = insufficient_evidence_diagnosis()
    assert result.status == DiagnosisStatus.INSUFFICIENT_EVIDENCE
    assert result.hypotheses == []


# ---------------------------------------------------------------------------
# Structured parsing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_diagnose_parses_structured_response(real_looking_key):
    payload = {
        "summary": "subtotal() uses the wrong operator.",
        "hypotheses": [
            {
                "rank": 1,
                "description": "subtotal() subtracts instead of adding.",
                "citations": [{"file_path": "src/cart.py", "start_line": 1, "end_line": 1, "symbol_name": "subtotal"}],
                "suggested_fix_approach": "Swap - for +.",
            }
        ],
        "confidence": "direct_evidence",
    }
    with _mock_gemini_response(payload):
        result = await diagnose("Fix subtotal", retrieved_context=[_context(), _context("src/other.py", "x=1", 1)])

    assert result.status == DiagnosisStatus.DIAGNOSED
    assert result.summary == "subtotal() uses the wrong operator."
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].citations[0].citation == "src/cart.py:1-1"


@pytest.mark.asyncio
async def test_error_analysis_and_context_are_included_in_prompt(real_looking_key):
    payload = {"summary": "x", "hypotheses": [], "confidence": "inferred"}
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.diagnosis.diagnoser.genai.Client", return_value=mock_client):
        await diagnose(
            "Fix subtotal",
            retrieved_context=[_context()],
            error_analysis="AssertionError: subtotal() returned -1",
        )

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    prompt = call_kwargs["contents"]
    assert "src/cart.py" in prompt
    assert "AssertionError: subtotal() returned -1" in prompt

    system_instruction = call_kwargs["config"]["system_instruction"]
    assert "ROOT CAUSE" in system_instruction


# ---------------------------------------------------------------------------
# Hallucinated citation filtering / confidence sanity-check (end-to-end,
# via validate_diagnosis)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hallucinated_citation_is_filtered_out(real_looking_key):
    payload = {
        "summary": "x",
        "hypotheses": [
            {
                "rank": 1,
                "description": "x",
                "citations": [
                    {"file_path": "src/cart.py", "start_line": 1, "end_line": 1},
                    {"file_path": "src/made_up_file.py", "start_line": 1, "end_line": 2},
                ],
            }
        ],
        "confidence": "inferred",
    }
    with _mock_gemini_response(payload):
        result = await diagnose("Fix subtotal", retrieved_context=[_context()])

    files = {c.file_path for h in result.hypotheses for c in h.citations}
    assert files == {"src/cart.py"}


@pytest.mark.asyncio
async def test_thin_evidence_direct_evidence_confidence_is_downgraded(real_looking_key):
    payload = {"summary": "x", "hypotheses": [], "confidence": "direct_evidence"}
    with _mock_gemini_response(payload):
        result = await diagnose("Fix subtotal", retrieved_context=[_context()])

    assert result.confidence == "inferred"


# ---------------------------------------------------------------------------
# Malformed Gemini output / exception -> DIAGNOSIS_FAILED, never a verdict
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_malformed_gemini_output_yields_diagnosis_failed(real_looking_key):
    mock_response = MagicMock()
    mock_response.text = "not json at all {{{"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.diagnosis.diagnoser.genai.Client", return_value=mock_client):
        result = await diagnose("Fix subtotal", retrieved_context=[_context()])

    assert result.status == DiagnosisStatus.DIAGNOSIS_FAILED
    assert result.failure_reason is not None


@pytest.mark.asyncio
async def test_gemini_exception_yields_diagnosis_failed(real_looking_key):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

    with patch("app.services.diagnosis.diagnoser.genai.Client", return_value=mock_client):
        result = await diagnose("Fix subtotal", retrieved_context=[_context()])

    assert result.status == DiagnosisStatus.DIAGNOSIS_FAILED
    assert "quota exceeded" in result.failure_reason


@pytest.mark.asyncio
async def test_non_dict_gemini_json_yields_diagnosis_failed(real_looking_key):
    mock_response = MagicMock()
    mock_response.text = json.dumps(["not", "a", "dict"])
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.diagnosis.diagnoser.genai.Client", return_value=mock_client):
        result = await diagnose("Fix subtotal", retrieved_context=[_context()])

    assert result.status == DiagnosisStatus.DIAGNOSIS_FAILED


# ---------------------------------------------------------------------------
# Test/mock environment (no real Gemini key) -- deterministic, no network
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_diagnose_without_gemini_key_uses_deterministic_mock():
    """Default test settings.gemini_api_key ("test_gemini_key_123", set in
    conftest.py) must never trigger a real network call."""
    with patch("app.services.diagnosis.diagnoser.genai") as mock_genai:
        result = await diagnose("Fix subtotal", retrieved_context=[_context()])

    mock_genai.Client.assert_not_called()
    assert result.status == DiagnosisStatus.DIAGNOSED
    assert result.confidence == "inferred"
    assert len(result.hypotheses) == 1
