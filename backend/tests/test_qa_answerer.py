"""Unit tests for structured answer synthesis (app.services.qa.answerer).

Gemini is always mocked -- no real API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.services.qa.answerer import NO_EVIDENCE_SUMMARY, generate_answer, no_evidence_answer
from app.services.qa.classifier import QuestionClass
from app.services.qa.investigator import Evidence
from app.services.rag.retriever import RetrievedChunk


@pytest.fixture
def real_looking_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaFakeRealLookingKey123")


def _evidence(
    project_root=".", ecosystem="python", languages=None, frameworks=None,
    chunks=None, files_inspected=None, symbol_matches=None,
) -> Evidence:
    return Evidence(
        project_root=project_root,
        ecosystem=ecosystem,
        languages=languages if languages is not None else ["Python"],
        frameworks=frameworks or [],
        build_system=None,
        package_manager=None,
        test_system=None,
        project_evidence=[],
        file_count=1,
        chunks=chunks or [],
        files_inspected=files_inspected or [],
        symbol_matches=symbol_matches or [],
    )


def _chunk(file_path, symbol_name=None) -> RetrievedChunk:
    return RetrievedChunk(
        file_path=file_path, symbol_name=symbol_name, symbol_type="function",
        start_line=1, end_line=5, source_code="stub", similarity_score=0.9,
    )


def _qclass(**overrides) -> QuestionClass:
    defaults = dict(kind="lookup", depth="shallow", subject_terms=[], user_asserted_tech=[], likely_multi_file=False)
    defaults.update(overrides)
    return QuestionClass(**defaults)


def _mock_gemini_response(payload: dict):
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return patch("app.services.qa.answerer.genai.Client", return_value=mock_client)


# ---------------------------------------------------------------------------
# Structured parsing / citation preservation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_answer_parses_structured_response(real_looking_key):
    evidence = [_evidence(chunks=[_chunk("src/cart.py", "subtotal"), _chunk("src/cart.py", "total")])]
    payload = {
        "summary": "The subtotal is calculated in subtotal().",
        "details": "It sums item prices.",
        "flow_trace": None,
        "evidence": [{"file_path": "src/cart.py", "start_line": 1, "end_line": 5, "symbol_name": "subtotal"}],
        "corrected_premise": None,
        "confidence": "direct_evidence",
    }
    with _mock_gemini_response(payload):
        answer = await generate_answer("Where is subtotal calculated?", _qclass(), evidence)

    assert answer.summary == "The subtotal is calculated in subtotal()."
    assert answer.details == "It sums item prices."
    assert answer.confidence == "direct_evidence"
    assert answer.evidence[0].file_path == "src/cart.py"
    # Existing citation convention preserved: file:start-end
    assert answer.evidence[0].citation == "src/cart.py:1-5"


@pytest.mark.asyncio
async def test_evidence_is_included_in_gemini_prompt(real_looking_key):
    evidence = [_evidence(chunks=[_chunk("src/cart.py", "subtotal"), _chunk("src/cart.py", "total")])]
    payload = {"summary": "x", "confidence": "inferred", "evidence": []}
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.qa.answerer.genai.Client", return_value=mock_client):
        await generate_answer("Where is subtotal calculated?", _qclass(), evidence)

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    prompt = call_kwargs["contents"]
    assert "src/cart.py:1-5" in prompt
    assert "subtotal" in prompt

    system_instruction = call_kwargs["config"]["system_instruction"]
    assert "Repository evidence is authoritative" in system_instruction
    assert "hypothesis" in system_instruction.lower()
    assert NO_EVIDENCE_SUMMARY in system_instruction


@pytest.mark.asyncio
async def test_flow_trace_is_parsed(real_looking_key):
    evidence = [_evidence(chunks=[_chunk("src/checkout.py", "checkout"), _chunk("src/payment.py", "charge")])]
    payload = {
        "summary": "The payment flow goes from checkout to charge.",
        "confidence": "inferred",
        "evidence": [],
        "flow_trace": [
            {
                "order": 1, "description": "User checks out", "file_path": "src/checkout.py",
                "citation": {"file_path": "src/checkout.py", "start_line": 1, "end_line": 5, "symbol_name": "checkout"},
            },
            {
                "order": 2, "description": "Payment is charged", "file_path": "src/payment.py",
                "citation": {"file_path": "src/payment.py", "start_line": 1, "end_line": 5, "symbol_name": "charge"},
            },
        ],
    }
    with _mock_gemini_response(payload):
        answer = await generate_answer(
            "How does the payment flow work?", _qclass(kind="flow", depth="medium"), evidence
        )

    assert answer.flow_trace is not None
    assert len(answer.flow_trace) == 2
    assert answer.flow_trace[0].order == 1
    assert answer.flow_trace[1].file_path == "src/payment.py"
    assert answer.flow_trace[1].citation.citation == "src/payment.py:1-5"


@pytest.mark.asyncio
async def test_multi_project_evidence_reports_all_projects_considered(real_looking_key):
    evidence = [
        _evidence(
            project_root="backend", ecosystem="java-maven",
            chunks=[_chunk("backend/Auth.java", "Auth"), _chunk("backend/Auth.java", "login")],
        ),
        _evidence(
            project_root="frontend", ecosystem="node",
            chunks=[_chunk("frontend/Login.jsx", "Login"), _chunk("frontend/Login.jsx", "onSubmit")],
        ),
    ]
    payload = {"summary": "x", "confidence": "inferred", "evidence": []}
    with _mock_gemini_response(payload):
        answer = await generate_answer("How does login work?", _qclass(), evidence)

    assert set(answer.projects_considered) == {"backend", "frontend"}


# ---------------------------------------------------------------------------
# No evidence: deterministic, never calls Gemini
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_answer_no_evidence_is_deterministic_and_skips_gemini(real_looking_key):
    evidence = [_evidence(chunks=[], files_inspected=[], symbol_matches=[])]

    with patch("app.services.qa.answerer.genai.Client") as mock_ctor:
        answer = await generate_answer("Where is the nonexistent gateway?", _qclass(), evidence)

    assert answer.confidence == "no_evidence"
    assert answer.summary == NO_EVIDENCE_SUMMARY
    mock_ctor.assert_not_called()


def test_no_evidence_answer_helper_matches_required_sentence():
    answer = no_evidence_answer(["."])
    assert answer.summary == "There is no evidence of the requested component/feature in this repository."
    assert answer.confidence == "no_evidence"
    assert answer.evidence == []


# ---------------------------------------------------------------------------
# Premise correction: user terminology vs real evidence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_premise_correction_react_vs_vue(real_looking_key):
    evidence = [
        _evidence(
            ecosystem="node", languages=["JavaScript"], frameworks=["Vue"],
            chunks=[_chunk("src/AuthView.vue", None), _chunk("src/AuthView.vue", "setup")],
        )
    ]
    qclass = _qclass(kind="existence_check", depth="targeted", user_asserted_tech=["React"])
    # Even if Gemini itself forgets to set corrected_premise, the answerer
    # must compute it deterministically -- never leave it purely to the LLM.
    payload = {"summary": "x", "confidence": "direct_evidence", "evidence": [], "corrected_premise": None}

    with _mock_gemini_response(payload):
        answer = await generate_answer("Is there a React auth component?", qclass, evidence)

    assert answer.corrected_premise is not None
    assert "React" in answer.corrected_premise
    assert "Vue" in answer.corrected_premise


@pytest.mark.asyncio
async def test_no_premise_correction_when_terminology_matches(real_looking_key):
    evidence = [
        _evidence(
            ecosystem="node", languages=["JavaScript"], frameworks=["React"],
            chunks=[_chunk("src/Login.jsx", "Login"), _chunk("src/Login.jsx", "onSubmit")],
        )
    ]
    qclass = _qclass(user_asserted_tech=["React"])
    payload = {"summary": "x", "confidence": "direct_evidence", "evidence": []}

    with _mock_gemini_response(payload):
        answer = await generate_answer("Where is the React login component?", qclass, evidence)

    assert answer.corrected_premise is None


# ---------------------------------------------------------------------------
# PART 5 safety rule: weak evidence must not support a confident claim
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_thin_evidence_direct_evidence_claim_is_downgraded(real_looking_key):
    """A single Login.jsx chunk must not let the model's self-reported
    "direct_evidence" for a broad claim (e.g. how auth tokens are stored)
    stand unchallenged."""
    evidence = [_evidence(ecosystem="node", frameworks=["React"], chunks=[_chunk("src/Login.jsx", "Login")])]
    payload = {
        "summary": "Authentication uses JWT stored in localStorage.",
        "confidence": "direct_evidence",
        "evidence": [{"file_path": "src/Login.jsx", "start_line": 1, "end_line": 5, "symbol_name": "Login"}],
    }
    with _mock_gemini_response(payload):
        answer = await generate_answer(
            "Explain authentication.", _qclass(kind="architecture", depth="deep"), evidence
        )

    assert answer.confidence == "inferred"


@pytest.mark.asyncio
async def test_sufficient_evidence_direct_evidence_claim_is_preserved(real_looking_key):
    """The downgrade rule must not fire when there genuinely is enough
    evidence to support a direct claim."""
    evidence = [
        _evidence(
            chunks=[_chunk("src/cart.py", "subtotal")],
            files_inspected=[],
            symbol_matches=[],
        )
    ]
    # Bump total evidence pieces above the thin-evidence threshold.
    evidence[0].chunks.append(_chunk("src/cart.py", "total"))
    payload = {"summary": "x", "confidence": "direct_evidence", "evidence": []}

    with _mock_gemini_response(payload):
        answer = await generate_answer("Where is subtotal calculated?", _qclass(), evidence)

    assert answer.confidence == "direct_evidence"


@pytest.mark.asyncio
async def test_hallucinated_citation_is_filtered_out(real_looking_key):
    """A citation for a file that was never actually gathered as evidence
    must be dropped, not trusted."""
    evidence = [_evidence(chunks=[_chunk("src/cart.py", "subtotal"), _chunk("src/cart.py", "total")])]
    payload = {
        "summary": "x",
        "confidence": "inferred",
        "evidence": [
            {"file_path": "src/cart.py", "start_line": 1, "end_line": 5, "symbol_name": "subtotal"},
            {"file_path": "src/made_up_file.py", "start_line": 1, "end_line": 2, "symbol_name": "fake"},
        ],
    }
    with _mock_gemini_response(payload):
        answer = await generate_answer("Where is subtotal calculated?", _qclass(), evidence)

    files = {c.file_path for c in answer.evidence}
    assert "src/cart.py" in files
    assert "src/made_up_file.py" not in files


# ---------------------------------------------------------------------------
# Malformed Gemini output degrades safely
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_malformed_gemini_output_degrades_safely(real_looking_key):
    evidence = [_evidence(chunks=[_chunk("src/cart.py", "subtotal"), _chunk("src/cart.py", "total")])]
    mock_response = MagicMock()
    mock_response.text = "not json at all {{{"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.qa.answerer.genai.Client", return_value=mock_client):
        answer = await generate_answer("Where is subtotal calculated?", _qclass(), evidence)

    assert answer.confidence in ("inferred", "no_evidence")
    assert answer.summary
    assert answer.projects_considered == ["."]


@pytest.mark.asyncio
async def test_gemini_exception_degrades_safely(real_looking_key):
    evidence = [_evidence(chunks=[_chunk("src/cart.py", "subtotal"), _chunk("src/cart.py", "total")])]
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

    with patch("app.services.qa.answerer.genai.Client", return_value=mock_client):
        answer = await generate_answer("Where is subtotal calculated?", _qclass(), evidence)

    assert answer.confidence == "inferred"
    assert "quota exceeded" in answer.summary
