"""Integration tests for the Deep Codebase Q&A orchestration
(app.services.qa.service.ask_codebase): classifier -> investigator ->
answerer, wired together end to end.

Gemini is always mocked -- no real API calls. RAG retrieval goes through a
small fake CodeRetriever (same pattern as test_qa_investigator.py) so these
tests stay fast and deterministic while still exercising real project
detection and real filesystem search/read against temporary fixtures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.services.qa.classifier import QuestionClass
from app.services.qa.investigator import _DEPTH_CONFIGS
from app.services.qa.service import QuestionValidationError, ask_codebase
from app.services.rag.retriever import RetrievalError, RetrievedChunk


def _chunk(file_path: str, symbol_name: Optional[str] = None) -> RetrievedChunk:
    return RetrievedChunk(
        file_path=file_path, symbol_name=symbol_name, symbol_type="function",
        start_line=1, end_line=5, source_code="stub", similarity_score=0.9,
    )


class _FakeRetriever:
    """Stands in for CodeRetriever, recording every call's kwargs."""

    def __init__(
        self,
        chunks_by_prefix: Optional[Dict[Optional[str], List[RetrievedChunk]]] = None,
        default_chunks: Optional[List[RetrievedChunk]] = None,
    ):
        self.calls: List[Dict] = []
        self._chunks_by_prefix = chunks_by_prefix or {}
        self._default_chunks = default_chunks or []

    async def retrieve_chunks(self, query, repository_id, top_k=None, file_prefix=None, db=None):
        self.calls.append(
            {"query": query, "repository_id": repository_id, "top_k": top_k, "file_prefix": file_prefix}
        )
        if file_prefix in self._chunks_by_prefix:
            return self._chunks_by_prefix[file_prefix]
        return self._default_chunks


class _FailingRetriever:
    """A retriever whose retrieve_chunks always raises RetrievalError."""

    async def retrieve_chunks(self, query, repository_id, top_k=None, file_prefix=None, db=None):
        raise RetrievalError("Gemini embedding API is down")


def _write(root: Path, rel_path: str, content: str) -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Full integration: classifier -> investigator -> answerer
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_integration_classifier_investigator_answerer(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaFakeRealLookingKey123")
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path, "src/cart.py", "def subtotal(items):\n    return sum(items)\n")

    classify_payload = {
        "kind": "lookup", "depth": "shallow",
        "subject_terms": ["subtotal"], "user_asserted_tech": [], "likely_multi_file": False,
    }
    answer_payload = {
        "summary": "subtotal() sums item prices.",
        "confidence": "direct_evidence",
        "evidence": [{"file_path": "src/cart.py", "start_line": 1, "end_line": 2, "symbol_name": "subtotal"}],
    }
    classify_response = MagicMock()
    classify_response.text = json.dumps(classify_payload)
    answer_response = MagicMock()
    answer_response.text = json.dumps(answer_payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [classify_response, answer_response]

    fake_retriever = _FakeRetriever(default_chunks=[_chunk("src/cart.py", "subtotal"), _chunk("src/cart.py", "total")])

    # google.genai is a single shared module object -- patching it here
    # covers both classifier.py's and answerer.py's `genai.Client` lookups.
    with patch("google.genai.Client", return_value=mock_client):
        answer = await ask_codebase(
            "Where is the cart subtotal calculated?", str(tmp_path), repository_id=1,
            db=None, retriever=fake_retriever,
        )

    assert answer.confidence == "direct_evidence"
    assert answer.evidence[0].file_path == "src/cart.py"
    assert answer.projects_considered == ["."]


# ---------------------------------------------------------------------------
# Depth is correctly threaded from classification into investigation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_service_passes_classified_depth_to_investigator(tmp_path, monkeypatch):
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")

    fake_qclass = QuestionClass(
        kind="architecture", depth="deep", subject_terms=[], user_asserted_tech=[], likely_multi_file=True
    )

    async def fake_classify(question):
        return fake_qclass

    monkeypatch.setattr("app.services.qa.service.classify_question", fake_classify)

    fake_retriever = _FakeRetriever(default_chunks=[])
    await ask_codebase(
        "Explain the architecture", str(tmp_path), repository_id=1, db=None, retriever=fake_retriever
    )

    assert fake_retriever.calls[0]["top_k"] == _DEPTH_CONFIGS["deep"].top_k


@pytest.mark.asyncio
async def test_classification_failure_uses_targeted_depth(tmp_path):
    """No real Gemini key configured (test env default) -> classify_question
    falls back internally; the service must still investigate at the safe
    targeted depth, never shallow."""
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path, "src/cart.py", "def subtotal(items):\n    return sum(items)\n")

    fake_retriever = _FakeRetriever(default_chunks=[_chunk("src/cart.py", "subtotal")])
    await ask_codebase(
        "Explain the architecture", str(tmp_path), repository_id=1, db=None, retriever=fake_retriever
    )

    assert fake_retriever.calls[0]["top_k"] == _DEPTH_CONFIGS["targeted"].top_k


# ---------------------------------------------------------------------------
# No evidence short-circuits the answerer's Gemini call entirely
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_evidence_short_circuits_gemini_answer_generation(tmp_path, monkeypatch):
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaFakeRealLookingKey123")

    async def fake_classify(question):
        return QuestionClass(
            kind="lookup", depth="shallow", subject_terms=[], user_asserted_tech=[], likely_multi_file=False
        )

    monkeypatch.setattr("app.services.qa.service.classify_question", fake_classify)

    fake_retriever = _FakeRetriever(default_chunks=[])

    with patch("google.genai.Client") as mock_ctor:
        answer = await ask_codebase(
            "Where is the nonexistent payment gateway?", str(tmp_path), repository_id=1,
            db=None, retriever=fake_retriever,
        )

    assert answer.confidence == "no_evidence"
    mock_ctor.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-project investigation, end to end
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multi_project_investigation_via_service(tmp_path, monkeypatch):
    _write(tmp_path, "backend/pom.xml", "<project></project>")
    _write(tmp_path, "frontend/package.json", json.dumps({"dependencies": {"react": "^18.0.0"}}))

    async def fake_classify(question):
        return QuestionClass(
            kind="lookup", depth="shallow", subject_terms=[], user_asserted_tech=[], likely_multi_file=False
        )

    monkeypatch.setattr("app.services.qa.service.classify_question", fake_classify)

    fake_retriever = _FakeRetriever(chunks_by_prefix={"frontend": [_chunk("frontend/src/App.jsx", "App")]})
    answer = await ask_codebase(
        "Explain the React App component", str(tmp_path), repository_id=1, db=None, retriever=fake_retriever
    )

    assert answer.projects_considered == ["frontend"]


# ---------------------------------------------------------------------------
# Errors propagate correctly
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_question_raises_validation_error(tmp_path):
    with pytest.raises(QuestionValidationError):
        await ask_codebase("   ", str(tmp_path), repository_id=1)


@pytest.mark.asyncio
async def test_retrieval_error_is_logged_not_silently_swallowed(tmp_path, caplog):
    """A RetrievalError from the RAG layer must not crash the service and
    must not be invisible -- it's logged, and the final result honestly
    reflects that no evidence was found rather than fabricating an answer."""
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")

    with caplog.at_level(logging.WARNING):
        answer = await ask_codebase(
            "Where is subtotal calculated?", str(tmp_path), repository_id=1,
            db=None, retriever=_FailingRetriever(),
        )

    assert answer.confidence == "no_evidence"
    assert any("RAG retrieval failed" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Part D wiring, end to end through the real orchestration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_classifier_subject_terms_reach_investigation_end_to_end(tmp_path, monkeypatch):
    """The classifier's subject_terms must survive the full
    classify_question -> ask_codebase -> investigate() path, not just be
    reachable when investigate() is called directly."""
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path, "src/payment_processor.py", "PAYMENTGATEWAY_TOKEN = 'x'\n")

    async def fake_classify(question):
        return QuestionClass(
            kind="lookup", depth="targeted",
            subject_terms=["PAYMENTGATEWAY_TOKEN"], user_asserted_tech=[], likely_multi_file=False,
        )

    monkeypatch.setattr("app.services.qa.service.classify_question", fake_classify)

    fake_retriever = _FakeRetriever(default_chunks=[_chunk("src/payment_processor.py")])
    answer = await ask_codebase(
        "How does this work?", str(tmp_path), repository_id=1, db=None, retriever=fake_retriever
    )

    assert answer.confidence != "no_evidence"
