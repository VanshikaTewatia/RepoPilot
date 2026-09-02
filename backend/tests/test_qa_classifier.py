"""Unit tests for question classification (app.services.qa.classifier).

Gemini is always mocked -- no real API calls. A "real-looking" (non-test/
mock-prefixed) API key is set via monkeypatch only for tests that need to
exercise the actual Gemini-call code path; tests that want the safe
fallback path rely on the test environment's own default key
(conftest.py's GEMINI_API_KEY=test_gemini_key_123, which classify_question
correctly treats as "not configured").
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.services.qa.classifier import FALLBACK_DEPTH, classify_question


@pytest.fixture
def real_looking_key(monkeypatch):
    """A key that doesn't start with 'test'/'mock', so classify_question
    actually attempts a (mocked) Gemini call instead of short-circuiting."""
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaFakeRealLookingKey123")


def _mock_gemini_response(payload: dict):
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return patch("app.services.qa.classifier.genai.Client", return_value=mock_client)


# ---------------------------------------------------------------------------
# Each required kind/depth mapping
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_classify_lookup_question(real_looking_key):
    payload = {
        "kind": "lookup", "depth": "shallow",
        "subject_terms": ["cart", "subtotal"], "user_asserted_tech": [], "likely_multi_file": False,
    }
    with _mock_gemini_response(payload):
        result = await classify_question("Where is the cart subtotal calculated?")

    assert result.kind == "lookup"
    assert result.depth == "shallow"
    assert result.classification_failed is False


@pytest.mark.asyncio
async def test_classify_symbol_question(real_looking_key):
    payload = {
        "kind": "symbol", "depth": "targeted",
        "subject_terms": ["calculateSubtotal"], "user_asserted_tech": [], "likely_multi_file": False,
    }
    with _mock_gemini_response(payload):
        result = await classify_question("What does calculateSubtotal do?")

    assert result.kind == "symbol"
    assert result.depth == "targeted"


@pytest.mark.asyncio
async def test_classify_flow_question(real_looking_key):
    payload = {
        "kind": "flow", "depth": "medium",
        "subject_terms": ["payment"], "user_asserted_tech": [], "likely_multi_file": True,
    }
    with _mock_gemini_response(payload):
        result = await classify_question("How does the payment flow work?")

    assert result.kind == "flow"
    assert result.depth in ("medium", "deep")
    assert result.likely_multi_file is True


@pytest.mark.asyncio
async def test_classify_architecture_question(real_looking_key):
    payload = {
        "kind": "architecture", "depth": "deep",
        "subject_terms": ["authentication"], "user_asserted_tech": [], "likely_multi_file": True,
    }
    with _mock_gemini_response(payload):
        result = await classify_question("Explain how authentication works in this project.")

    assert result.kind == "architecture"
    assert result.depth == "deep"


@pytest.mark.asyncio
async def test_classify_existence_check_question(real_looking_key):
    payload = {
        "kind": "existence_check", "depth": "targeted",
        "subject_terms": ["authentication", "component"], "user_asserted_tech": ["React"], "likely_multi_file": False,
    }
    with _mock_gemini_response(payload):
        result = await classify_question("Is there a React authentication component?")

    assert result.kind == "existence_check"
    assert result.depth in ("targeted", "deep")


# ---------------------------------------------------------------------------
# User terminology is captured verbatim, never treated as fact
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_user_asserted_tech_is_captured_but_not_validated(real_looking_key):
    """The classifier extracts the user's claimed tech verbatim without
    attempting to verify it -- verification against real repository
    evidence is the investigator/answerer's job, not the classifier's."""
    payload = {
        "kind": "existence_check", "depth": "targeted",
        "subject_terms": ["authentication"], "user_asserted_tech": ["React"], "likely_multi_file": False,
    }
    with _mock_gemini_response(payload):
        result = await classify_question("Is there a React authentication component?")

    assert result.user_asserted_tech == ["React"]


# ---------------------------------------------------------------------------
# Failure handling: never falls back to "shallow"
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_classify_invalid_json_falls_back_to_targeted(real_looking_key):
    mock_response = MagicMock()
    mock_response.text = "this is not valid json at all {{{"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.qa.classifier.genai.Client", return_value=mock_client):
        result = await classify_question("Where is the cart subtotal calculated?")

    assert result.classification_failed is True
    assert result.depth == FALLBACK_DEPTH
    assert result.depth != "shallow"
    assert result.failure_reason is not None


@pytest.mark.asyncio
async def test_classify_invalid_schema_falls_back_to_targeted(real_looking_key):
    """Valid JSON, but 'kind' isn't one of the five allowed values."""
    payload = {
        "kind": "not-a-real-kind", "depth": "shallow",
        "subject_terms": [], "user_asserted_tech": [], "likely_multi_file": False,
    }
    with _mock_gemini_response(payload):
        result = await classify_question("Where is the cart subtotal calculated?")

    assert result.classification_failed is True
    assert result.depth == FALLBACK_DEPTH


@pytest.mark.asyncio
async def test_classify_gemini_exception_falls_back_to_targeted(real_looking_key):
    """A network/quota error (Gemini unavailable mid-call) must not raise
    and must not silently default to shallow."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

    with patch("app.services.qa.classifier.genai.Client", return_value=mock_client):
        result = await classify_question("Explain how authentication works in this project.")

    assert result.classification_failed is True
    assert result.depth == "targeted"
    assert "quota" in result.failure_reason


@pytest.mark.asyncio
async def test_classify_without_configured_key_falls_back_to_targeted():
    """No monkeypatched key -- the test environment's own default
    (GEMINI_API_KEY=test_gemini_key_123) must be treated as unconfigured."""
    result = await classify_question("Explain how authentication works in this project.")

    assert result.classification_failed is True
    assert result.depth == "targeted"


@pytest.mark.asyncio
async def test_classify_empty_question_falls_back():
    result = await classify_question("   ")

    assert result.classification_failed is True
    assert result.depth == "targeted"


# ---------------------------------------------------------------------------
# Fallback user_asserted_tech extraction: on classification failure, the
# deterministic corrected_premise check (app.services.qa.answerer) still
# needs an explicit premise to compare against real repository evidence.
# ---------------------------------------------------------------------------
async def test_fallback_extracts_vue_on_classification_failure():
    """No configured key -> classification fails -> falls back -- Vue must
    still be captured as the asserted premise."""
    result = await classify_question("Explain the Vue authentication component.")

    assert result.classification_failed is True
    assert result.user_asserted_tech == ["Vue"]


async def test_fallback_extracts_react_on_classification_failure():
    result = await classify_question("Explain the React authentication component.")

    assert result.classification_failed is True
    assert result.user_asserted_tech == ["React"]


async def test_fallback_extracts_no_technology_when_none_mentioned():
    result = await classify_question("Explain the architecture of this frontend project.")

    assert result.classification_failed is True
    assert result.user_asserted_tech == []


async def test_fallback_extraction_is_case_insensitive():
    result = await classify_question("EXPLAIN THE react COMPONENT")

    assert result.user_asserted_tech == ["React"]


async def test_fallback_extraction_handles_node_and_dotnet_aliases():
    node_result = await classify_question("Is this built with node or Node.js?")
    assert node_result.user_asserted_tech == ["Node.js"]

    dotnet_result = await classify_question("Is this a .NET or C# project?")
    assert set(dotnet_result.user_asserted_tech) == {".NET", "C#"}


async def test_fallback_extraction_does_not_confuse_java_and_javascript():
    java_only = await classify_question("Explain the Java authentication logic.")
    assert java_only.user_asserted_tech == ["Java"]

    js_only = await classify_question("Explain the JavaScript authentication logic.")
    assert js_only.user_asserted_tech == ["JavaScript"]


@pytest.mark.asyncio
async def test_successful_classification_user_asserted_tech_is_not_overwritten(real_looking_key):
    """When Gemini classification succeeds, the deterministic fallback
    extractor must never run or alter its output -- the classifier's own
    (verbatim, unvalidated) user_asserted_tech is preserved exactly."""
    payload = {
        "kind": "existence_check", "depth": "targeted",
        "subject_terms": ["authentication"],
        # Deliberately different from what the deterministic extractor
        # would produce for this question, to prove it's untouched.
        "user_asserted_tech": ["React", "some-made-up-framework"],
        "likely_multi_file": False,
    }
    with _mock_gemini_response(payload):
        result = await classify_question("Is there a Vue authentication component?")

    assert result.classification_failed is False
    assert result.user_asserted_tech == ["React", "some-made-up-framework"]
