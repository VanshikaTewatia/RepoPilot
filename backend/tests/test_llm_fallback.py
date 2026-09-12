"""Tests for the centralized Gemini -> Groq generation fallback
(``app.services.llm``).

Covers the shared fallback core in isolation (rate-limit classification,
Groq success/failure after a genuine Gemini quota error, ordinary Gemini
failures never falling back, missing-Groq-key degrading clearly) plus one
call-site integration test proving a real caller (``diagnoser.diagnose``)
gets the fallback for free through the shared helper, and an async-
responsiveness proof that a Groq fallback call never blocks the event loop
either.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai.errors import ClientError

from app.core.config import settings
from app.services.llm.errors import LLMProvidersExhaustedError, is_gemini_rate_limit_error
from app.services.llm.fallback import generate_with_fallback, generate_with_fallback_sync


def _rate_limit_error(retry_after=None):
    """A real google-genai ClientError shaped like a 429 RESOURCE_EXHAUSTED
    response -- mirrors tests/test_embeddings.py's identical helper."""
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    response = SimpleNamespace(headers=headers)
    return ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}}, response)


def _groq_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


# ===========================================================================
# is_gemini_rate_limit_error classification
# ===========================================================================
def test_rate_limit_classifier_matches_real_resource_exhausted_error():
    assert is_gemini_rate_limit_error(_rate_limit_error()) is True


def test_rate_limit_classifier_rejects_generic_runtime_error():
    """A plain error that merely mentions "quota" in its message text must
    NOT be misclassified as a recognized rate-limit condition."""
    assert is_gemini_rate_limit_error(RuntimeError("quota exceeded")) is False


def test_rate_limit_classifier_rejects_ordinary_client_error():
    other = ClientError(400, {"error": {"status": "INVALID_ARGUMENT", "message": "bad request"}}, SimpleNamespace(headers={}))
    assert is_gemini_rate_limit_error(other) is False


# ===========================================================================
# generate_with_fallback (async) -- the 6 async call sites' shared core
# ===========================================================================
@pytest.mark.asyncio
async def test_gemini_success_never_calls_groq():
    mock_response = MagicMock()
    mock_response.text = "gemini answer"
    gemini_call = MagicMock(return_value=mock_response)

    with patch("app.services.llm.fallback.call_groq_async") as mock_groq:
        result = await generate_with_fallback(gemini_call, prompt="p", system_instruction="s")

    assert result == "gemini answer"
    gemini_call.assert_called_once()
    mock_groq.assert_not_called()


@pytest.mark.asyncio
async def test_gemini_rate_limit_falls_back_to_groq_exactly_once():
    gemini_call = MagicMock(side_effect=_rate_limit_error())

    with patch("app.core.config.settings.groq_api_key", "real_like_groq_key"):
        with patch(
            "app.services.llm.fallback.call_groq_async", return_value="groq answer"
        ) as mock_groq:
            result = await generate_with_fallback(gemini_call, prompt="p", system_instruction="s")

    assert result == "groq answer"
    mock_groq.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_ordinary_failure_never_falls_back_and_propagates():
    gemini_call = MagicMock(side_effect=RuntimeError("network timeout"))

    with patch("app.core.config.settings.groq_api_key", "real_like_groq_key"):
        with patch("app.services.llm.fallback.call_groq_async") as mock_groq:
            with pytest.raises(RuntimeError, match="network timeout"):
                await generate_with_fallback(gemini_call, prompt="p", system_instruction="s")

    mock_groq.assert_not_called()


@pytest.mark.asyncio
async def test_groq_failure_after_gemini_quota_raises_clean_error_without_secrets():
    gemini_call = MagicMock(side_effect=_rate_limit_error())

    with patch("app.core.config.settings.groq_api_key", "sk-super-secret-groq-key"):
        with patch(
            "app.services.llm.fallback.call_groq_async",
            side_effect=RuntimeError("Groq said no"),
        ):
            with pytest.raises(LLMProvidersExhaustedError) as exc_info:
                await generate_with_fallback(gemini_call, prompt="p", system_instruction="s")

    message = str(exc_info.value)
    assert "sk-super-secret-groq-key" not in message
    assert "Groq said no" not in message
    assert "try again later" in message.lower()


@pytest.mark.asyncio
async def test_gemini_quota_with_no_groq_key_configured_raises_clean_error():
    gemini_call = MagicMock(side_effect=_rate_limit_error())

    with patch("app.core.config.settings.groq_api_key", ""):
        with patch("app.services.llm.fallback.call_groq_async") as mock_groq:
            with pytest.raises(LLMProvidersExhaustedError) as exc_info:
                await generate_with_fallback(gemini_call, prompt="p", system_instruction="s")

    mock_groq.assert_not_called()
    assert "GROQ_API_KEY" in str(exc_info.value)


# ===========================================================================
# generate_with_fallback_sync -- graph.py's _generate_patches_with_gemini
# ===========================================================================
def test_sync_gemini_success_never_calls_groq():
    mock_response = MagicMock()
    mock_response.text = "gemini answer"
    gemini_call = MagicMock(return_value=mock_response)

    with patch("app.services.llm.fallback.call_groq_sync") as mock_groq:
        result = generate_with_fallback_sync(gemini_call, prompt="p", system_instruction="s")

    assert result == "gemini answer"
    mock_groq.assert_not_called()


def test_sync_gemini_rate_limit_falls_back_to_groq_exactly_once():
    gemini_call = MagicMock(side_effect=_rate_limit_error())

    with patch("app.core.config.settings.groq_api_key", "real_like_groq_key"):
        with patch(
            "app.services.llm.fallback.call_groq_sync", return_value="groq answer"
        ) as mock_groq:
            result = generate_with_fallback_sync(gemini_call, prompt="p", system_instruction="s")

    assert result == "groq answer"
    mock_groq.assert_called_once()


def test_sync_gemini_ordinary_failure_never_falls_back():
    gemini_call = MagicMock(side_effect=ValueError("malformed response"))

    with patch("app.core.config.settings.groq_api_key", "real_like_groq_key"):
        with patch("app.services.llm.fallback.call_groq_sync") as mock_groq:
            with pytest.raises(ValueError, match="malformed response"):
                generate_with_fallback_sync(gemini_call, prompt="p", system_instruction="s")

    mock_groq.assert_not_called()


# ===========================================================================
# Real Groq HTTP client (httpx) -- 429/error/success shapes
# ===========================================================================
@pytest.mark.asyncio
async def test_call_groq_async_success_extracts_message_content(monkeypatch):
    from app.services.llm import groq_client

    monkeypatch.setattr(settings, "groq_api_key", "real_like_groq_key")

    async def fake_post(self, url, json=None, headers=None):
        assert "Bearer real_like_groq_key" == headers["Authorization"]
        return _groq_response("groq generated text")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await groq_client.call_groq_async(prompt="hello", system_instruction="be terse")
    assert result == "groq generated text"


@pytest.mark.asyncio
async def test_call_groq_async_http_error_raises_without_leaking_key(monkeypatch):
    from app.services.llm import groq_client

    monkeypatch.setattr(settings, "groq_api_key", "sk-secret-value")

    async def fake_post(self, url, json=None, headers=None):
        return httpx.Response(
            429,
            text="rate limited",
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(groq_client.GroqAPIError) as exc_info:
        await groq_client.call_groq_async(prompt="hello", system_instruction="")

    assert "sk-secret-value" not in str(exc_info.value)


def test_call_groq_without_api_key_raises_not_configured():
    from app.services.llm import groq_client

    with patch("app.core.config.settings.groq_api_key", ""):
        with pytest.raises(groq_client.GroqNotConfiguredError):
            groq_client.call_groq_sync(prompt="hello", system_instruction="")


# ===========================================================================
# Structured-response compatibility: the fallback preserves the exact
# format a real agent generation path expects (JSON patch array text).
# ===========================================================================
def test_fallback_response_still_parses_as_a_valid_patch_array():
    """Groq's returned text must flow through the same
    parse_and_validate_patches path a Gemini success would -- proving the
    fallback doesn't require the agent's parsing logic to change."""
    import json

    from app.services.agent.graph import parse_and_validate_patches

    groq_text = json.dumps(
        [{"file_path": "src/order_service.py", "code": "return 1\n", "start_line": 1, "end_line": 1}]
    )
    patches = parse_and_validate_patches(groq_text, workspace_dir=None)
    assert len(patches) == 1
    assert patches[0]["file_path"] == "src/order_service.py"


# ===========================================================================
# Call-site integration: diagnose() gets the fallback through the shared
# helper with zero diagnoser-specific fallback code.
# ===========================================================================
@pytest.mark.asyncio
async def test_diagnose_falls_back_to_groq_on_real_gemini_quota_error():
    import json

    from app.services.diagnosis.diagnoser import diagnose

    payload = {
        "status": "diagnosed",
        "summary": "Root cause found via Groq fallback.",
        "hypotheses": [
            {
                "rank": 1,
                "description": "The bug is here.",
                "citations": [{"file_path": "src/a.py", "start_line": 1, "end_line": 2}],
                "suggested_fix_approach": None,
            }
        ],
        "confidence": "direct_evidence",
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _rate_limit_error()

    with patch("app.core.config.settings.gemini_api_key", "real_like_test_key_12345"):
        with patch("app.core.config.settings.groq_api_key", "real_like_groq_key"):
            with patch("app.services.diagnosis.diagnoser.genai.Client", return_value=mock_client):
                with patch(
                    "app.services.llm.fallback.call_groq_async",
                    return_value=json.dumps(payload),
                ) as mock_groq:
                    result = await diagnose(
                        "Fix the bug",
                        retrieved_context=[{"file_path": "src/a.py", "content": "code", "total_lines": 10}],
                    )

    mock_groq.assert_called_once()
    assert result.status.value == "DIAGNOSED"
    assert result.summary == "Root cause found via Groq fallback."


# ===========================================================================
# Async responsiveness: a Groq fallback call must not block the event loop
# either (mirrors test_agent_graph.py's identical Gemini-side proof).
# ===========================================================================
@pytest.mark.asyncio
async def test_groq_fallback_call_does_not_block_the_event_loop():
    import asyncio
    import time

    from app.services.llm import groq_client

    async def blocking_post(self, url, json=None, headers=None):
        # A real httpx.AsyncClient.post is a genuine, non-blocking await;
        # this stand-in proves the surrounding code doesn't accidentally
        # serialize it behind other work by awaiting a long blocking call
        # on the same thread instead of truly concurrently.
        await asyncio.sleep(0.3)
        return _groq_response("groq text")

    other_task_ran_at = {}

    async def other_task():
        await asyncio.sleep(0.05)
        other_task_ran_at["t"] = time.monotonic()

    with patch("app.core.config.settings.groq_api_key", "real_like_groq_key"):
        with patch.object(httpx.AsyncClient, "post", blocking_post):
            start = time.monotonic()
            _, _ = await asyncio.gather(
                groq_client.call_groq_async(prompt="p", system_instruction="s"), other_task()
            )

    assert other_task_ran_at["t"] - start < 0.2
