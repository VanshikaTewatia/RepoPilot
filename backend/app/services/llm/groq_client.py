"""Minimal Groq chat-completions client used only as the generation fallback.

Groq exposes an OpenAI-compatible REST API
(``POST {base_url}/chat/completions``); every current Gemini call site in
this codebase only ever asks for free-text JSON via a ``system_instruction``
+ prompt (see ``app.services.llm.fallback`` module docstring for the survey),
so this client only needs to expose the same shape: system instruction +
prompt in, raw completion text out. No response_schema/tool-calling/
streaming is used because none of the Gemini call sites use them either.

Uses ``httpx`` (already a project dependency) rather than the ``groq`` SDK,
per the approved plan -- avoids adding a new dependency. Two entry points
mirror the split already used by ``app.services.embeddings.gemini`` (a sync
core wrapped for the async call sites) and by
``app.services.agent.graph._generate_patches_with_gemini`` (a genuinely
synchronous call site that's already offloaded via ``asyncio.to_thread`` by
its own caller, so it must not need an event loop of its own):

- ``call_groq_async`` -- used by the 6 ``async def`` Gemini call sites; makes
  a real async HTTP request (``httpx.AsyncClient``), so it never blocks the
  FastAPI event loop.
- ``call_groq_sync`` -- used only by ``_generate_patches_with_gemini``, which
  is itself already running inside a worker thread (via the caller's
  ``asyncio.to_thread``); a plain blocking ``httpx.Client`` call here is safe
  because it never touches the main event loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings


class GroqNotConfiguredError(RuntimeError):
    """Raised when a Groq call is attempted but GROQ_API_KEY is unset."""


class GroqAPIError(RuntimeError):
    """A Groq request failed. The message is truncated and never includes
    the Authorization header or API key -- safe to include in logs, but
    callers should still not surface it verbatim to end users (see
    ``app.services.llm.fallback``, which wraps this into a clean,
    credential-free ``LLMProvidersExhaustedError`` before it can reach an
    API response)."""


def _build_messages(system_instruction: str, prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    return messages


def _request_payload(prompt: str, system_instruction: str, groq_model: Optional[str]) -> Dict[str, Any]:
    return {
        "model": groq_model or settings.groq_model_name,
        "messages": _build_messages(system_instruction, prompt),
    }


def _require_api_key() -> str:
    api_key = settings.groq_api_key
    if not api_key:
        raise GroqNotConfiguredError("GROQ_API_KEY is not configured.")
    return api_key


def _extract_content(response: httpx.Response) -> str:
    if response.status_code != 200:
        # Truncate defensively: a provider error body could in principle
        # echo request content back; never let it grow unbounded into logs.
        raise GroqAPIError(
            f"Groq request failed with HTTP {response.status_code}: {response.text[:500]}"
        )
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise GroqAPIError(f"Groq response could not be parsed: {e}") from e


async def call_groq_async(
    *, prompt: str, system_instruction: str = "", groq_model: Optional[str] = None
) -> str:
    """Async Groq chat-completion call. Raises ``GroqNotConfiguredError`` or
    ``GroqAPIError`` on failure; never returns a partial/guessed result."""
    api_key = _require_api_key()
    payload = _request_payload(prompt, system_instruction, groq_model)
    async with httpx.AsyncClient(
        base_url=settings.groq_api_base_url, timeout=settings.groq_timeout_seconds
    ) as client:
        response = await client.post(
            "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    return _extract_content(response)


def call_groq_sync(
    *, prompt: str, system_instruction: str = "", groq_model: Optional[str] = None
) -> str:
    """Synchronous Groq chat-completion call for the one call site
    (``_generate_patches_with_gemini``) that's already running off the main
    event loop inside a worker thread -- see module docstring."""
    api_key = _require_api_key()
    payload = _request_payload(prompt, system_instruction, groq_model)
    with httpx.Client(
        base_url=settings.groq_api_base_url, timeout=settings.groq_timeout_seconds
    ) as client:
        response = client.post(
            "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    return _extract_content(response)
