"""Centralized Gemini -> Groq generation fallback.

Every text-generation call site in the codebase (agent diagnosis, baseline
reproduction planning, patch planning, patch generation, Deep Q&A
classification/answering, and the legacy RAG answer path) follows the exact
same shape: build a prompt + system_instruction, call
``genai.Client(...).models.generate_content(...)``, read back ``.text``. None
of them use Gemini's native structured output, tool-calling, or streaming --
see the investigation report for the full survey -- so this module only
needs to unify on that one shape.

This module deliberately does NOT construct the Gemini client itself; each
call site still builds its own ``genai.Client(...)`` and binds it into a
zero-arg callable (so every call site's existing prompt/model/config
construction, and every existing test that patches
``app.services.<module>.genai.Client`` and inspects
``mock_client.models.generate_content.call_args.kwargs``, is completely
unaffected on the Gemini-success and Gemini-ordinary-failure paths). This
module owns exactly the part that must not be duplicated: classifying
whether a Gemini failure is a genuine quota/rate-limit condition, and if so,
calling Groq and turning any failure there into one clear, credential-free
error.

Two entry points, matching the two calling conventions already in the
codebase (see ``app.services.llm.groq_client`` module docstring for why):

- ``generate_with_fallback`` (async) -- for the 6 ``async def`` call sites,
  which already offload the blocking Gemini SDK call via
  ``asyncio.to_thread`` themselves; this function does that offloading
  internally instead, so callers just ``await`` it.
- ``generate_with_fallback_sync`` -- for
  ``app.services.agent.graph._generate_patches_with_gemini``, which must stay
  a synchronous function (it's directly unit-tested as one, and it's already
  offloaded by its own caller's ``asyncio.to_thread``). Calling the Gemini
  SDK directly here is safe precisely because this function is only ever
  invoked from inside that worker thread, never from the main event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from app.core.config import settings
from app.core.logging import logger
from app.services.llm.errors import LLMProvidersExhaustedError, is_gemini_rate_limit_error
from app.services.llm.groq_client import GroqRequestTooLargeError, call_groq_async, call_groq_sync

_NO_FALLBACK_CONFIGURED_MESSAGE = (
    "Gemini's quota/rate limit was exhausted and no fallback provider is "
    "configured. Wait for the Gemini quota to reset, or configure "
    "GROQ_API_KEY to enable automatic fallback."
)
_BOTH_PROVIDERS_FAILED_MESSAGE = (
    "Gemini's quota/rate limit was exhausted and the Groq fallback provider "
    "also failed to generate a response. Please try again later."
)
# Deliberately distinct from _BOTH_PROVIDERS_FAILED_MESSAGE: a request too
# large for the fallback's rate-limit tier is a structurally different
# condition from a genuine provider outage or true rate limit -- "try
# again later" is actively misleading here, since waiting can never help a
# request that is simply too big; only a smaller/narrower request can.
_REQUEST_TOO_LARGE_MESSAGE = (
    "Gemini's quota/rate limit was exhausted, and this request is too large "
    "for the fallback provider to handle. Try a smaller or more specific "
    "request."
)


async def generate_with_fallback(
    gemini_call: Callable[[], Any],
    *,
    prompt: str,
    system_instruction: str = "",
    groq_model: Optional[str] = None,
) -> str:
    """Run ``gemini_call`` (a zero-arg callable bound to
    ``client.models.generate_content(...)``) off the event loop, returning
    ``response.text``. Falls back to Groq only when ``gemini_call`` raises a
    recognized 429/RESOURCE_EXHAUSTED condition; any other exception
    propagates unchanged so existing callers' ordinary-failure handling is
    untouched.

    ``groq_model`` (when given) overrides ``settings.groq_model_name`` for
    the fallback call only -- it is NOT the Gemini model; each call site
    still passes its own Gemini model directly to ``gemini_call``.
    """
    try:
        response = await asyncio.to_thread(gemini_call)
        return response.text or ""
    except Exception as e:  # noqa: BLE001 -- re-raised as-is unless it's a recognized rate-limit condition
        if not is_gemini_rate_limit_error(e):
            raise
        logger.warning(f"Gemini quota/rate-limit hit; attempting Groq fallback. Gemini error: {e}")
        if not settings.groq_api_key:
            raise LLMProvidersExhaustedError(_NO_FALLBACK_CONFIGURED_MESSAGE) from e
        try:
            return await call_groq_async(
                prompt=prompt, system_instruction=system_instruction, groq_model=groq_model
            )
        except GroqRequestTooLargeError as groq_exc:
            logger.warning(f"Groq fallback skipped (request too large): {groq_exc}")
            raise LLMProvidersExhaustedError(_REQUEST_TOO_LARGE_MESSAGE) from groq_exc
        except Exception as groq_exc:  # noqa: BLE001 -- any other Groq failure is wrapped, never leaked raw
            logger.error(f"Groq fallback also failed after Gemini quota exhaustion: {groq_exc}")
            raise LLMProvidersExhaustedError(_BOTH_PROVIDERS_FAILED_MESSAGE) from groq_exc


def generate_with_fallback_sync(
    gemini_call: Callable[[], Any],
    *,
    prompt: str,
    system_instruction: str = "",
    groq_model: Optional[str] = None,
) -> str:
    """Synchronous counterpart of ``generate_with_fallback`` for the one
    call site that must stay a plain function -- see module docstring.

    ``groq_model`` (when given) overrides ``settings.groq_model_name`` for
    the fallback call only -- it is NOT the Gemini model.
    """
    try:
        response = gemini_call()
        return response.text or ""
    except Exception as e:  # noqa: BLE001 -- re-raised as-is unless it's a recognized rate-limit condition
        if not is_gemini_rate_limit_error(e):
            raise
        logger.warning(f"Gemini quota/rate-limit hit; attempting Groq fallback. Gemini error: {e}")
        if not settings.groq_api_key:
            raise LLMProvidersExhaustedError(_NO_FALLBACK_CONFIGURED_MESSAGE) from e
        try:
            return call_groq_sync(
                prompt=prompt, system_instruction=system_instruction, groq_model=groq_model
            )
        except GroqRequestTooLargeError as groq_exc:
            logger.warning(f"Groq fallback skipped (request too large): {groq_exc}")
            raise LLMProvidersExhaustedError(_REQUEST_TOO_LARGE_MESSAGE) from groq_exc
        except Exception as groq_exc:  # noqa: BLE001 -- any other Groq failure is wrapped, never leaked raw
            logger.error(f"Groq fallback also failed after Gemini quota exhaustion: {groq_exc}")
            raise LLMProvidersExhaustedError(_BOTH_PROVIDERS_FAILED_MESSAGE) from groq_exc
