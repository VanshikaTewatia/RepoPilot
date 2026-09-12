"""Shared error types and Gemini rate-limit classification for the generation
fallback path (``app.services.llm.fallback``).

Deliberately NOT imported by ``app.services.embeddings.gemini`` -- embeddings
keep their own, separate ``_is_rate_limit_error``/``EmbeddingRateLimitError``
so embedding behavior is never affected by changes made here. The detection
logic is intentionally duplicated (not shared) to keep the two providers
fully independent, per the fallback project's explicit "don't touch
embeddings" requirement; the two copies check the exact same google-genai
``ClientError`` shape (``.code == 429`` / ``.status == "RESOURCE_EXHAUSTED"``).
"""

from __future__ import annotations


def is_gemini_rate_limit_error(exc: BaseException) -> bool:
    """True only for a recognized Gemini 429 / RESOURCE_EXHAUSTED condition.

    Mirrors ``app.services.embeddings.gemini._is_rate_limit_error`` exactly.
    Any other exception (network error, malformed response, ordinary 4xx/5xx,
    a ``RuntimeError`` that merely mentions "quota" in its message, etc.)
    returns False -- those must propagate as ordinary Gemini failures and
    must never trigger a Groq fallback.
    """
    return getattr(exc, "code", None) == 429 or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"


class LLMProvidersExhaustedError(RuntimeError):
    """Raised when Gemini hit a recognized quota/rate-limit condition and the
    Groq fallback also failed (or Groq isn't configured at all).

    The message is always a clean, actionable, credential-free summary --
    never the raw provider exception text or headers -- so it's safe to
    surface in a ``failure_reason``/error field that may reach a frontend.
    The original exception chain (via ``raise ... from ...``) still carries
    full diagnostic detail for server-side logs.
    """
