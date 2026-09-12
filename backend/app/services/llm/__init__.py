"""Text-generation provider fallback (Gemini primary, Groq fallback on quota/rate-limit only).

Embeddings are a separate concern and are never touched by this package --
see ``app.services.embeddings`` for the Gemini-only embedding provider.
"""

from app.services.llm.errors import LLMProvidersExhaustedError, is_gemini_rate_limit_error
from app.services.llm.fallback import generate_with_fallback, generate_with_fallback_sync

__all__ = [
    "LLMProvidersExhaustedError",
    "is_gemini_rate_limit_error",
    "generate_with_fallback",
    "generate_with_fallback_sync",
]
