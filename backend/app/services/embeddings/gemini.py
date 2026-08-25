"""Gemini embedding provider implementation using official google-genai SDK."""

import asyncio
import time
from typing import Callable, List, Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger
from app.services.embeddings.base import (
    EmbeddingProvider,
    EmbeddingRateLimitError,
    EmbeddingValidationError,
)
from app.services.embeddings.rate_limiter import TokenPerMinuteRateLimiter, estimate_tokens


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True for Gemini 429 / RESOURCE_EXHAUSTED errors (the SDK's ClientError)."""
    return getattr(exc, "code", None) == 429 or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"


def _extract_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Read a Retry-After hint off the SDK error's underlying HTTP response, if any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    value = getter("Retry-After", None) or getter("retry-after", None)
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using Google Gemini's text-embedding models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        rate_limiter: Optional[TokenPerMinuteRateLimiter] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ):
        self._api_key = api_key if api_key is not None else settings.gemini_api_key
        self._model_name = model_name or settings.gemini_embedding_model
        self._dimension = dimension if dimension is not None else settings.gemini_embedding_dimension
        self._client: Optional[genai.Client] = None
        self._rate_limiter = rate_limiter or TokenPerMinuteRateLimiter(
            tpm_limit=settings.gemini_embedding_tpm_limit,
            window_seconds=settings.gemini_embedding_rate_limit_window_seconds,
        )
        # Injectable for tests: the retry loop runs inside a worker thread
        # (via asyncio.to_thread), so it uses a blocking sleep, not asyncio.sleep.
        self._sleep_fn = sleep_fn or time.sleep

    @property
    def dimension(self) -> int:
        return self._dimension

    def _is_mock_or_test_key(self) -> bool:
        return (
            settings.is_test
            or not self._api_key
            or self._api_key.startswith("test")
            or self._api_key.startswith("mock")
        )

    def _get_client(self) -> genai.Client:
        if not self._client:
            if not self._api_key:
                raise ValueError(
                    "Gemini API key is required. Set GEMINI_API_KEY environment variable."
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _validate_vector(self, vector: List[float], index: int = 0) -> None:
        """Validate a single embedding vector's shape and dimensionality."""
        if not isinstance(vector, (list, tuple)) or len(vector) == 0:
            raise EmbeddingValidationError(
                f"Embedding at index {index} is empty or missing; "
                f"expected a {self._dimension}-dimensional vector."
            )
        if len(vector) != self._dimension:
            raise EmbeddingValidationError(
                f"Embedding at index {index} has dimension {len(vector)}, "
                f"expected {self._dimension}."
            )

    def _run_with_bounded_retry(self, fn: Callable[[], List], attempt_label: str) -> List:
        """Run ``fn`` with bounded exponential backoff, honoring Retry-After on 429s.

        Never retries ``EmbeddingValidationError`` (a malformed response, not a
        transient failure). After exhausting the configured max attempts on a
        rate-limit error, raises ``EmbeddingRateLimitError`` with a clear message
        instead of hanging or surfacing an opaque SDK exception.
        """
        max_attempts = max(1, settings.gemini_embedding_max_retries)
        base_delay = settings.gemini_embedding_retry_base_seconds
        max_delay = settings.gemini_embedding_retry_max_seconds
        last_exc: Optional[BaseException] = None

        for attempt in range(1, max_attempts + 1):
            try:
                return fn()
            except EmbeddingValidationError:
                raise
            except Exception as exc:  # noqa: BLE001 - re-classified below
                last_exc = exc
                if attempt >= max_attempts:
                    break
                retry_after = _extract_retry_after_seconds(exc)
                if retry_after is not None:
                    delay = min(retry_after, max_delay)
                else:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"Gemini {attempt_label} embedding call failed "
                    f"(attempt {attempt}/{max_attempts}): {exc}. Retrying in {delay:.1f}s."
                )
                self._sleep_fn(delay)

        assert last_exc is not None
        if _is_rate_limit_error(last_exc):
            raise EmbeddingRateLimitError(
                f"Gemini embedding quota exceeded for model {self._model_name!r} after "
                f"{max_attempts} attempt(s); the tokens-per-minute limit could not be "
                f"satisfied within the bounded retry budget. Last error: {last_exc}",
                retry_after=_extract_retry_after_seconds(last_exc),
            ) from last_exc
        raise last_exc

    def _call_embed_single_once(self, text: str) -> List[float]:
        """Synchronous single text embedding call.

        The google-genai SDK normalizes every ``embed_content`` response into an
        ``EmbedContentResponse`` whose only embedding field is ``embeddings`` (a
        list). A single-text request must yield exactly one embedding.
        """
        client = self._get_client()
        result = client.models.embed_content(
            model=self._model_name,
            contents=text,
        )
        embeddings = result.embeddings
        if not embeddings:
            raise EmbeddingValidationError(
                "Embedding provider returned no embeddings for a single-text request."
            )
        if len(embeddings) != 1:
            raise EmbeddingValidationError(
                f"Embedding provider returned {len(embeddings)} embeddings for a "
                f"single-text request; expected exactly 1."
            )
        values = embeddings[0].values
        if not values:
            raise EmbeddingValidationError(
                "Embedding provider returned an embedding with no values for a "
                "single-text request."
            )
        return list(values)

    def _call_embed_single_sync(self, text: str) -> List[float]:
        """Bounded-retry wrapper around a single-text embedding call."""
        return self._run_with_bounded_retry(
            lambda: self._call_embed_single_once(text), attempt_label="single"
        )

    def _call_embed_batch_once(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding call.

        The google-genai SDK returns all embeddings (single or batch) under
        ``result.embeddings``. The caller validates that the returned count
        matches the number of input texts.
        """
        client = self._get_client()
        contents = [
            types.Content(parts=[types.Part.from_text(text=t)])
            for t in texts
        ]
        result = client.models.embed_content(
            model=self._model_name,
            contents=contents,
        )
        embeddings = result.embeddings
        if not embeddings:
            raise EmbeddingValidationError(
                f"Embedding provider returned no embeddings for {len(texts)} input texts."
            )
        vectors: List[List[float]] = []
        for e in embeddings:
            if not e.values:
                raise EmbeddingValidationError(
                    "Embedding provider returned an embedding with no values."
                )
            vectors.append(list(e.values))
        return vectors

    def _call_embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Bounded-retry wrapper around a batch embedding call."""
        return self._run_with_bounded_retry(
            lambda: self._call_embed_batch_once(texts), attempt_label="batch"
        )

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string asynchronously."""
        if not text.strip():
            return [0.0] * self._dimension

        # Return mock vector in test or mock environment
        if self._is_mock_or_test_key():
            return [0.01] * self._dimension

        await self._rate_limiter.acquire(estimate_tokens(text))
        vector = await asyncio.to_thread(self._call_embed_single_sync, text)
        self._validate_vector(vector)
        return vector

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Generate embeddings in chunked batches asynchronously.

        Batches are sent strictly sequentially, each paced by the configured
        tokens-per-minute rate limiter, so this method never fires requests
        faster than the Gemini embedding quota allows. Guarantees exactly one
        embedding vector per input text. Raises ``EmbeddingValidationError`` if
        the provider returns fewer vectors than requested or any vector has the
        wrong dimension, instead of silently returning a partial result.

        ``batch_size`` defaults to ``settings.gemini_embedding_batch_size`` when
        omitted, so callers don't need to hardcode a value.
        """
        if not texts:
            return []

        # Return mock vectors in test or mock environment
        if self._is_mock_or_test_key():
            return [[0.01] * self._dimension for _ in texts]

        if batch_size is None:
            batch_size = settings.gemini_embedding_batch_size

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            empty_indices = {j for j, t in enumerate(batch) if not t.strip()}
            non_empty_texts = [t for j, t in enumerate(batch) if j not in empty_indices]

            batch_vectors: List[List[float]] = []
            if non_empty_texts:
                batch_tokens = sum(estimate_tokens(t) for t in non_empty_texts)
                await self._rate_limiter.acquire(batch_tokens)
                batch_vectors = await asyncio.to_thread(
                    self._call_embed_batch_sync, non_empty_texts
                )
                if len(batch_vectors) != len(non_empty_texts):
                    raise EmbeddingValidationError(
                        f"Embedding provider returned {len(batch_vectors)} vectors for "
                        f"{len(non_empty_texts)} input texts; expected exactly one "
                        f"embedding vector per input text."
                    )
                for j, vector in enumerate(batch_vectors):
                    self._validate_vector(vector, index=j)

            batch_results: List[List[float]] = []
            non_empty_iter = iter(batch_vectors)
            for j in range(len(batch)):
                if j in empty_indices:
                    batch_results.append([0.0] * self._dimension)
                else:
                    batch_results.append(next(non_empty_iter))
            all_embeddings.extend(batch_results)

        return all_embeddings
