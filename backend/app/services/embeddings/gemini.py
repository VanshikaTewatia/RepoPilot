"""Gemini embedding provider implementation using official google-genai SDK."""

import asyncio
from typing import List, Optional

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger
from app.services.embeddings.base import EmbeddingProvider, EmbeddingValidationError


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using Google Gemini's text-embedding models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self._api_key = api_key if api_key is not None else settings.gemini_api_key
        self._model_name = model_name or settings.gemini_embedding_model
        self._dimension = dimension if dimension is not None else settings.gemini_embedding_dimension
        self._client: Optional[genai.Client] = None

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        retry=retry_if_not_exception_type(EmbeddingValidationError),
    )
    def _call_embed_single_sync(self, text: str) -> List[float]:
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        retry=retry_if_not_exception_type(EmbeddingValidationError),
    )
    def _call_embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
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

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string asynchronously."""
        if not text.strip():
            return [0.0] * self._dimension

        # Return mock vector in test or mock environment
        if self._is_mock_or_test_key():
            return [0.01] * self._dimension

        vector = await asyncio.to_thread(self._call_embed_single_sync, text)
        self._validate_vector(vector)
        return vector

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 50,
    ) -> List[List[float]]:
        """Generate embeddings in chunked batches asynchronously.

        Guarantees exactly one embedding vector per input text. Raises
        ``EmbeddingValidationError`` if the provider returns fewer vectors than
        requested or any vector has the wrong dimension, instead of silently
        returning a partial result.
        """
        if not texts:
            return []

        # Return mock vectors in test or mock environment
        if self._is_mock_or_test_key():
            return [[0.01] * self._dimension for _ in texts]

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            empty_indices = {j for j, t in enumerate(batch) if not t.strip()}
            non_empty_texts = [t for j, t in enumerate(batch) if j not in empty_indices]

            batch_vectors: List[List[float]] = []
            if non_empty_texts:
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
