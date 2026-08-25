"""Abstract base interface for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingValidationError(ValueError):
    """Raised when an embedding provider returns a malformed or incomplete response."""


class EmbeddingRateLimitError(RuntimeError):
    """Raised when the provider's rate/quota limit could not be satisfied after bounded retries.

    ``retry_after`` carries the provider-supplied wait time (seconds), when known,
    so callers (e.g. the API layer) can surface it to clients via a Retry-After header.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class EmbeddingProvider(ABC):
    """Abstract interface for generating dense vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector embedding dimension size."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text string."""
        pass

    @abstractmethod
    async def embed_batch(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embedding vectors for a batch of text strings.

        If ``batch_size`` is omitted, implementations should fall back to their
        configured default (e.g. application settings) rather than a hardcoded value.
        """
        pass
