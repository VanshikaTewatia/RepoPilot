"""Abstract base interface for embedding providers."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingValidationError(ValueError):
    """Raised when an embedding provider returns a malformed or incomplete response."""


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
    async def embed_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Generate embedding vectors for a batch of text strings."""
        pass
