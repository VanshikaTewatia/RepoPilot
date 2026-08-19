"""Unit tests for CodeRetriever and RAG synthesis."""

import pytest
from app.services.rag.retriever import CodeRetriever, RetrievedChunk, RetrievalError


def test_retrieved_chunk_citation_format():
    """Test citation formatting."""
    chunk = RetrievedChunk(
        file_path="src/calculator.py",
        symbol_name="Calculator.add",
        symbol_type="method",
        start_line=12,
        end_line=18,
        source_code="def add(self, x):\n    return x + 1\n",
        similarity_score=0.942,
    )
    assert chunk.citation == "src/calculator.py:12-18"


@pytest.mark.asyncio
async def test_code_retriever_empty_query():
    """Test retrieving with empty query returns empty list."""
    retriever = CodeRetriever()
    results = await retriever.retrieve_chunks(query="", repository_id=1, db=None)
    assert results == []


@pytest.mark.asyncio
async def test_answer_question_without_chunks():
    """Test answer synthesis when no chunks are found."""
    retriever = CodeRetriever()
    res = await retriever.answer_question(query="where is main?", repository_id=1, db=None)
    assert "No relevant code chunks" in res["answer"]
    assert res["citations"] == []


class FailingEmbeddingProvider:
    """Provider that fails to embed the query, simulating a Gemini outage."""

    dimension = 3072

    async def embed_text(self, text):
        raise RuntimeError("Gemini API call failed")

    async def embed_batch(self, texts, batch_size=50):
        return [[0.01] * self.dimension for _ in texts]


@pytest.mark.asyncio
async def test_retrieve_chunks_embedding_failure_raises_retrieval_error():
    """Query embedding failure must surface as a typed RetrievalError, not a raw 500."""
    retriever = CodeRetriever(embedding_provider=FailingEmbeddingProvider())
    with pytest.raises(RetrievalError):
        await retriever.retrieve_chunks(query="PaymentValidator", repository_id=1, db=object())


@pytest.mark.asyncio
async def test_answer_question_handles_embedding_failure_gracefully():
    """A failed query embedding must not propagate an unhandled exception."""
    retriever = CodeRetriever(embedding_provider=FailingEmbeddingProvider())
    res = await retriever.answer_question(
        query="PaymentValidator", repository_id=1, db=object()
    )

    assert "Could not perform semantic retrieval" in res["answer"]
    assert res["retrieved_chunks"] == []
    assert res["citations"] == []
