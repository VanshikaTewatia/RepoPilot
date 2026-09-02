"""Unit tests for CodeRetriever and RAG synthesis."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from app.core.config import settings
from app.services.rag.retriever import CodeRetriever, RetrievedChunk, RetrievalError


def _mock_db_capturing_statement(captured: dict) -> MagicMock:
    """A fake AsyncSession whose execute() records the compiled SELECT
    statement (so tests can assert on LIMIT/WHERE) and returns zero rows."""

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        result = MagicMock()
        result.all.return_value = []
        return result

    db = MagicMock()
    db.execute = AsyncMock(side_effect=fake_execute)
    return db


def _compiled_sql(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


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


# ---------------------------------------------------------------------------
# vector_top_k wiring (Phase 0.2): the retrieval limit is configurable via
# settings, not a hardcoded literal, while remaining overridable per call.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retrieve_chunks_defaults_top_k_from_settings():
    """Omitting top_k must use settings.vector_top_k, not a hardcoded 5."""
    captured: dict = {}
    db = _mock_db_capturing_statement(captured)

    retriever = CodeRetriever()
    await retriever.retrieve_chunks(query="where is auth handled", repository_id=1, db=db)

    assert f"LIMIT {settings.vector_top_k}" in _compiled_sql(captured["stmt"])


@pytest.mark.asyncio
async def test_retrieve_chunks_explicit_top_k_overrides_settings_default():
    captured: dict = {}
    db = _mock_db_capturing_statement(captured)

    retriever = CodeRetriever()
    await retriever.retrieve_chunks(query="where is auth handled", repository_id=1, top_k=3, db=db)

    assert "LIMIT 3" in _compiled_sql(captured["stmt"])


@pytest.mark.asyncio
async def test_answer_question_defaults_top_k_from_settings():
    """answer_question() shares the same default-resolution as retrieve_chunks()."""
    captured: dict = {}
    db = _mock_db_capturing_statement(captured)

    retriever = CodeRetriever()
    await retriever.answer_question(query="where is auth handled", repository_id=1, db=db)

    assert f"LIMIT {settings.vector_top_k}" in _compiled_sql(captured["stmt"])


# ---------------------------------------------------------------------------
# file_prefix scoping (Phase 1 prerequisite): additive, existing callers
# (no file_prefix passed) must see byte-identical, unscoped queries.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retrieve_chunks_file_prefix_scopes_to_project_root():
    captured: dict = {}
    db = _mock_db_capturing_statement(captured)

    retriever = CodeRetriever()
    await retriever.retrieve_chunks(
        query="cart subtotal", repository_id=1, file_prefix="frontend", db=db
    )

    sql = _compiled_sql(captured["stmt"])
    assert "LIKE" in sql
    assert "'frontend/'" in sql


@pytest.mark.asyncio
async def test_retrieve_chunks_no_file_prefix_is_unscoped():
    """Default (no file_prefix) behaves exactly as before: no LIKE filter at all."""
    captured: dict = {}
    db = _mock_db_capturing_statement(captured)

    retriever = CodeRetriever()
    await retriever.retrieve_chunks(query="cart subtotal", repository_id=1, db=db)

    assert "LIKE" not in _compiled_sql(captured["stmt"])


@pytest.mark.asyncio
async def test_retrieve_chunks_root_file_prefix_is_unscoped():
    """A project rooted at "." (the whole repo) must not add a self-defeating filter."""
    captured: dict = {}
    db = _mock_db_capturing_statement(captured)

    retriever = CodeRetriever()
    await retriever.retrieve_chunks(query="cart subtotal", repository_id=1, file_prefix=".", db=db)

    assert "LIKE" not in _compiled_sql(captured["stmt"])
