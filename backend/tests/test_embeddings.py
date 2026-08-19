"""Unit tests for Gemini embedding provider."""

from types import SimpleNamespace

import pytest
from google.genai.types import ContentEmbedding, EmbedContentResponse

from app.services.embeddings.base import EmbeddingValidationError
from app.services.embeddings.gemini import GeminiEmbeddingProvider


class FakeEmbedClient:
    """Mimics the google-genai client surface with a canned embed_content response."""

    def __init__(self, response_factory):
        self.models = SimpleNamespace(
            embed_content=lambda model, contents: response_factory(contents)
        )


def _provider_with_sdk_response(monkeypatch, response_factory) -> GeminiEmbeddingProvider:
    """Provider that bypasses the mock path and returns a real SDK-shaped response."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(provider, "_get_client", lambda: FakeEmbedClient(response_factory))
    return provider


def _sdk_response(values_list):
    """Build an EmbedContentResponse in the exact shape the installed SDK returns."""
    return EmbedContentResponse(
        embeddings=[ContentEmbedding(values=v) for v in values_list]
    )


@pytest.mark.asyncio
async def test_gemini_embedding_provider_initialization():
    """Test embedding provider dimension and test-mode embedding generation."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    assert provider.dimension == 3072

    vector = await provider.embed_text("sample code snippet")
    assert len(vector) == 3072


@pytest.mark.asyncio
async def test_gemini_embedding_batch():
    """Test batch embedding generation."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    texts = ["def foo(): pass", "def bar(): pass", "class Baz: pass"]
    vectors = await provider.embed_batch(texts, batch_size=2)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 3072


@pytest.mark.asyncio
async def test_embed_batch_success_returns_one_vector_per_text(monkeypatch):
    """A successful batch of N inputs returns exactly N valid vectors."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(
        provider,
        "_call_embed_batch_sync",
        lambda texts: [[0.01] * 3072 for _ in texts],
    )

    vectors = await provider.embed_batch(["a", "b", "c"], batch_size=2)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 3072


@pytest.mark.asyncio
async def test_embed_batch_incomplete_response_raises(monkeypatch):
    """An incomplete batch response must raise instead of returning partial vectors."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(
        provider,
        "_call_embed_batch_sync",
        lambda texts: [[0.01] * 3072] * (len(texts) - 1),
    )

    with pytest.raises(EmbeddingValidationError):
        await provider.embed_batch(["a", "b", "c"], batch_size=2)


@pytest.mark.asyncio
async def test_embed_batch_wrong_dimension_raises(monkeypatch):
    """A vector with the wrong dimension must raise."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(
        provider,
        "_call_embed_batch_sync",
        lambda texts: [[0.5] * 768 for _ in texts],
    )

    with pytest.raises(EmbeddingValidationError):
        await provider.embed_batch(["a", "b"])


@pytest.mark.asyncio
async def test_embed_text_wrong_dimension_raises(monkeypatch):
    """A single embedding with the wrong dimension must raise."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(provider, "_call_embed_single_sync", lambda text: [0.5] * 768)

    with pytest.raises(EmbeddingValidationError):
        await provider.embed_text("hello")


@pytest.mark.asyncio
async def test_embed_batch_empty_text_returns_zero_vector(monkeypatch):
    """Empty texts yield zero vectors and are not sent to the provider."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    captured: list = []

    def fake_sync(texts):
        captured.append(list(texts))
        return [[0.01] * 3072 for _ in texts]

    monkeypatch.setattr(provider, "_call_embed_batch_sync", fake_sync)

    vectors = await provider.embed_batch(["", "def foo(): pass"], batch_size=2)
    assert len(vectors) == 2
    assert vectors[0] == [0.0] * 3072
    assert len(vectors[1]) == 3072
    assert captured == [["def foo(): pass"]]


def test_sdk_response_has_no_embedding_attribute():
    """The installed SDK exposes `embeddings`, not `embedding` (regression guard)."""
    response = EmbedContentResponse(embeddings=[ContentEmbedding(values=[0.1])])
    assert response.embeddings is not None
    with pytest.raises(AttributeError):
        _ = response.embedding


@pytest.mark.asyncio
async def test_embed_text_uses_sdk_embeddings_field(monkeypatch):
    """Single-text embed extracts its vector from the SDK `embeddings` field."""
    provider = _provider_with_sdk_response(
        monkeypatch,
        lambda contents: _sdk_response([[0.01] * 3072]),
    )
    vector = await provider.embed_text("PaymentValidator")
    assert len(vector) == 3072


@pytest.mark.asyncio
async def test_embed_text_missing_embeddings_raises(monkeypatch):
    """A single-text response without embeddings must raise."""
    provider = _provider_with_sdk_response(
        monkeypatch, lambda contents: EmbedContentResponse()
    )
    with pytest.raises(EmbeddingValidationError):
        await provider.embed_text("PaymentValidator")


@pytest.mark.asyncio
async def test_embed_text_too_many_embeddings_raises(monkeypatch):
    """A single-text request returning multiple embeddings must raise."""
    provider = _provider_with_sdk_response(
        monkeypatch,
        lambda contents: _sdk_response([[0.01] * 3072, [0.02] * 3072]),
    )
    with pytest.raises(EmbeddingValidationError):
        await provider.embed_text("PaymentValidator")


@pytest.mark.asyncio
async def test_embed_text_embedding_without_values_raises(monkeypatch):
    """An embedding object without values must raise."""
    provider = _provider_with_sdk_response(
        monkeypatch,
        lambda contents: EmbedContentResponse(embeddings=[ContentEmbedding(values=None)]),
    )
    with pytest.raises(EmbeddingValidationError):
        await provider.embed_text("PaymentValidator")


@pytest.mark.asyncio
async def test_embed_batch_uses_sdk_embeddings_field(monkeypatch):
    """Batch embed returns exactly N vectors from the SDK `embeddings` field."""
    texts = ["def foo(): pass", "def bar(): pass", "class Baz: pass"]
    provider = _provider_with_sdk_response(
        monkeypatch,
        lambda contents: _sdk_response([[0.01] * 3072 for _ in contents]),
    )
    vectors = await provider.embed_batch(texts, batch_size=2)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 3072


@pytest.mark.asyncio
async def test_embed_batch_missing_embeddings_raises(monkeypatch):
    """A batch response without embeddings must raise."""
    provider = _provider_with_sdk_response(
        monkeypatch, lambda contents: EmbedContentResponse()
    )
    with pytest.raises(EmbeddingValidationError):
        await provider.embed_batch(["a", "b"], batch_size=2)


@pytest.mark.asyncio
async def test_embed_batch_sends_distinct_content_objects(monkeypatch):
    """Batch embed must construct separate types.Content objects per input text."""
    from google.genai.types import Content

    captured_contents = []

    def mock_embed_content(contents):
        captured_contents.append(contents)
        return _sdk_response([[0.01] * 3072 for _ in contents])

    provider = _provider_with_sdk_response(monkeypatch, mock_embed_content)
    texts = ["def foo(): pass", "def bar(): pass", "class Baz: pass"]
    vectors = await provider.embed_batch(texts, batch_size=50)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 3072

    assert len(captured_contents) == 1
    sent_list = captured_contents[0]
    assert len(sent_list) == 3
    assert all(isinstance(c, Content) for c in sent_list)
    assert sent_list[0].parts[0].text == "def foo(): pass"
    assert sent_list[1].parts[0].text == "def bar(): pass"
    assert sent_list[2].parts[0].text == "class Baz: pass"

