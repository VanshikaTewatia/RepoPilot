"""Unit tests for Gemini embedding provider."""

from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError
from google.genai.types import ContentEmbedding, EmbedContentResponse

from app.core.config import settings
from app.services.embeddings.base import EmbeddingRateLimitError, EmbeddingValidationError
from app.services.embeddings.gemini import GeminiEmbeddingProvider
from app.services.embeddings.rate_limiter import TokenPerMinuteRateLimiter


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


# ---------------------------------------------------------------------------
# 429 retry, Retry-After handling, and bounded-retry behavior
# ---------------------------------------------------------------------------


def _rate_limit_error(retry_after=None):
    """Build a real google-genai ClientError shaped like a 429 RESOURCE_EXHAUSTED response."""
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    response = SimpleNamespace(headers=headers)
    return ClientError(
        429,
        {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}},
        response,
    )


class FlakyBatchClient:
    """Mimics the SDK client, raising queued exceptions before eventually succeeding."""

    def __init__(self, exceptions):
        self._exceptions = list(exceptions)
        self.call_count = 0
        self.models = SimpleNamespace(embed_content=self._embed_content)

    def _embed_content(self, model, contents):
        self.call_count += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        return _sdk_response([[0.01] * 3072 for _ in contents])


class AlwaysFailingBatchClient:
    """Mimics the SDK client, always raising the same exception."""

    def __init__(self, make_exception):
        self._make_exception = make_exception
        self.call_count = 0
        self.models = SimpleNamespace(embed_content=self._embed_content)

    def _embed_content(self, model, contents):
        self.call_count += 1
        raise self._make_exception()


def _provider_for_retry_tests(monkeypatch, client, max_retries=5):
    """A provider wired to a fake client with sleeping replaced by a no-op recorder."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(provider, "_get_client", lambda: client)
    sleeps: list = []
    monkeypatch.setattr(provider, "_sleep_fn", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(settings, "gemini_embedding_max_retries", max_retries)
    return provider, sleeps


@pytest.mark.asyncio
async def test_batch_retries_on_429_then_succeeds(monkeypatch):
    """A 429 RESOURCE_EXHAUSTED on the first attempt is retried and eventually succeeds."""
    client = FlakyBatchClient([_rate_limit_error()])
    provider, sleeps = _provider_for_retry_tests(monkeypatch, client)

    vectors = await provider.embed_batch(["a", "b"], batch_size=2)

    assert len(vectors) == 2
    assert all(len(v) == 3072 for v in vectors)
    assert client.call_count == 2  # one failure, one success
    assert len(sleeps) == 1  # backed off exactly once before retrying


@pytest.mark.asyncio
async def test_batch_retry_respects_retry_after_header(monkeypatch):
    """When the API supplies Retry-After, the backoff delay uses that value, not the default."""
    client = FlakyBatchClient([_rate_limit_error(retry_after=17)])
    provider, sleeps = _provider_for_retry_tests(monkeypatch, client)

    vectors = await provider.embed_batch(["a"], batch_size=2)

    assert len(vectors) == 1
    assert sleeps == [17.0]


@pytest.mark.asyncio
async def test_batch_retry_after_header_capped_at_max_delay(monkeypatch):
    """A Retry-After value larger than the configured max delay is capped, not honored blindly."""
    client = FlakyBatchClient([_rate_limit_error(retry_after=500)])
    provider, sleeps = _provider_for_retry_tests(monkeypatch, client)
    monkeypatch.setattr(settings, "gemini_embedding_retry_max_seconds", 60.0)

    await provider.embed_batch(["a"], batch_size=2)

    assert sleeps == [60.0]


@pytest.mark.asyncio
async def test_batch_bounded_retry_raises_quota_error_after_exhausting_attempts(monkeypatch):
    """Persistent 429s must not retry forever: raise a clear EmbeddingRateLimitError after
    the configured max attempts, bounding indexing so it cannot hang."""
    client = AlwaysFailingBatchClient(lambda: _rate_limit_error(retry_after=1))
    provider, sleeps = _provider_for_retry_tests(monkeypatch, client, max_retries=4)

    with pytest.raises(EmbeddingRateLimitError) as exc_info:
        await provider.embed_batch(["a", "b"], batch_size=2)

    assert client.call_count == 4  # exactly the bounded max attempts, not infinite
    assert len(sleeps) == 3  # backs off between attempts, not after the last one
    assert exc_info.value.retry_after == 1.0
    assert "quota" in str(exc_info.value).lower() or "rate" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_batch_non_rate_limit_error_is_retried_then_reraised_unchanged(monkeypatch):
    """A non-429 failure (e.g. a transient server error) still retries, but is re-raised as-is
    (not misclassified as a quota error) once attempts are exhausted."""
    client = AlwaysFailingBatchClient(lambda: RuntimeError("transient network blip"))
    provider, sleeps = _provider_for_retry_tests(monkeypatch, client, max_retries=2)

    with pytest.raises(RuntimeError, match="transient network blip"):
        await provider.embed_batch(["a"], batch_size=2)

    assert client.call_count == 2
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_validation_error_is_never_retried(monkeypatch):
    """A malformed response is a permanent failure, not transient -- must not be retried."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)

    def _raise_validation(texts):
        raise EmbeddingValidationError("malformed response")

    monkeypatch.setattr(provider, "_call_embed_batch_once", _raise_validation)
    sleeps: list = []
    monkeypatch.setattr(provider, "_sleep_fn", lambda seconds: sleeps.append(seconds))

    with pytest.raises(EmbeddingValidationError):
        await provider.embed_batch(["a"], batch_size=2)

    assert sleeps == []  # no retry/backoff for a validation error


# ---------------------------------------------------------------------------
# Rate limiting between batches
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _FakeAsyncSleeper:
    def __init__(self, clock: "_FakeClock"):
        self.clock = clock
        self.calls: list = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.now += seconds


@pytest.mark.asyncio
async def test_embed_batch_paces_requests_under_tpm_budget(monkeypatch):
    """embed_batch must throttle between batches so cumulative tokens stay under the
    configured tokens-per-minute budget, instead of firing every batch back-to-back."""
    clock = _FakeClock()
    sleeper = _FakeAsyncSleeper(clock)
    # A tight 100-token budget makes it trivial for two batches to exceed it.
    limiter = TokenPerMinuteRateLimiter(tpm_limit=100, clock=clock, sleep=sleeper)

    provider = GeminiEmbeddingProvider(api_key="mock_key", rate_limiter=limiter)
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(
        provider, "_call_embed_batch_sync", lambda texts: [[0.01] * 3072 for _ in texts]
    )

    # Each text is ~200 chars => ~50 estimated tokens; two single-text batches (batch_size=1)
    # of ~50 tokens each stay under 100 alone, but back-to-back sending of many should throttle.
    texts = ["x" * 200 for _ in range(4)]
    await provider.embed_batch(texts, batch_size=1)

    assert sleeper.calls, "expected the rate limiter to throttle between batches"


@pytest.mark.asyncio
async def test_embed_batch_no_throttling_needed_when_comfortably_under_budget(monkeypatch):
    """A generous TPM budget relative to the payload should never trigger a wait."""
    clock = _FakeClock()
    sleeper = _FakeAsyncSleeper(clock)
    limiter = TokenPerMinuteRateLimiter(tpm_limit=1_000_000, clock=clock, sleep=sleeper)

    provider = GeminiEmbeddingProvider(api_key="mock_key", rate_limiter=limiter)
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(
        provider, "_call_embed_batch_sync", lambda texts: [[0.01] * 3072 for _ in texts]
    )

    vectors = await provider.embed_batch(["short text"] * 3, batch_size=1)

    assert len(vectors) == 3
    assert sleeper.calls == []


@pytest.mark.asyncio
async def test_embed_batch_default_batch_size_comes_from_settings(monkeypatch):
    """Omitting batch_size must use settings.gemini_embedding_batch_size, not a hardcoded value."""
    provider = GeminiEmbeddingProvider(api_key="mock_key")
    monkeypatch.setattr(provider, "_is_mock_or_test_key", lambda: False)
    monkeypatch.setattr(settings, "gemini_embedding_batch_size", 2)

    captured_batches: list = []

    def fake_sync(texts):
        captured_batches.append(list(texts))
        return [[0.01] * 3072 for _ in texts]

    monkeypatch.setattr(provider, "_call_embed_batch_sync", fake_sync)

    await provider.embed_batch(["a", "b", "c", "d", "e"])

    assert [len(b) for b in captured_batches] == [2, 2, 1]

