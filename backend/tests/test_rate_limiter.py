"""Unit tests for the embedding tokens-per-minute rate limiter."""

import pytest

from app.services.embeddings.rate_limiter import TokenPerMinuteRateLimiter, estimate_tokens


class FakeClock:
    """Deterministic, manually-advanced clock for testing sliding windows."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleeper:
    """Records sleep durations and advances a FakeClock instead of really waiting."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.calls: list = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * 8) == 2
    assert estimate_tokens("a" * 9) == 3


@pytest.mark.asyncio
async def test_acquire_does_not_throttle_when_under_budget():
    """Requests comfortably within the TPM budget incur no wait at all."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    limiter = TokenPerMinuteRateLimiter(tpm_limit=30_000, clock=clock, sleep=sleeper)

    waited_1 = await limiter.acquire(5_000)
    waited_2 = await limiter.acquire(5_000)

    assert waited_1 == 0.0
    assert waited_2 == 0.0
    assert sleeper.calls == []


@pytest.mark.asyncio
async def test_acquire_throttles_between_batches_over_budget():
    """A second batch that would exceed the TPM budget must wait for the window to clear."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    limiter = TokenPerMinuteRateLimiter(
        tpm_limit=30_000, window_seconds=60.0, clock=clock, sleep=sleeper
    )

    await limiter.acquire(20_000)
    # A second 20K-token batch right away would total 40K > 30K TPM: must throttle.
    await limiter.acquire(20_000)

    assert sleeper.calls, "expected the limiter to sleep before admitting the second batch"
    assert sum(sleeper.calls) >= 59.0  # waited out most of the 60s window


@pytest.mark.asyncio
async def test_acquire_admits_request_after_window_expires():
    """Once the sliding window has fully rolled over, usage resets and no wait is needed."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    limiter = TokenPerMinuteRateLimiter(
        tpm_limit=30_000, window_seconds=60.0, clock=clock, sleep=sleeper
    )

    await limiter.acquire(25_000)
    clock.advance(61.0)  # window has fully elapsed
    waited = await limiter.acquire(25_000)

    assert waited == 0.0
    assert sleeper.calls == []


@pytest.mark.asyncio
async def test_acquire_allows_single_oversized_request_without_deadlock():
    """A single request larger than the whole budget must still be admitted (not hang forever)."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    limiter = TokenPerMinuteRateLimiter(tpm_limit=1_000, clock=clock, sleep=sleeper)

    waited = await limiter.acquire(5_000)

    assert waited == 0.0
    assert sleeper.calls == []


@pytest.mark.asyncio
async def test_acquire_sequential_calls_never_exceed_budget_in_window():
    """Repeated acquisitions never let cumulative in-window usage exceed the TPM limit."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    limiter = TokenPerMinuteRateLimiter(
        tpm_limit=10_000, window_seconds=60.0, clock=clock, sleep=sleeper
    )

    for _ in range(5):
        await limiter.acquire(4_000)
        now = clock.now
        limiter._evict_expired(now)
        assert limiter._used_tokens() <= 10_000
