"""Client-side pacing for embedding API calls to respect a tokens-per-minute budget."""

import asyncio
import math
import time
from collections import deque
from typing import Awaitable, Callable, Deque, Optional, Tuple


def estimate_tokens(text: str) -> int:
    """Cheap local estimate of token count using the standard ~4 chars/token heuristic.

    Deliberately avoids calling the API's token counter, which would itself
    consume request-per-minute quota just to plan around the token-per-minute one.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


class TokenPerMinuteRateLimiter:
    """Sliding-window limiter that paces async callers to stay under a TPM budget.

    ``clock`` and ``sleep`` are injectable so tests can exercise real throttling
    decisions without waiting on a wall clock.
    """

    def __init__(
        self,
        tpm_limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ):
        if tpm_limit <= 0:
            raise ValueError("tpm_limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._tpm_limit = tpm_limit
        self._window = window_seconds
        self._clock = clock
        self._sleep = sleep or asyncio.sleep
        self._usage: Deque[Tuple[float, int]] = deque()

    def _evict_expired(self, now: float) -> None:
        while self._usage and now - self._usage[0][0] >= self._window:
            self._usage.popleft()

    def _used_tokens(self) -> int:
        return sum(tokens for _, tokens in self._usage)

    async def acquire(self, tokens: int) -> float:
        """Block until sending ``tokens`` more would stay within the TPM budget.

        A request that alone exceeds the budget is still let through once the
        window is empty, rather than deadlocking forever. Returns the total
        seconds actually waited (0.0 when no throttling was needed).
        """
        if tokens <= 0:
            return 0.0

        total_waited = 0.0
        while True:
            now = self._clock()
            self._evict_expired(now)
            used = self._used_tokens()
            if not self._usage or used + tokens <= self._tpm_limit:
                self._usage.append((now, tokens))
                return total_waited

            oldest_ts = self._usage[0][0]
            wait_time = max(self._window - (now - oldest_ts), 0.01)
            total_waited += wait_time
            await self._sleep(wait_time)
