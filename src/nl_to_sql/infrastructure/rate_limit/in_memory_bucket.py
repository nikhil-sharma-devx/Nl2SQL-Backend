"""In-process token bucket — implements IRateLimiter.

Suitable for dev / single-worker deployments. With several workers or pods each
process keeps its own buckets, so production should use the Redis store.
"""

import asyncio
import math
import time
from collections.abc import Callable

from nl_to_sql.core.interfaces.i_rate_limiter import BucketPolicy, IRateLimiter, RateDecision

# Prune fully-refilled buckets once the map grows past this many keys so an
# attacker spraying random emails can't grow memory without bound.
_PRUNE_THRESHOLD = 10_000


class InMemoryTokenBucket(IRateLimiter):
    """Dict-backed token buckets guarded by a single asyncio lock."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        # key -> (tokens, last_refill_ts, capacity, refill_seconds)
        self._buckets: dict[str, tuple[float, float, int, float]] = {}
        self._lock = asyncio.Lock()
        self._clock: Callable[[], float] = clock or time.monotonic

    def _refilled(self, key: str, policy: BucketPolicy, now: float) -> float:
        state = self._buckets.get(key)
        if state is None:
            return float(policy.capacity)
        tokens, ts, _, _ = state
        return min(float(policy.capacity), tokens + (now - ts) / policy.refill_seconds)

    def _prune(self, now: float) -> None:
        full = [
            k
            for k, (tokens, ts, cap, refill) in self._buckets.items()
            if tokens + (now - ts) / refill >= cap
        ]
        for k in full:
            del self._buckets[k]

    async def consume(self, key: str, policy: BucketPolicy, cost: int = 1) -> RateDecision:
        async with self._lock:
            now = self._clock()
            tokens = self._refilled(key, policy, now)
            need = max(cost, 1)
            if tokens >= need:
                if cost > 0:
                    tokens -= cost
                    self._buckets[key] = (tokens, now, policy.capacity, policy.refill_seconds)
                    if len(self._buckets) > _PRUNE_THRESHOLD:
                        self._prune(now)
                return RateDecision(allowed=True, remaining=tokens, retry_after=0)
            retry = math.ceil((need - tokens) * policy.refill_seconds)
            return RateDecision(allowed=False, remaining=tokens, retry_after=max(retry, 1))

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._buckets.pop(key, None)

    async def health_check(self) -> bool:
        return True
