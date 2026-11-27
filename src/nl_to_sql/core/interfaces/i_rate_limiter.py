"""IRateLimiter — abstract token-bucket rate limiter.

Token bucket semantics: each key owns a bucket holding at most ``capacity``
tokens; one token is regenerated every ``refill_seconds``. A request costs
``cost`` tokens and is allowed only when the bucket holds enough. This allows a
small burst (``capacity``) and then throttles to a steady rate, with O(1)
storage per key — the right shape for login / OTP brute-force protection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketPolicy:
    """Bucket shape: burst size and seconds needed to regenerate one token."""

    name: str
    capacity: int
    refill_seconds: float


@dataclass(frozen=True)
class RateDecision:
    """Outcome of a bucket check.

    ``retry_after`` is whole seconds until enough tokens exist (0 if allowed).
    """

    allowed: bool
    remaining: float
    retry_after: int


class IRateLimiter(ABC):
    """Contract for any token-bucket store (in-memory, Redis, ...).

    SOLID: Liskov — any store is substitutable; Dependency Inversion —
           AuthThrottleService depends on this, never on Redis directly.
    """

    @abstractmethod
    async def consume(self, key: str, policy: BucketPolicy, cost: int = 1) -> RateDecision:
        """Atomically take ``cost`` tokens if available.

        ``cost=0`` is a non-mutating peek: allowed iff at least one token remains.
        """

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Refill the bucket for ``key`` to full (e.g. after a successful login)."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backing store is reachable."""
