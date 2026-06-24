"""In-memory cache — implements ICache (dev/test/fallback)."""
import asyncio
import time
from typing import Any

import structlog

from nl_to_sql.core.interfaces.i_cache import ICache

logger = structlog.get_logger(__name__)

_SENTINEL = object()


class InMemoryCache(ICache):  # type: ignore[misc]
    """Thread-safe, TTL-aware in-memory cache.

    Designed for development and testing. No external dependencies.

    SOLID:
      L — Fully substitutable for RedisCache in any consumer.
    """

    def __init__(self, default_ttl: int = 3600) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}  # key → (value, expires_at)
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key, _SENTINEL)
            if entry is _SENTINEL:
                return None
            value: Any
            expires_at: float | None
            value, expires_at = entry  # type: ignore[misc]
            if expires_at is not None and time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + effective_ttl if effective_ttl > 0 else None
        async with self._lock:
            self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
