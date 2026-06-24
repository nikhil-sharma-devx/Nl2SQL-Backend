"""Redis cache — implements ICache."""
import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from nl_to_sql.core.exceptions import CacheError
from nl_to_sql.core.interfaces.i_cache import ICache

logger = structlog.get_logger(__name__)


_DEFAULT_PREFIX = "nl2sql:"


class RedisCache(ICache):  # type: ignore[misc]
    """Async Redis cache with JSON serialisation.

    All keys are transparently prefixed with ``_prefix`` (default ``nl2sql:``)
    so that ``clear()`` can safely delete only this application's keys without
    touching other tenants on a shared Redis instance.

    SOLID:
      S — Only caches/retrieves; orchestration lives in services.
      D — Callers depend on ICache, not on this class.
    """

    def __init__(self, redis_url: str, default_ttl: int = 3600, prefix: str = _DEFAULT_PREFIX) -> None:
        self._redis: aioredis.Redis[Any] = aioredis.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )
        self._default_ttl = default_ttl
        self._prefix = prefix

    def _k(self, key: str) -> str:
        """Return the prefixed Redis key."""
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._redis.get(self._k(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis get failed", key=key, error=str(exc))
            return None  # graceful degradation — cache miss on error

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        try:
            await self._redis.set(self._k(key), json.dumps(value), ex=effective_ttl)
        except Exception as exc:
            logger.warning("Redis set failed", key=key, error=str(exc))
            # Do not raise — cache failures are non-fatal

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(self._k(key))
        except Exception as exc:
            raise CacheError(f"Redis delete failed for key={key}: {exc}") from exc

    async def clear(self) -> None:
        """Delete all keys belonging to this application using SCAN + DEL.

        Never calls FLUSHDB — that would wipe keys belonging to other services
        or tenants sharing the same Redis instance.
        """
        try:
            cursor = 0
            pattern = f"{self._prefix}*"
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as exc:
            raise CacheError(f"Redis clear failed: {exc}") from exc
