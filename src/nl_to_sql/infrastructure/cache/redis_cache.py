"""Redis cache — implements ICache."""
import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from nl_to_sql.core.exceptions import CacheError
from nl_to_sql.core.interfaces.i_cache import ICache

logger = structlog.get_logger(__name__)


class RedisCache(ICache):
    """Async Redis cache with JSON serialisation.

    SOLID:
      S — Only caches/retrieves; orchestration lives in services.
      D — Callers depend on ICache, not on this class.  Remove-Item -Recurse -Force .git
    """

    def __init__(self, redis_url: str, default_ttl: int = 3600) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis get failed", key=key, error=str(exc))
            return None  # graceful degradation — cache miss on error

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        try:
            await self._redis.set(key, json.dumps(value), ex=effective_ttl)
        except Exception as exc:
            logger.warning("Redis set failed", key=key, error=str(exc))
            # Do not raise — cache failures are non-fatal

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(key)
        except Exception as exc:
            raise CacheError(f"Redis delete failed for key={key}: {exc}") from exc

    async def clear(self) -> None:
        try:
            await self._redis.flushdb()
        except Exception as exc:
            raise CacheError(f"Redis flush failed: {exc}") from exc
