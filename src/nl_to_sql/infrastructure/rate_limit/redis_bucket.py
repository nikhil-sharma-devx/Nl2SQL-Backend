"""Redis token bucket — implements IRateLimiter.

The refill-and-take step runs as one Lua script, so it is atomic across every
worker and pod sharing the Redis instance. Time comes from Redis ``TIME`` so
app-server clock skew can't distort refill. If Redis is unreachable the store
degrades to a per-process in-memory bucket (still limited, never unlimited).
"""

import math
from typing import Any

import redis.asyncio as aioredis
import structlog

from nl_to_sql.core.interfaces.i_rate_limiter import BucketPolicy, IRateLimiter, RateDecision
from nl_to_sql.infrastructure.rate_limit.in_memory_bucket import InMemoryTokenBucket

logger = structlog.get_logger(__name__)

_PREFIX = "nl2sql:rl:"

# KEYS[1] bucket key; ARGV: capacity, refill_seconds, cost
# Returns {allowed(0/1), tokens_remaining(str), retry_after_seconds(str)}
_LUA_TOKEN_BUCKET = """
local cap = tonumber(ARGV[1])
local interval = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local data = redis.call('HMGET', KEYS[1], 't', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = cap
  ts = now
end
tokens = math.min(cap, tokens + (now - ts) / interval)
local need = math.max(cost, 1)
if tokens >= need then
  if cost > 0 then
    tokens = tokens - cost
    redis.call('HSET', KEYS[1], 't', tostring(tokens), 'ts', tostring(now))
    redis.call('PEXPIRE', KEYS[1], math.ceil(cap * interval * 1000) + 1000)
  end
  return {1, tostring(tokens), '0'}
end
return {0, tostring(tokens), tostring((need - tokens) * interval)}
"""


class RedisTokenBucket(IRateLimiter):
    """Distributed token bucket backed by a Redis hash per key."""

    def __init__(self, redis_url: str, socket_timeout: float = 2.0) -> None:
        self._redis: aioredis.Redis[Any] = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
        )
        self._script = self._redis.register_script(_LUA_TOKEN_BUCKET)
        self._fallback = InMemoryTokenBucket()

    async def consume(self, key: str, policy: BucketPolicy, cost: int = 1) -> RateDecision:
        try:
            allowed, remaining, retry = await self._script(
                keys=[_PREFIX + key],
                args=[policy.capacity, policy.refill_seconds, cost],
            )
        except Exception as exc:
            logger.warning("Redis rate limiter unavailable; using in-memory fallback", error=str(exc))
            return await self._fallback.consume(key, policy, cost)
        if int(allowed) == 1:
            return RateDecision(allowed=True, remaining=float(remaining), retry_after=0)
        return RateDecision(
            allowed=False,
            remaining=float(remaining),
            retry_after=max(math.ceil(float(retry)), 1),
        )

    async def reset(self, key: str) -> None:
        try:
            await self._redis.delete(_PREFIX + key)
        except Exception as exc:
            logger.warning("Redis rate limiter reset failed", error=str(exc))
        await self._fallback.reset(key)

    async def health_check(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False
