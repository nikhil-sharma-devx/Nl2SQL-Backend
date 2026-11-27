"""Cache backend factory — selects the exact-match cache from settings."""
from nl_to_sql.config.settings import Settings
from nl_to_sql.core.interfaces.i_cache import ICache
from nl_to_sql.core.interfaces.i_rate_limiter import IRateLimiter
from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache
from nl_to_sql.infrastructure.cache.redis_cache import RedisCache
from nl_to_sql.infrastructure.rate_limit.in_memory_bucket import InMemoryTokenBucket
from nl_to_sql.infrastructure.rate_limit.redis_bucket import RedisTokenBucket


def build_cache(settings: Settings) -> ICache:
    """Factory: choose the cache backend from settings."""
    if settings.cache_provider == "redis":
        return RedisCache(
            redis_url=settings.redis_url,
            default_ttl=settings.cache_ttl_seconds,
        )
    return InMemoryCache(default_ttl=settings.cache_ttl_seconds)


def build_rate_limiter(settings: Settings) -> IRateLimiter:
    """Factory: token-bucket store - Redis (shared across workers) or in-process."""
    if settings.cache_provider == "redis":
        return RedisTokenBucket(redis_url=settings.redis_url)
    return InMemoryTokenBucket()
