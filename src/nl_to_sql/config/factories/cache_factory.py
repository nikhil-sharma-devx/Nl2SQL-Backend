"""Cache backend factory — selects the exact-match cache from settings."""
from nl_to_sql.config.settings import Settings
from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache
from nl_to_sql.infrastructure.cache.redis_cache import RedisCache


def build_cache(settings: Settings) -> object:
    """Factory: choose the cache backend from settings."""
    if settings.cache_provider == "redis":
        return RedisCache(
            redis_url=settings.redis_url,
            default_ttl=settings.cache_ttl_seconds,
        )
    return InMemoryCache(default_ttl=settings.cache_ttl_seconds)
