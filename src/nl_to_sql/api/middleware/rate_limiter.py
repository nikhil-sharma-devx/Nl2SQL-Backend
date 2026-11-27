"""Rate limiter middleware using SlowAPI.

Per-user rate limiting when a valid JWT is present, falls back to IP address
for unauthenticated requests. This makes limiting fairer and harder to bypass
via IP rotation.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from nl_to_sql.config.settings import get_settings


def _get_user_or_ip_key(request: Request) -> str:
    """Rate-limit key: authenticated user ID if token is valid, else client IP."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            from nl_to_sql.services.auth_service import decode_access_token

            token_data = decode_access_token(auth[7:].strip())
            return f"user:{token_data.user_id}"
        except Exception:
            pass
    return get_remote_address(request)


# SlowAPI reads `Limiter.enabled` fresh on every request (a plain instance
# attribute), so it can be flipped after startup too — see
# PUT /api/v1/config/rate-limit in api/routes/config.py, which mutates
# `limiter.enabled` directly instead of leaving this initial value as the
# only place it's ever set.
def _build_limiter() -> Limiter:
    settings = get_settings()
    use_redis = settings.cache_provider == "redis"
    return Limiter(
        key_func=_get_user_or_ip_key,
        enabled=settings.rate_limit_enabled,
        # Sliding window: a fixed window lets a client fire 2x the limit across
        # a window boundary (e.g. 5 at 0:59 + 5 at 1:00) — bad for auth routes.
        strategy="moving-window",
        # Redis makes counters shared across workers/pods; in-memory otherwise.
        storage_uri=settings.redis_url if use_redis else "memory://",
        # If Redis is down, keep limiting in-process instead of failing requests.
        in_memory_fallback_enabled=use_redis,
        swallow_errors=use_redis,
        headers_enabled=False,  # Retry-After is set by the envelope handler
    )


limiter = _build_limiter()
