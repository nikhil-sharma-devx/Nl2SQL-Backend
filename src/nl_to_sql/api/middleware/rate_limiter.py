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


# Medium: rate_limit_enabled existed on Settings but was never read anywhere,
# so there was no way to actually disable rate limiting (e.g. for load
# testing or an operational emergency) short of removing every @limiter.limit
# decorator. SlowAPI's own `enabled` flag makes every decorated route a no-op
# when False, without touching route code.
limiter = Limiter(key_func=_get_user_or_ip_key, enabled=get_settings().rate_limit_enabled)
