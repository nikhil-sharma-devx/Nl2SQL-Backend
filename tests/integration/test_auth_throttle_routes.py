"""Integration tests: auth routes enforce the per-account / per-IP token buckets.

SlowAPI's per-IP decorators are disabled here so the 429s observed come from
AuthThrottleService alone (each test gets a fresh in-memory bucket).
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.api.dependencies import get_auth_throttle, get_session_service
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.infrastructure.rate_limit.in_memory_bucket import InMemoryTokenBucket
from nl_to_sql.services.auth_throttle_service import (
    LOGIN_FAIL_ACCOUNT,
    OTP_VERIFY_ACCOUNT,
    AuthThrottleService,
)


@pytest.fixture
def db_user() -> MagicMock:
    """The User row returned by every SELECT (None ⇒ unknown account)."""
    user = MagicMock()
    user.id = "user-123"
    user.email = "victim@example.com"
    user.hashed_password = "$2b$12$fakehash"
    user.auth_provider = "email"
    user.is_active = True
    user.is_verified = False
    user.otp_code = "123456"
    user.otp_expires_at = None
    user.otp_failed_attempts = 0
    return user


@pytest.fixture
async def client(app: Any, db_user: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = db_user
    db.execute = AsyncMock(return_value=result)

    session_service = MagicMock()
    session_service._session_factory = MagicMock(return_value=db)

    throttle = AuthThrottleService(rate_limiter=InMemoryTokenBucket(), enabled=True)
    app.dependency_overrides[get_session_service] = lambda: session_service
    app.dependency_overrides[get_auth_throttle] = lambda: throttle

    original = limiter.enabled
    limiter.enabled = False
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        limiter.enabled = original
        app.dependency_overrides.clear()


class TestLoginThrottle:
    async def test_locks_account_after_repeated_bad_passwords(self, client) -> None:
        body = {"email": "victim@example.com", "password": "wrong-password"}
        with patch("nl_to_sql.api.routes.auth.verify_password", return_value=False):
            for _ in range(LOGIN_FAIL_ACCOUNT.capacity):
                r = await client.post("/api/v1/auth/login", json=body)
                assert r.status_code == 401
            r = await client.post("/api/v1/auth/login", json=body)

        assert r.status_code == 429
        data = r.json()
        assert data["code"] == "AUTH_RATE_LIMITED"
        assert data["retry_after"] > 0
        assert r.headers["Retry-After"] == str(data["retry_after"])

    @pytest.mark.parametrize("db_user", [None])
    async def test_unknown_account_throttled_identically(self, client) -> None:
        """No enumeration: a non-existent email locks exactly like a real one."""
        body = {"email": "ghost@example.com", "password": "wrong-password"}
        for _ in range(LOGIN_FAIL_ACCOUNT.capacity):
            assert (await client.post("/api/v1/auth/login", json=body)).status_code == 401
        r = await client.post("/api/v1/auth/login", json=body)
        assert r.status_code == 429


class TestForgotPasswordThrottle:
    async def test_second_request_is_silently_suppressed(self, client) -> None:
        body = {"email": "victim@example.com"}
        with patch(
            "nl_to_sql.services.auth_service.send_otp_email", new=AsyncMock()
        ) as send:
            r1 = await client.post("/api/v1/auth/forgot-password", json=body)
            r2 = await client.post("/api/v1/auth/forgot-password", json=body)

        # Identical generic reply both times; only the first actually mails.
        assert r1.status_code == r2.status_code == 202
        assert r1.json() == r2.json()
        assert send.await_count == 1


class TestResendOtpThrottle:
    async def test_cooldown_returns_429(self, client) -> None:
        body = {"email": "victim@example.com"}
        with patch("nl_to_sql.services.auth_service.send_otp_email", new=AsyncMock()):
            assert (await client.post("/api/v1/auth/resend-otp", json=body)).status_code == 202
            r = await client.post("/api/v1/auth/resend-otp", json=body)
        assert r.status_code == 429
        assert r.json()["code"] == "AUTH_RATE_LIMITED"


class TestSlowApiEnvelope:
    async def test_per_ip_limit_uses_canonical_envelope(self, client) -> None:
        limiter.enabled = True
        limiter.reset()
        try:
            with patch("nl_to_sql.services.auth_service.send_otp_email", new=AsyncMock()):
                codes = [
                    (
                        await client.post(
                            "/api/v1/auth/forgot-password", json={"email": f"u{i}@example.com"}
                        )
                    )
                    for i in range(4)
                ]
        finally:
            limiter.reset()
        r = codes[-1]
        assert [c.status_code for c in codes[:3]] == [202, 202, 202]
        assert r.status_code == 429
        assert r.json()["code"] == "RATE_LIMIT_EXCEEDED"
        assert int(r.headers["Retry-After"]) > 0


class TestVerifyOtpThrottle:
    async def test_guess_budget_exhausts(self, client) -> None:
        body = {"email": "victim@example.com", "otp_code": "000000"}
        for _ in range(OTP_VERIFY_ACCOUNT.capacity):
            r = await client.post("/api/v1/auth/verify-otp", json=body)
            assert r.status_code in (400, 429)  # DB per-OTP counter may also trip
        r = await client.post("/api/v1/auth/verify-otp", json=body)
        assert r.status_code == 429
        assert r.json()["code"] == "AUTH_RATE_LIMITED"
