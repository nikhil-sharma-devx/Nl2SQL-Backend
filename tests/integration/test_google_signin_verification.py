"""Google sign-in accounts: verification, one-account-per-email, adding a password.

Real SQLite DB; only Google's token check, the OTP mailer and the brute-force
throttle are stubbed.

Regressions covered:
- Google users were created with ``is_verified=False`` (hidden from the team
  directory, impossible to invite) although Google already verified the email.
- Google sign-ups couldn't add a password: forgot/reset-password only served
  ``auth_provider == "email"`` accounts.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import Base, User


@pytest.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()
        try:
            os.remove(path)
        except OSError:
            pass


def _app(factory: async_sessionmaker[AsyncSession]) -> Any:
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_auth_throttle, get_session_service

        app = create_app()
    app.dependency_overrides[get_session_service] = lambda: SimpleNamespace(_session_factory=factory)
    permissive = MagicMock()
    for name in (
        "enforce_otp_send", "check_login_allowed", "record_login_failure", "record_login_success",
        "check_otp_send_ip", "enforce_otp_verify",
    ):
        setattr(permissive, name, AsyncMock())
    permissive.try_otp_send_account = AsyncMock(return_value=SimpleNamespace(allowed=True))
    app.dependency_overrides[get_auth_throttle] = lambda: permissive
    return app


async def _post(factory: async_sessionmaker[AsyncSession], path: str, body: dict[str, Any]) -> Any:
    async with AsyncClient(transport=ASGITransport(app=_app(factory)), base_url="http://test") as client:
        return await client.post(path, json=body)


async def _google_sign_in(factory: async_sessionmaker[AsyncSession], claims: dict[str, Any]) -> int:
    with patch("nl_to_sql.api.routes.auth.verify_google_token", AsyncMock(return_value=claims)):
        resp = await _post(factory, "/api/v1/auth/google", {"credential": "stub-token"})
    return resp.status_code


async def _user(factory: async_sessionmaker[AsyncSession], email: str) -> User:
    async with factory() as db:
        return (await db.execute(select(User).where(User.email == email))).scalar_one()


async def test_new_google_user_is_verified(factory: async_sessionmaker[AsyncSession]) -> None:
    claims = {"sub": "g-1", "email": "new@gmail.com", "email_verified": True, "name": "New Person"}
    assert await _google_sign_in(factory, claims) == 200
    user = await _user(factory, "new@gmail.com")
    assert (user.auth_provider, user.is_verified) == ("google", True)


async def test_unverified_email_signup_becomes_verified_via_google(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    async with factory() as db:
        db.add(User(id="u1", email="half@gmail.com", hashed_password="x", is_verified=False))
        await db.commit()
    claims = {"sub": "g-2", "email": "half@gmail.com", "email_verified": True, "name": "Half"}
    assert await _google_sign_in(factory, claims) == 200
    user = await _user(factory, "half@gmail.com")
    assert (user.google_sub, user.is_verified) == ("g-2", True)


async def test_unverified_google_email_is_still_rejected(factory: async_sessionmaker[AsyncSession]) -> None:
    claims = {"sub": "g-3", "email": "nope@gmail.com", "email_verified": False}
    assert await _google_sign_in(factory, claims) == 401
    async with factory() as db:
        assert (await db.execute(select(User).where(User.email == "nope@gmail.com"))).first() is None


# ── One account per email; Google users can add a password ─────────────────


async def test_registering_a_google_users_email_is_refused_not_duplicated(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    claims = {"sub": "g-4", "email": "dup@gmail.com", "email_verified": True, "name": "Dup"}
    assert await _google_sign_in(factory, claims) == 200
    with patch("nl_to_sql.services.auth_service.send_otp_email", AsyncMock()):
        resp = await _post(
            factory, "/api/v1/auth/register", {"email": "DUP@gmail.com", "password": "a-long-password-1"}
        )
    assert resp.status_code == 409
    assert "Forgot password" in resp.json()["message"]
    async with factory() as db:
        count = (await db.execute(select(func.count()).select_from(User).where(User.email == "dup@gmail.com"))).scalar()
    assert count == 1


async def test_google_user_sets_a_password_via_forgot_password_then_signs_in(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    claims = {"sub": "g-5", "email": "pw@gmail.com", "email_verified": True, "name": "Pw"}
    assert await _google_sign_in(factory, claims) == 200
    google_user = await _user(factory, "pw@gmail.com")
    assert google_user.hashed_password is None

    # No password yet → password sign-in fails generically.
    assert (await _post(factory, "/api/v1/auth/login", {"email": "pw@gmail.com", "password": "new-password-123"})).status_code == 401

    sent = AsyncMock()
    with patch("nl_to_sql.services.auth_service.send_otp_email", sent):
        assert (await _post(factory, "/api/v1/auth/forgot-password", {"email": "pw@gmail.com"})).status_code == 202
    sent.assert_awaited_once()
    otp = (await _user(factory, "pw@gmail.com")).otp_code
    assert otp

    reset = await _post(
        factory,
        "/api/v1/auth/reset-password",
        {"email": "pw@gmail.com", "otp_code": otp, "new_password": "new-password-123"},
    )
    assert reset.status_code == 200, reset.text

    login = await _post(factory, "/api/v1/auth/login", {"email": "pw@gmail.com", "password": "new-password-123"})
    assert login.status_code == 200, login.text
    assert login.json()["user"]["id"] == google_user.id  # same account, not a new one
    # Google sign-in still works for the same account.
    assert await _google_sign_in(factory, claims) == 200


async def test_deactivated_accounts_get_no_reset_code(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory() as db:
        db.add(User(id="off", email="off@gmail.com", google_sub="g-6", is_verified=True, is_active=False))
        await db.commit()
    sent = AsyncMock()
    with patch("nl_to_sql.services.auth_service.send_otp_email", sent):
        resp = await _post(factory, "/api/v1/auth/forgot-password", {"email": "off@gmail.com"})
    assert resp.status_code == 202  # same generic reply — no enumeration
    sent.assert_not_awaited()
