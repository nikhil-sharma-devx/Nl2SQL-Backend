"""Deactivated accounts: blocked sign-in, owner alert, reactivation request, admin decision.

Real SQLite DB and real services; Google's token check, the brute-force
throttle and SMTP are stubbed (emails are captured in a list).

Regression covered: a deactivated Google account was issued tokens, landed in
the app once and was then logged out by the next request. Now neither sign-in
path issues tokens to a deactivated account.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import (
    AccountReactivationRequest,
    Base,
    LoginEvent,
    RefreshToken,
    User,
)
from nl_to_sql.services.account_reactivation_service import AccountReactivationService
from nl_to_sql.services.admin_service import AdminService
from nl_to_sql.services.auth_service import (
    create_reactivation_token,
    decode_access_token,
    hash_password,
    to_user_public,
)

PASSWORD = "correct-horse-battery"


@dataclass
class World:
    factory: async_sessionmaker[AsyncSession]
    svc: AccountReactivationService
    sent: list[tuple[str, str, str]] = field(default_factory=list)
    acting_as: str = "admin"


@pytest.fixture
async def world() -> AsyncIterator[World]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        db.add_all(
            [
                User(id="owner", email="owner@example.com", is_verified=True, is_admin=True, is_owner=True),
                User(id="admin", email="admin@example.com", is_verified=True, is_admin=True),
                User(
                    id="off",
                    email="off@example.com",
                    hashed_password=hash_password(PASSWORD),
                    google_sub="g-off",
                    is_verified=True,
                    is_active=False,
                ),
            ]
        )
        await db.commit()

    sent: list[tuple[str, str, str]] = []

    async def record(to: str, subject: str, body: str) -> None:
        sent.append((to, subject, body))

    w = World(factory, AccountReactivationService(factory, record, "https://app.test/admin"), sent)
    original = limiter.enabled
    limiter.enabled = False
    try:
        yield w
    finally:
        limiter.enabled = original
        await engine.dispose()
        try:
            os.remove(path)
        except OSError:
            pass


def _app(w: World) -> Any:
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_account_reactivation_service,
            get_admin_service,
            get_auth_throttle,
            get_current_user,
            get_session_service,
        )

        app = create_app()
    app.dependency_overrides[get_session_service] = lambda: SimpleNamespace(_session_factory=w.factory)
    app.dependency_overrides[get_account_reactivation_service] = lambda: w.svc
    app.dependency_overrides[get_admin_service] = lambda: AdminService(w.factory)
    permissive = MagicMock()
    for name in ("check_login_allowed", "record_login_failure", "record_login_success"):
        setattr(permissive, name, AsyncMock())
    app.dependency_overrides[get_auth_throttle] = lambda: permissive

    async def current_user() -> UserPublic:
        async with w.factory() as db:
            return to_user_public(await db.get(User, w.acting_as))

    app.dependency_overrides[get_current_user] = current_user
    return app


async def _call(w: World, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    async with AsyncClient(transport=ASGITransport(app=_app(w)), base_url="http://test") as client:
        resp = await client.request(method, path, json=body)
    await w.svc.drain()
    return resp


async def _login(w: World) -> Any:
    return await _call(w, "POST", "/api/v1/auth/login", {"email": "off@example.com", "password": PASSWORD})


# ── Sign-in is blocked ───────────────────────────────────────────────────────


async def test_password_sign_in_is_blocked_and_owner_is_emailed(world: World) -> None:
    resp = await _login(world)
    body = resp.json()
    assert resp.status_code == 403 and body["code"] == "ACCOUNT_DEACTIVATED"
    assert "access_token" not in body
    token = body["details"]["detail"]
    with pytest.raises(JWTError):
        decode_access_token(token)  # the reactivation token can't authenticate requests
    assert [(to, subj) for to, subj, _ in world.sent] == [
        ("owner@example.com", "Deactivated account tried to sign in: off@example.com")
    ]
    async with world.factory() as db:
        assert (await db.execute(select(RefreshToken))).first() is None
        outcomes = (await db.execute(select(LoginEvent.outcome))).scalars().all()
        assert outcomes == ["failed"]


async def test_wrong_password_reveals_nothing(world: World) -> None:
    resp = await _call(world, "POST", "/api/v1/auth/login", {"email": "off@example.com", "password": "wrong-password"})
    assert resp.status_code == 401
    assert world.sent == []


async def test_google_sign_in_is_blocked_too(world: World) -> None:
    claims = {"sub": "g-off", "email": "off@example.com", "email_verified": True}
    with patch("nl_to_sql.api.routes.auth.verify_google_token", AsyncMock(return_value=claims)):
        resp = await _call(world, "POST", "/api/v1/auth/google", {"credential": "stub"})
    assert resp.status_code == 403 and resp.json()["code"] == "ACCOUNT_DEACTIVATED"
    async with world.factory() as db:
        assert (await db.execute(select(RefreshToken))).first() is None


async def test_owner_alert_is_throttled_per_account(world: World) -> None:
    await _login(world)
    await _login(world)
    assert len(world.sent) == 1


# ── Request → decision ───────────────────────────────────────────────────────


async def test_request_then_admin_approves(world: World) -> None:
    token = (await _login(world)).json()["details"]["detail"]
    world.sent.clear()

    resp = await _call(
        world, "POST", "/api/v1/auth/reactivation-request", {"token": token, "message": "I'm back from leave"}
    )
    assert resp.status_code == 201
    assert sorted(to for to, _, _ in world.sent) == ["admin@example.com", "owner@example.com"]
    assert "I'm back from leave" in world.sent[0][2]

    # Signing in again now says the request is pending (no new token).
    again = await _login(world)
    assert again.status_code == 403 and again.json()["code"] == "REACTIVATION_PENDING"
    dup = await _call(world, "POST", "/api/v1/auth/reactivation-request", {"token": token})
    assert dup.status_code == 409

    listed = (await _call(world, "GET", "/api/v1/admin/reactivation-requests")).json()
    assert listed["pending"] == 1 and listed["items"][0]["email"] == "off@example.com"
    req_id = listed["items"][0]["id"]

    world.sent.clear()
    decided = await _call(world, "POST", f"/api/v1/admin/reactivation-requests/{req_id}/approve")
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved" and decided.json()["decided_by_email"] == "admin@example.com"
    assert [to for to, _, _ in world.sent] == ["off@example.com"]

    ok = await _login(world)
    assert ok.status_code == 200 and ok.json()["access_token"]
    again = await _call(world, "POST", f"/api/v1/admin/reactivation-requests/{req_id}/reject")
    assert again.status_code == 409


async def test_reject_keeps_account_off_and_allows_asking_again(world: World) -> None:
    token = (await _login(world)).json()["details"]["detail"]
    await _call(world, "POST", "/api/v1/auth/reactivation-request", {"token": token})
    req_id = (await _call(world, "GET", "/api/v1/admin/reactivation-requests")).json()["items"][0]["id"]

    rejected = await _call(world, "POST", f"/api/v1/admin/reactivation-requests/{req_id}/reject")
    assert rejected.json()["status"] == "rejected"
    async with world.factory() as db:
        assert (await db.get(User, "off")).is_active is False

    fresh = await _login(world)
    assert fresh.json()["code"] == "ACCOUNT_DEACTIVATED"  # may ask again
    history = (await _call(world, "GET", "/api/v1/admin/reactivation-requests?status=all")).json()
    assert [i["status"] for i in history["items"]] == ["rejected"]


async def test_direct_reactivation_closes_pending_request(world: World) -> None:
    token = (await _login(world)).json()["details"]["detail"]
    await _call(world, "POST", "/api/v1/auth/reactivation-request", {"token": token})
    resp = await _call(world, "PATCH", "/api/v1/admin/users/off", {"is_active": True})
    assert resp.status_code == 200
    async with world.factory() as db:
        statuses = (await db.execute(select(AccountReactivationRequest.status))).scalars().all()
    assert statuses == ["approved"]


# ── Guards ───────────────────────────────────────────────────────────────────


async def test_request_needs_a_valid_reactivation_token(world: World) -> None:
    bad = await _call(world, "POST", "/api/v1/auth/reactivation-request", {"token": "not-a-jwt"})
    assert bad.status_code == 401
    active = create_reactivation_token("admin")  # an active account can't file one
    assert (await _call(world, "POST", "/api/v1/auth/reactivation-request", {"token": active})).status_code == 409


async def test_non_admin_cannot_see_or_decide_requests(world: World) -> None:
    async with world.factory() as db:
        db.add(User(id="plain", email="plain@example.com", is_verified=True))
        await db.commit()
    world.acting_as = "plain"
    assert (await _call(world, "GET", "/api/v1/admin/reactivation-requests")).status_code == 403
    assert (await _call(world, "POST", "/api/v1/admin/reactivation-requests/x/approve")).status_code == 403
