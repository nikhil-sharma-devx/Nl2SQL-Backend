"""Daily AI-query limits (UsageQuotaService) on a real SQLite DB."""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.core.exceptions import DailyQuotaExceededError
from nl_to_sql.infrastructure.database.models import Base, User
from nl_to_sql.services.usage_quota_service import UsageQuotaService


@pytest.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    f = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with f() as db:
        db.add_all(
            [
                User(id="u1", email="a@x.com"),
                User(id="u2", email="b@x.com", daily_query_limit=5),
                User(id="boss", email="boss@x.com", is_admin=True, daily_query_limit=1),
            ]
        )
        await db.commit()
    try:
        yield f
    finally:
        await engine.dispose()
        try:
            os.remove(path)
        except OSError:
            pass


async def test_unlimited_by_default_but_still_counted(factory) -> None:
    svc = UsageQuotaService(factory, default_limit=0)
    for _ in range(3):
        await svc.consume("u1", is_admin=False)
    assert await svc.used_today("u1") == 3
    assert await svc.limit_for("u1", is_admin=False) is None


async def test_platform_default_applies_and_stops_exactly_at_the_limit(factory) -> None:
    svc = UsageQuotaService(factory, default_limit=2)
    await svc.consume("u1", is_admin=False)
    await svc.consume("u1", is_admin=False)
    with pytest.raises(DailyQuotaExceededError) as exc:
        await svc.consume("u1", is_admin=False)
    assert exc.value.retry_after and 0 < exc.value.retry_after <= 86400
    assert await svc.used_today("u1") == 2  # the refused call isn't counted


async def test_own_limit_overrides_default_and_admins_are_exempt(factory) -> None:
    svc = UsageQuotaService(factory, default_limit=1)
    assert await svc.limit_for("u2", is_admin=False) == 5
    assert await svc.effective_limit("boss") is None  # admin: own limit ignored
    for _ in range(3):
        await svc.consume("boss", is_admin=True)


async def test_concurrent_requests_cannot_exceed_the_limit(factory) -> None:
    svc = UsageQuotaService(factory, default_limit=0)
    await svc.set_limit("u1", 3)
    results = await asyncio.gather(
        *(svc.consume("u1", is_admin=False) for _ in range(20)), return_exceptions=True
    )
    assert sum(1 for r in results if r is None) == 3
    assert await svc.used_today("u1") == 3


async def test_set_limit_returns_previous_and_validates(factory) -> None:
    svc = UsageQuotaService(factory)
    assert await svc.set_limit("u2", 10) == ("b@x.com", 5)
    assert await svc.set_limit("u2", None) == ("b@x.com", 10)
    with pytest.raises(ValueError):
        await svc.set_limit("u2", -1)


async def test_limit_reached_is_a_429_with_retry_after(factory) -> None:
    """The envelope the query routes return when the dependency refuses."""
    from nl_to_sql.api.app import create_app
    from nl_to_sql.api.dependencies import (
        consume_daily_query,
        get_current_user,
        get_usage_quota_service,
    )
    from nl_to_sql.core.models.auth import UserPublic

    svc = UsageQuotaService(factory, default_limit=1)
    app: FastAPI = create_app()

    @app.get("/_quota_probe")
    async def probe(_q: None = Depends(consume_daily_query)) -> dict[str, bool]:
        return {"ok": True}

    async def user() -> UserPublic:
        from datetime import datetime

        return UserPublic(
            id="u1", email="a@x.com", full_name=None, auth_provider="email",
            is_verified=True, created_at=datetime(2024, 1, 1),
        )

    app.dependency_overrides[get_current_user] = user
    app.dependency_overrides[get_usage_quota_service] = lambda: svc
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/_quota_probe")).status_code == 200
        r = await c.get("/_quota_probe")
    assert r.status_code == 429
    assert r.json()["code"] == "DAILY_LIMIT_REACHED"
    assert int(r.headers["Retry-After"]) > 0
