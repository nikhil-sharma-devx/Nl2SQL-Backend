"""Integration tests for admin-account routes in ``api/routes/admin.py``.

  GET    /api/v1/admin/setup            any signed-in user
  POST   /api/v1/admin/setup            any signed-in user + ADMIN_SETUP_TOKEN, once
  GET    /api/v1/admin/admins           admin
  POST   /api/v1/admin/admins           admin
  DELETE /api/v1/admin/admins/{id}      owner only

``get_current_user`` is overridden to whichever account the test acts as,
built from the live DB row (as the real dependency does), so the REAL
``require_admin`` / ``require_owner`` gates are exercised. AdminService runs
against a temporary SQLite database.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.api.app import create_app
from nl_to_sql.api.dependencies import (
    get_admin_service,
    get_audit_log_service,
    get_current_user,
    get_usage_quota_service,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import Base, User
from nl_to_sql.services.admin_service import AdminService
from nl_to_sql.services.audit_log_service import AuditLogService
from nl_to_sql.services.auth_service import to_user_public
from nl_to_sql.services.usage_quota_service import UsageQuotaService

TOKEN = "route-setup-token"


class World:
    def __init__(self, client: TestClient, factory: async_sessionmaker[AsyncSession]) -> None:
        self.client = client
        self.factory = factory
        self.acting_as = "owner"


@pytest.fixture
def world() -> Iterator[World]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def seed() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with factory() as db:
            db.add_all(
                [
                    User(id="owner", email="owner@example.com", is_verified=True),
                    User(id="u1", email="ana@example.com", is_verified=True),
                    User(id="u2", email="ben@example.com", is_verified=True),
                ]
            )
            await db.commit()

    asyncio.run(seed())
    svc = AdminService(factory, SecretStr(TOKEN))
    app = create_app()
    w = World(TestClient(app), factory)

    async def fake_current_user() -> UserPublic:
        async with factory() as db:
            row = (await db.execute(select(User).where(User.id == w.acting_as))).scalar_one()
            return to_user_public(row)

    app.dependency_overrides[get_current_user] = fake_current_user
    app.dependency_overrides[get_admin_service] = lambda: svc
    audit = AuditLogService(factory)
    quota = UsageQuotaService(factory)
    app.dependency_overrides[get_audit_log_service] = lambda: audit
    app.dependency_overrides[get_usage_quota_service] = lambda: quota
    # Every test claims setup (5/minute per user); rate limits aren't under test.
    original = limiter.enabled
    limiter.enabled = False
    try:
        yield w
    finally:
        limiter.enabled = original
        asyncio.run(engine.dispose())
        try:
            os.remove(path)
        except OSError:
            pass


def _setup(w: World) -> None:
    w.acting_as = "owner"
    assert w.client.post("/api/v1/admin/setup", json={"token": TOKEN}).status_code == 200


def test_setup_status_then_claim(world: World) -> None:
    world.acting_as = "u1"
    assert world.client.get("/api/v1/admin/setup").json() == {"needs_setup": True, "enabled": True}
    bad = world.client.post("/api/v1/admin/setup", json={"token": "nope"})
    assert bad.status_code == 403 and bad.json()["code"] == "ADMIN_FORBIDDEN"

    _setup(world)
    assert world.client.get("/api/v1/admin/setup").json()["needs_setup"] is False
    world.acting_as = "u1"
    again = world.client.post("/api/v1/admin/setup", json={"token": TOKEN})
    assert again.status_code == 409 and again.json()["code"] == "ADMIN_CONFLICT"


def test_non_admin_cannot_list_or_grant(world: World) -> None:
    _setup(world)
    world.acting_as = "u1"
    assert world.client.get("/api/v1/admin/admins").status_code == 403
    assert world.client.post("/api/v1/admin/admins", json={"user_id": "u1"}).status_code == 403
    assert world.client.get("/api/v1/admin/users").status_code == 403


def test_admin_grants_but_only_owner_revokes(world: World) -> None:
    _setup(world)
    granted = world.client.post("/api/v1/admin/admins", json={"user_id": "u1"})
    assert granted.status_code == 201 and granted.json()["email"] == "ana@example.com"

    world.acting_as = "u1"  # now a (non-owner) admin
    assert world.client.post("/api/v1/admin/admins", json={"user_id": "u2"}).status_code == 201
    listed = world.client.get("/api/v1/admin/admins").json()["items"]
    assert [(a["email"], a["is_owner"]) for a in listed] == [
        ("owner@example.com", True),
        ("ana@example.com", False),
        ("ben@example.com", False),
    ]
    assert world.client.delete("/api/v1/admin/admins/u2").status_code == 403

    world.acting_as = "owner"
    assert world.client.delete("/api/v1/admin/admins/u2").status_code == 204
    assert world.client.delete("/api/v1/admin/admins/owner").status_code == 409
    world.acting_as = "u2"
    assert world.client.get("/api/v1/admin/admins").status_code == 403


def test_users_list_shows_admin_flags(world: World) -> None:
    _setup(world)
    items = world.client.get("/api/v1/admin/users").json()["items"]
    flags = {u["email"]: (u["is_admin"], u["is_owner"]) for u in items}
    assert flags["owner@example.com"] == (True, True)
    assert flags["ana@example.com"] == (False, False)


def test_admin_actions_are_audited(world: World) -> None:
    _setup(world)  # owner_claimed
    world.client.post("/api/v1/admin/admins", json={"user_id": "u1"})  # admin_granted
    world.client.patch("/api/v1/admin/users/u2", json={"is_active": False})  # user_deactivated
    world.client.delete("/api/v1/admin/admins/u1")  # admin_revoked

    log = world.client.get("/api/v1/admin/audit").json()
    assert [e["action"] for e in log["items"]] == [
        "admin_revoked",
        "user_deactivated",
        "admin_granted",
        "owner_claimed",
    ]
    assert log["items"][0]["actor_email"] == "owner@example.com"
    assert log["items"][0]["target_label"] == "ana@example.com"
    assert "query_limit_changed" in log["actions"]

    only = world.client.get("/api/v1/admin/audit?action=admin_granted").json()
    assert only["total"] == 1 and only["items"][0]["target_label"] == "ana@example.com"


def test_admin_sets_a_daily_limit(world: World) -> None:
    _setup(world)
    r = world.client.put("/api/v1/admin/users/u1/limit", json={"daily_query_limit": 25})
    assert r.status_code == 200
    assert r.json() == {
        "user_id": "u1", "daily_query_limit": 25, "effective_limit": 25, "queries_today": 0,
    }
    users = {u["email"]: u for u in world.client.get("/api/v1/admin/users").json()["items"]}
    assert users["ana@example.com"]["daily_query_limit"] == 25
    event = world.client.get("/api/v1/admin/audit").json()["items"][0]
    assert event["action"] == "query_limit_changed" and event["details"] == {"from": None, "to": 25}

    assert world.client.put("/api/v1/admin/users/u1/limit", json={"daily_query_limit": -5}).status_code == 422
    world.acting_as = "u2"  # not an admin
    assert world.client.put("/api/v1/admin/users/u1/limit", json={"daily_query_limit": 1}).status_code == 403
    assert world.client.get("/api/v1/admin/audit").status_code == 403
