"""AdminService admin-account rules against a real SQLite DB.

Covers the one-time owner setup (token-gated, single owner), granting admin
(any admin) and revoking it (owner only, never the owner itself), and that a
non-owner admin can't deactivate another admin.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.core.exceptions import (
    AdminConflictError,
    AdminPermissionError,
    SessionNotFoundError,
    WorkspaceValidationError,
)
from nl_to_sql.infrastructure.database.models import Base, User
from nl_to_sql.services.admin_service import AdminService

TOKEN = "setup-token-123"


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
                User(id="owner", email="owner@example.com", is_verified=True),
                User(id="u1", email="ana@example.com", is_verified=True),
                User(id="u2", email="ben@example.com", is_verified=True),
                User(id="unverified", email="new@example.com", is_verified=False),
                User(id="inactive", email="gone@example.com", is_verified=True, is_active=False),
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


async def _flags(f: async_sessionmaker[AsyncSession], user_id: str) -> tuple[bool, bool]:
    async with f() as db:
        u = (await db.execute(select(User).where(User.id == user_id))).scalar_one()
        return bool(u.is_admin), bool(u.is_owner)


async def _setup(f: async_sessionmaker[AsyncSession]) -> AdminService:
    svc = AdminService(f, SecretStr(TOKEN))
    await svc.claim_owner("owner", TOKEN)
    return svc


# ── Setup ────────────────────────────────────────────────────────────────────


async def test_new_accounts_are_not_admins(factory: async_sessionmaker[AsyncSession]) -> None:
    assert await _flags(factory, "u1") == (False, False)


async def test_setup_claims_owner_once(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = AdminService(factory, SecretStr(TOKEN))
    assert await svc.needs_setup() is True
    account = await svc.claim_owner("owner", TOKEN)
    assert account.is_owner is True
    assert await _flags(factory, "owner") == (True, True)
    assert await svc.needs_setup() is False
    with pytest.raises(AdminConflictError):
        await svc.claim_owner("u1", TOKEN)  # second claim, even with the right token
    assert await _flags(factory, "u1") == (False, False)


async def test_setup_rejects_wrong_token(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = AdminService(factory, SecretStr(TOKEN))
    with pytest.raises(AdminPermissionError):
        await svc.claim_owner("u1", "wrong")
    assert await _flags(factory, "u1") == (False, False)


async def test_setup_disabled_without_token(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = AdminService(factory)
    assert svc.setup_enabled is False
    with pytest.raises(AdminPermissionError):
        await svc.claim_owner("u1", "anything")


async def test_database_allows_only_one_owner(factory: async_sessionmaker[AsyncSession]) -> None:
    await _setup(factory)
    async with factory() as db:
        u = (await db.execute(select(User).where(User.id == "u1"))).scalar_one()
        u.is_owner = True
        with pytest.raises(IntegrityError):
            await db.commit()


# ── Grant / revoke ───────────────────────────────────────────────────────────


async def test_any_admin_can_grant_admin(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = await _setup(factory)
    await svc.grant_admin("owner", "u1")
    await svc.grant_admin("u1", "u2")  # u1 is a non-owner admin; still may add
    assert await _flags(factory, "u2") == (True, False)
    emails = [a.email for a in await svc.list_admins()]
    assert emails == ["owner@example.com", "ana@example.com", "ben@example.com"]  # owner first


async def test_grant_rejects_existing_unverified_inactive_and_unknown(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    svc = await _setup(factory)
    with pytest.raises(AdminConflictError):
        await svc.grant_admin("owner", "owner")
    with pytest.raises(WorkspaceValidationError):
        await svc.grant_admin("owner", "unverified")
    with pytest.raises(WorkspaceValidationError):
        await svc.grant_admin("owner", "inactive")
    with pytest.raises(SessionNotFoundError):
        await svc.grant_admin("owner", "nobody")


async def test_owner_revokes_admin(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = await _setup(factory)
    await svc.grant_admin("owner", "u1")
    await svc.revoke_admin("owner", "u1")
    assert await _flags(factory, "u1") == (False, False)
    with pytest.raises(AdminConflictError):
        await svc.revoke_admin("owner", "u1")  # no longer an admin


async def test_owner_cannot_be_revoked(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = await _setup(factory)
    with pytest.raises(AdminConflictError):
        await svc.revoke_admin("owner", "owner")
    assert await _flags(factory, "owner") == (True, True)


# ── Deactivation of admins ───────────────────────────────────────────────────


async def test_only_owner_deactivates_an_admin(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = await _setup(factory)
    await svc.grant_admin("owner", "u1")
    await svc.grant_admin("owner", "u2")
    with pytest.raises(AdminPermissionError):
        await svc.set_active("u1", "u2", False)  # admin vs admin
    with pytest.raises(AdminPermissionError):
        await svc.set_active("u1", "owner", False)  # admin vs owner
    summary = await svc.set_active("owner", "u2", False, actor_is_owner=True)
    assert summary.is_active is False and summary.is_admin is True
