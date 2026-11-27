"""AdminService against a real SQLite DB: cross-user reads, deletes, deactivation."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.core.exceptions import SessionNotFoundError, WorkspaceValidationError
from nl_to_sql.infrastructure.database.models import (
    Base,
    ChatMessage,
    ChatSession,
    RefreshToken,
    User,
)
from nl_to_sql.services.admin_service import AdminService


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
                User(id="admin", email="admin@example.com", is_verified=True),
                User(id="u1", email="ana@example.com", full_name="Ana", is_verified=True),
                User(id="u2", email="ben@example.com", is_verified=True),
                ChatSession(id="s1", user_id="u1", title="Revenue"),
                ChatSession(id="s2", user_id="u2", title="Churn"),
            ]
        )
        await db.flush()
        db.add_all(
            [
                ChatMessage(session_id="s1", question="Top customers?", sql="SELECT 1", dialect="postgresql",
                            is_valid=True, execution_result=[{"n": 1}],
                            trust={"answer": "Ana's answer", "confidence": "high"}),
                ChatMessage(session_id="s2", question="Churn by plan?", sql="SELECT 2", dialect="postgresql",
                            is_valid=True),
                RefreshToken(id="t1", user_id="u1", token_hash="h" * 64,
                             expires_at=datetime(2099, 1, 1)),
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


async def test_lists_and_searches_every_account(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = AdminService(factory)
    users, total = await svc.list_users()
    assert total == 3
    by_email = {u.email: u.chat_count for u in users}
    assert by_email["ana@example.com"] == 1 and by_email["ben@example.com"] == 1
    found, n = await svc.list_users("ANA")
    assert n == 1 and found[0].id == "u1"


async def test_reads_any_users_chat_with_answers(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = AdminService(factory)
    detail = await svc.get_user("u1")
    assert [s.title for s in detail.sessions] == ["Revenue"]
    chat = await svc.get_session("u1", "s1")
    msg = chat["messages"][0]
    assert (msg["question"], msg["answer"], msg["confidence"], msg["row_count"]) == (
        "Top customers?", "Ana's answer", "high", 1,
    )


async def test_chat_must_belong_to_the_named_user(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = AdminService(factory)
    with pytest.raises(SessionNotFoundError):
        await svc.get_session("u1", "s2")  # s2 is u2's


async def test_delete_chat_removes_messages(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = AdminService(factory)
    await svc.delete_session("admin", "u1", "s1")
    async with factory() as db:
        assert (await db.execute(select(ChatSession).where(ChatSession.id == "s1"))).first() is None
        assert (await db.execute(select(ChatMessage).where(ChatMessage.session_id == "s1"))).first() is None
        assert (await db.execute(select(ChatSession).where(ChatSession.id == "s2"))).first() is not None


async def test_deactivate_locks_out_and_revokes_tokens(factory: async_sessionmaker[AsyncSession]) -> None:
    svc = AdminService(factory)
    summary = await svc.set_active("admin", "u1", False)
    assert summary.is_active is False
    async with factory() as db:
        token = (await db.execute(select(RefreshToken).where(RefreshToken.id == "t1"))).scalar_one()
        assert token.revoked_at is not None
    assert (await svc.set_active("admin", "u1", True)).is_active is True


async def test_admin_cannot_deactivate_self(factory: async_sessionmaker[AsyncSession]) -> None:
    with pytest.raises(WorkspaceValidationError):
        await AdminService(factory).set_active("admin", "admin", False)
