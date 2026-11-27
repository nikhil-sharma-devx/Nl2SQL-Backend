"""Unit tests for ChatSessionService — cross-user history isolation.

Regression test for the Phase 9 leak: GET /api/v1/history used to call
``list_all_messages``/``count_all_messages`` with no ``user_id`` at all
("global across all sessions"), so any authenticated user saw every user's
chat messages. Runs against a temp SQLite DB (mirrors
``test_organization_service.py``'s real-DB pattern) so the join-based
scoping is exercised for real rather than mocked.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import Base, ChatMessage, ChatSession, User
from nl_to_sql.services.chat_session_service import ChatSessionService

USER_A = "user-a"
USER_B = "user-b"


@pytest.fixture
async def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.fixture
async def seeded(db_path):
    """Seed two users, one session each, two messages in A's session, one in B's."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    session_a = str(uuid4())
    session_b = str(uuid4())
    now = datetime.utcnow()

    async with factory() as db:
        db.add(User(id=USER_A, email="a@example.com", full_name="User A"))
        db.add(User(id=USER_B, email="b@example.com", full_name="User B"))
        db.add(ChatSession(
            id=session_a, user_id=USER_A, title="A's chat", created_at=now, updated_at=now,
        ))
        db.add(ChatSession(
            id=session_b, user_id=USER_B, title="B's chat", created_at=now, updated_at=now,
        ))
        db.add(ChatMessage(
            session_id=session_a, timestamp=now, question="A q1", sql="SELECT 1",
            dialect="postgresql", is_valid=True,
        ))
        db.add(ChatMessage(
            session_id=session_a, timestamp=now - timedelta(minutes=1),
            question="A q2", sql="SELECT 2", dialect="postgresql", is_valid=True,
        ))
        db.add(ChatMessage(
            session_id=session_b, timestamp=now, question="B q1 (secret)", sql="SELECT 3",
            dialect="postgresql", is_valid=True,
        ))
        await db.commit()

    await engine.dispose()


@pytest.fixture
async def svc(db_path, seeded):
    service = ChatSessionService(database_url=f"sqlite+aiosqlite:///{db_path}")
    # ChatSessionService's real engine passes an asyncpg-only connect_arg
    # (prepared_statement_cache_size) that aiosqlite rejects; swap in a
    # plain SQLite engine against the same file for the test, leaving the
    # service's query logic (the thing under test) untouched.
    test_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    service._session_factory = async_sessionmaker(
        bind=test_engine, expire_on_commit=False, class_=AsyncSession
    )
    try:
        yield service
    finally:
        await test_engine.dispose()
        await service.dispose()


@pytest.mark.asyncio
async def test_list_all_messages_scoped_to_user(svc: ChatSessionService) -> None:
    messages = await svc.list_all_messages(user_id=USER_A)
    assert {m.question for m in messages} == {"A q1", "A q2"}


@pytest.mark.asyncio
async def test_list_all_messages_never_leaks_another_users_question(
    svc: ChatSessionService,
) -> None:
    messages = await svc.list_all_messages(user_id=USER_A)
    assert all("secret" not in m.question for m in messages)


@pytest.mark.asyncio
async def test_count_all_messages_scoped_to_user(svc: ChatSessionService) -> None:
    assert await svc.count_all_messages(user_id=USER_A) == 2
    assert await svc.count_all_messages(user_id=USER_B) == 1


@pytest.mark.asyncio
async def test_new_user_with_no_sessions_sees_nothing(svc: ChatSessionService) -> None:
    messages = await svc.list_all_messages(user_id="brand-new-user")
    assert messages == []
    assert await svc.count_all_messages(user_id="brand-new-user") == 0


@pytest.mark.asyncio
async def test_unscoped_call_is_refused(svc: ChatSessionService) -> None:
    """There is no unscoped (every user's chats) mode: omitting user_id is an
    error, so a future caller can't silently read or wipe other users' chats."""
    with pytest.raises(TypeError):
        await svc.count_all_messages()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await svc.list_all_messages()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await svc.delete_all_sessions()  # type: ignore[call-arg]
