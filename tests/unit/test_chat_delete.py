"""Deleting chats with foreign keys ENFORCED (as on PostgreSQL).

Regression: ``delete_session`` issued a bulk ``DELETE FROM chat_sessions``,
which skips ORM cascades; the database has no ON DELETE CASCADE on
``chat_messages.session_id`` / ``sql_versions.message_id``, so deleting any chat
with messages failed. ``delete_all_sessions`` left ``sql_versions`` behind.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import (
    Base,
    ChatMessage,
    ChatSession,
    SqlVersion,
    User,
)
from nl_to_sql.services.chat_session_service import ChatSessionService


@pytest.fixture
async def svc() -> AsyncIterator[ChatSessionService]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        db.add_all([User(id="u1", email="a@x.com"), User(id="u2", email="b@x.com")])
        db.add_all(
            [
                ChatSession(id="s1", user_id="u1", title="one"),
                ChatSession(id="s2", user_id="u1", title="two"),
                ChatSession(id="s3", user_id="u2", title="theirs"),
            ]
        )
        await db.flush()
        for sid in ("s1", "s2", "s3"):
            m = ChatMessage(session_id=sid, question="q", sql="SELECT 1", dialect="postgresql", is_valid=True)
            db.add(m)
            await db.flush()
            db.add(SqlVersion(message_id=m.id, version_number=1, sql="SELECT 1"))
        await db.commit()

    service = ChatSessionService.__new__(ChatSessionService)
    service._session_factory = factory  # type: ignore[attr-defined]
    service._engine = engine  # type: ignore[attr-defined]
    import structlog

    service._logger = structlog.get_logger("test")  # type: ignore[attr-defined]
    try:
        yield service
    finally:
        await engine.dispose()
        try:
            os.remove(path)
        except OSError:
            pass


async def _count(svc: ChatSessionService, model) -> int:
    async with svc._session_factory() as db:  # type: ignore[attr-defined]
        return int((await db.execute(select(func.count()).select_from(model))).scalar_one())


async def test_delete_one_chat_with_messages_and_sql_versions(svc: ChatSessionService) -> None:
    await svc.delete_session("s1")
    assert await _count(svc, ChatSession) == 2
    assert await _count(svc, ChatMessage) == 2
    assert await _count(svc, SqlVersion) == 2


async def test_delete_all_only_touches_that_user(svc: ChatSessionService) -> None:
    await svc.delete_all_sessions(user_id="u1")
    assert await _count(svc, ChatSession) == 1  # u2's chat remains
    assert await _count(svc, ChatMessage) == 1
    assert await _count(svc, SqlVersion) == 1


async def test_retention_purge_removes_old_messages_with_sql_versions(svc: ChatSessionService) -> None:
    from nl_to_sql.workers.retention_worker import _purge_user_history

    purged = await _purge_user_history("u1", "none", svc._session_factory)  # type: ignore[attr-defined]
    assert purged == 2  # u1's two messages
    assert await _count(svc, ChatMessage) == 1  # u2's message untouched
    assert await _count(svc, SqlVersion) == 1
    assert await _count(svc, ChatSession) == 3  # retention trims messages, keeps chats
