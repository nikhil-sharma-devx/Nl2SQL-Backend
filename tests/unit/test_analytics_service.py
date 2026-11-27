"""Unit tests for AnalyticsService — cross-user isolation.

Regression test for the Phase 9 leak: every ``AnalyticsService`` method used
to aggregate across *all* users with no ``user_id``/``organization_id``
predicate at all — ``get_popular_queries`` even returned other users'
verbatim question text. Runs against a temp SQLite DB (mirrors
``test_organization_service.py``'s real-DB pattern) so the join-based
scoping is exercised for real rather than mocked.

``get_table_usage`` uses Postgres-only ``jsonb_array_elements_text``/``::jsonb``
raw SQL that SQLite can't execute, so it's covered separately by asserting the
generated SQL text (scoping predicate + bound user_id) rather than a live
round trip.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import Base, ChatMessage, ChatSession, User
from nl_to_sql.services.analytics_service import AnalyticsService

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
    """Seed two users' sessions/messages: A has 2 valid + 1 failed, B has 1 valid (secret)."""
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
        db.add(ChatSession(id=session_a, user_id=USER_A, title="A", created_at=now, updated_at=now))
        db.add(ChatSession(id=session_b, user_id=USER_B, title="B", created_at=now, updated_at=now))
        db.add(ChatMessage(
            session_id=session_a, timestamp=now, question="How many orders?", sql="SELECT 1",
            dialect="postgresql", is_valid=True, cached=False, tokens_used=100,
            intent_type="aggregation", prompt_version="v1",
        ))
        db.add(ChatMessage(
            session_id=session_a, timestamp=now, question="How many orders?", sql="SELECT 1",
            dialect="postgresql", is_valid=False, cached=False, tokens_used=50,
            intent_type="aggregation", prompt_version="v1", validation_errors=["bad column"],
        ))
        db.add(ChatMessage(
            session_id=session_b, timestamp=now, question="B secret question", sql="SELECT 2",
            dialect="postgresql", is_valid=True, cached=False, tokens_used=200,
            intent_type="lookup", prompt_version="v2",
        ))
        await db.commit()

    await engine.dispose()


@pytest.fixture
async def svc(db_path, seeded):
    service = AnalyticsService(database_url=f"sqlite+aiosqlite:///{db_path}")
    try:
        yield service
    finally:
        await service.dispose()


@pytest.mark.asyncio
async def test_summary_scoped_to_user(svc: AnalyticsService) -> None:
    summary = await svc.get_summary(user_id=USER_A)
    assert summary["total_queries"] == 2

    summary_b = await svc.get_summary(user_id=USER_B)
    assert summary_b["total_queries"] == 1


@pytest.mark.asyncio
async def test_popular_queries_never_leaks_another_users_question(svc: AnalyticsService) -> None:
    popular = await svc.get_popular_queries(user_id=USER_A)
    assert all("secret" not in row["question"] for row in popular)
    assert all(row["question"] == "How many orders?" for row in popular)


@pytest.mark.asyncio
async def test_failure_patterns_scoped_to_user(svc: AnalyticsService) -> None:
    patterns_a = await svc.get_failure_patterns(user_id=USER_A)
    assert len(patterns_a) == 1  # A's one failed message

    patterns_b = await svc.get_failure_patterns(user_id=USER_B)
    assert patterns_b == []  # B has no failures


@pytest.mark.asyncio
async def test_intent_distribution_scoped_to_user(svc: AnalyticsService) -> None:
    dist_a = await svc.get_intent_distribution(user_id=USER_A)
    assert {row["intent_type"] for row in dist_a} == {"aggregation"}

    dist_b = await svc.get_intent_distribution(user_id=USER_B)
    assert {row["intent_type"] for row in dist_b} == {"lookup"}


@pytest.mark.asyncio
async def test_prompt_version_performance_scoped_to_user(svc: AnalyticsService) -> None:
    perf_a = await svc.get_prompt_version_performance(user_id=USER_A)
    assert {row["prompt_version"] for row in perf_a} == {"v1"}
    total_a = next(row for row in perf_a if row["prompt_version"] == "v1")
    assert total_a["total_uses"] == 2


@pytest.mark.asyncio
async def test_new_user_with_no_messages_sees_empty_analytics(svc: AnalyticsService) -> None:
    summary = await svc.get_summary(user_id="brand-new-user")
    assert summary["total_queries"] == 0
    assert await svc.get_popular_queries(user_id="brand-new-user") == []
    assert await svc.get_intent_distribution(user_id="brand-new-user") == []


@pytest.mark.asyncio
async def test_table_usage_query_is_scoped_by_user_id(svc: AnalyticsService) -> None:
    """get_table_usage uses Postgres-only jsonb SQL SQLite can't run — assert the
    generated statement joins chat_sessions and binds user_id instead of a live round trip.
    """
    captured = {}
    real_execute = AsyncSession.execute

    async def _spy_execute(self, statement, params=None, *a, **kw):
        if params:
            captured.update(params)
        else:
            try:
                captured.update(statement.compile().params)
            except Exception:
                pass
        captured["sql_text"] = str(statement)
        raise RuntimeError("stop-before-real-execute")

    AsyncSession.execute = _spy_execute
    try:
        with pytest.raises(RuntimeError):
            await svc.get_table_usage(user_id=USER_A, days=30, limit=20)
    finally:
        AsyncSession.execute = real_execute

    assert captured.get("user_id") == USER_A
    assert "chat_sessions" in captured["sql_text"]
    assert "chat_sessions.user_id" in captured["sql_text"]
