"""Unit tests for TrainingDataService.get_recent_examples tenant isolation (C8).

Regression test: ``get_recent_examples`` (few-shot context injected into the
SQL-generation prompt) used to have no ``connection_id`` filter at all — it
returned the globally most-recent high-score examples across every tenant.
Runs against a temp SQLite DB (mirrors ``test_analytics_service.py``'s
real-DB pattern) so the filtering predicate is exercised for real.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import Base, TrainingDataRecord
from nl_to_sql.services.training_data_service import TrainingDataService

CONN_A = "conn-a"
CONN_B = "conn-b"


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
async def svc(db_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    now = datetime.utcnow()
    async with factory() as db:
        db.add(TrainingDataRecord(
            question="A's secret question", sql="SELECT * FROM a_secrets",
            success_score=1.0, connection_id=CONN_A, created_at=now,
        ))
        db.add(TrainingDataRecord(
            question="B's question", sql="SELECT * FROM b_table",
            success_score=1.0, connection_id=CONN_B,
            created_at=now + timedelta(seconds=1),
        ))
        db.add(TrainingDataRecord(
            question="shared question", sql="SELECT * FROM shared_table",
            success_score=1.0, connection_id=None,
            created_at=now + timedelta(seconds=2),
        ))
        # Low-score row must never surface regardless of scope.
        db.add(TrainingDataRecord(
            question="low score", sql="SELECT 1",
            success_score=0.1, connection_id=CONN_A,
            created_at=now + timedelta(seconds=3),
        ))
        await db.commit()

    service = TrainingDataService(database_url=f"sqlite+aiosqlite:///{db_path}")
    service._session_factory = factory  # reuse the seeded engine/factory
    try:
        yield service
    finally:
        await engine.dispose()


class TestGetRecentExamplesTenantIsolation:
    async def test_scoped_caller_never_sees_another_tenants_example(self, svc) -> None:
        examples = await svc.get_recent_examples(limit=10, connection_id=CONN_B)
        questions = {e["question"] for e in examples}
        assert "A's secret question" not in questions

    async def test_scoped_caller_sees_own_and_shared(self, svc) -> None:
        examples = await svc.get_recent_examples(limit=10, connection_id=CONN_A)
        questions = {e["question"] for e in examples}
        assert questions == {"A's secret question", "shared question"}

    async def test_unscoped_caller_fails_closed_to_shared_only(self, svc) -> None:
        """C7/C8: connection_id=None (e.g. a resolution failure) must only
        see shared examples, never every tenant's."""
        examples = await svc.get_recent_examples(limit=10, connection_id=None)
        questions = {e["question"] for e in examples}
        assert questions == {"shared question"}

    async def test_low_score_examples_excluded(self, svc) -> None:
        examples = await svc.get_recent_examples(limit=10, connection_id=CONN_A)
        assert all(e["question"] != "low score" for e in examples)
