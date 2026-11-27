"""Unit tests for ScheduledQueryService.

Covers CRUD, ownership enforcement (cross-user -> ScheduleNotFoundError),
connection-ownership validation on create (delegates to ConnectionService),
pause/resume next_run_at recompute, and the due-schedule worker query. DB
access is mocked via an AsyncMock session factory (matching
test_dashboard_service.py's pattern).
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from nl_to_sql.core.exceptions import ConnectionNotFoundError, ScheduleValidationError
from nl_to_sql.infrastructure.database.models import ScheduledQuery
from nl_to_sql.services.scheduled_query_service import ScheduledQueryService

USER = "user-1"
OTHER = "user-2"
DT = datetime(2026, 1, 1, 12, 0, 0)


def _mock_session() -> AsyncMock:
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.delete = AsyncMock()
    db.expunge = MagicMock()
    return db


def _result_scalar(value: object) -> MagicMock:
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    return r


def _result_scalar_one(value: object) -> MagicMock:
    r = MagicMock()
    r.scalar_one = MagicMock(return_value=value)
    return r


def _result_scalars(items: list) -> MagicMock:
    r = MagicMock()
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=items)))
    return r


def _schedule(**overrides) -> ScheduledQuery:
    defaults = {
        "id": "sched-1",
        "user_id": USER,
        "connection_id": "conn-1",
        "name": "Daily revenue",
        "nl_prompt": "total revenue by day",
        "cron_expr": "0 8 * * *",
        "raw_schedule_text": "every morning",
        "timezone": "UTC",
        "is_paused": False,
        "notify_email": True,
        "notify_in_app": False,
        "notify_condition": "always",
        "last_row_count": None,
        "last_result_hash": None,
        "next_run_at": DT,
        "last_run_at": None,
        "last_status": None,
        "consecutive_failures": 0,
        "created_at": DT,
        "updated_at": DT,
    }
    defaults.update(overrides)
    return ScheduledQuery(**defaults)


def _service_with(
    db: AsyncMock, connection_service: MagicMock | None = None
) -> ScheduledQueryService:
    factory = MagicMock(return_value=db)
    conn_svc = connection_service or MagicMock(assert_owned=AsyncMock(return_value=None))
    return ScheduledQueryService(session_factory=factory, connection_service=conn_svc)


class TestOwnership:
    async def test_get_cross_user_raises_not_found(self) -> None:
        from nl_to_sql.core.exceptions import ScheduleNotFoundError

        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))
        svc = _service_with(db)
        with pytest.raises(ScheduleNotFoundError):
            await svc.get(OTHER, "sched-1")

    async def test_get_owned_returns_info(self) -> None:
        db = _mock_session()
        row = _schedule()
        db.execute = AsyncMock(return_value=_result_scalar(row))
        svc = _service_with(db)
        info = await svc.get(USER, "sched-1")
        assert info.id == "sched-1"
        assert info.connection_id == "conn-1"


class TestCreate:
    async def test_create_validates_connection_ownership(self) -> None:
        conn_svc = MagicMock(
            assert_owned=AsyncMock(side_effect=ConnectionNotFoundError("no"))
        )
        db = _mock_session()
        svc = _service_with(db, connection_service=conn_svc)
        with pytest.raises(ScheduleValidationError):
            await svc.create(
                USER, "conn-x", "My Schedule", "revenue today", "every morning"
            )

    async def test_create_rejects_invalid_schedule_text(self) -> None:
        db = _mock_session()
        svc = _service_with(db)
        with pytest.raises(ScheduleValidationError):
            await svc.create(
                USER, "conn-1", "My Schedule", "revenue today", "when pigs fly"
            )

    async def test_create_rejects_empty_name(self) -> None:
        db = _mock_session()
        svc = _service_with(db)
        with pytest.raises(ScheduleValidationError):
            await svc.create(USER, "conn-1", "  ", "revenue today", "every morning")

    async def test_create_enforces_per_user_cap(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar_one(25))
        svc = _service_with(db)
        with pytest.raises(ScheduleValidationError):
            await svc.create(
                USER, "conn-1", "My Schedule", "revenue today", "every morning"
            )

    async def test_create_success(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar_one(0))
        svc = _service_with(db)
        info = await svc.create(
            USER, "conn-1", "My Schedule", "revenue today", "every morning"
        )
        assert info is not None
        db.add.assert_called_once()
        db.commit.assert_awaited()


class TestPauseResume:
    async def test_pause_sets_flag(self) -> None:
        db = _mock_session()
        row = _schedule(is_paused=False)
        db.execute = AsyncMock(return_value=_result_scalar(row))
        svc = _service_with(db)
        info = await svc.pause(USER, "sched-1")
        assert info.is_paused is True

    async def test_resume_recomputes_next_run(self) -> None:
        db = _mock_session()
        row = _schedule(is_paused=True, next_run_at=datetime(2020, 1, 1))
        db.execute = AsyncMock(return_value=_result_scalar(row))
        svc = _service_with(db)
        info = await svc.resume(USER, "sched-1")
        assert info.is_paused is False
        assert info.next_run_at > datetime(2020, 1, 1)


class TestDelete:
    async def test_delete_cross_user_raises(self) -> None:
        from nl_to_sql.core.exceptions import ScheduleNotFoundError

        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))
        svc = _service_with(db)
        with pytest.raises(ScheduleNotFoundError):
            await svc.delete(OTHER, "sched-1")

    async def test_delete_owned_row(self) -> None:
        db = _mock_session()
        row = _schedule()
        db.execute = AsyncMock(return_value=_result_scalar(row))
        svc = _service_with(db)
        await svc.delete(USER, "sched-1")
        db.delete.assert_awaited_once_with(row)


class TestListSchedules:
    """Medium finding: list_schedules had no limit/offset at all."""

    async def test_returns_infos_and_total(self) -> None:
        db = _mock_session()
        rows = [_schedule()]
        db.execute = AsyncMock(side_effect=[_result_scalar_one(1), _result_scalars(rows)])
        svc = _service_with(db)

        infos, total = await svc.list_schedules(USER)
        assert total == 1
        assert len(infos) == 1

    async def test_limit_and_offset_applied_to_query(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(side_effect=[_result_scalar_one(0), _result_scalars([])])
        svc = _service_with(db)

        await svc.list_schedules(USER, limit=5, offset=10)
        select_call = db.execute.call_args_list[1].args[0]
        compiled = str(select_call)
        assert "LIMIT" in compiled.upper()
        assert "OFFSET" in compiled.upper()


class TestListDue:
    async def test_list_due_returns_matching_rows(self) -> None:
        db = _mock_session()
        due = [_schedule(next_run_at=datetime(2025, 1, 1))]
        db.execute = AsyncMock(return_value=_result_scalars(due))
        svc = _service_with(db)
        rows = await svc.list_due(before=datetime(2026, 1, 1))
        assert rows == due
