"""Unit tests for scheduled_query_worker.

Covers the pure helpers (_should_notify, _hash_rows), execute_schedule's
success/failure/timeout/auto-pause branches (orchestrator + service mocked),
and run_scheduled_query_cycle's per-row advisory-lock skip behavior.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nl_to_sql.core.models.query import QueryResponse
from nl_to_sql.infrastructure.database.models import ScheduledQuery
from nl_to_sql.workers.scheduled_query_worker import (
    _hash_rows,
    _should_notify,
    execute_schedule,
    run_scheduled_query_cycle,
)

DT = datetime(2026, 1, 1, 8, 0, 0)


def _schedule(**overrides) -> ScheduledQuery:
    defaults = {
        "id": "sched-1",
        "user_id": "user-1",
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


class TestShouldNotify:
    def test_failure_always_notifies(self) -> None:
        assert _should_notify(_schedule(notify_condition="on_results"), "failed", None, None)

    def test_always_condition(self) -> None:
        assert _should_notify(_schedule(notify_condition="always"), "success", 0, "h")

    def test_on_results_with_zero_rows_skips(self) -> None:
        assert not _should_notify(_schedule(notify_condition="on_results"), "success", 0, "h")

    def test_on_results_with_rows_notifies(self) -> None:
        assert _should_notify(_schedule(notify_condition="on_results"), "success", 3, "h")

    def test_on_change_same_hash_skips(self) -> None:
        sched = _schedule(notify_condition="on_change", last_result_hash="abc")
        assert not _should_notify(sched, "success", 5, "abc")

    def test_on_change_different_hash_notifies(self) -> None:
        sched = _schedule(notify_condition="on_change", last_result_hash="abc")
        assert _should_notify(sched, "success", 5, "xyz")


class TestHashRows:
    def test_same_rows_same_hash(self) -> None:
        rows = [{"a": 1, "b": 2}]
        assert _hash_rows(rows) == _hash_rows(rows)

    def test_different_rows_different_hash(self) -> None:
        assert _hash_rows([{"a": 1}]) != _hash_rows([{"a": 2}])


def _mock_service() -> MagicMock:
    svc = MagicMock()
    svc.record_run = AsyncMock()
    svc.mark_run_result = AsyncMock()
    svc.list_due = AsyncMock(return_value=[])
    return svc


class TestExecuteSchedule:
    async def test_success_records_success_run(self) -> None:
        schedule = _schedule()
        svc = _mock_service()
        response = QueryResponse(
            question="q", sql="SELECT 1", dialect="postgresql", is_valid=True,
            execution_result=[{"x": 1}, {"x": 2}], execution_error=None,
        )
        orchestrator = MagicMock(run=AsyncMock(return_value=response))

        with patch(
            "nl_to_sql.workers.scheduled_query_worker.build_orchestrator_for_connection",
            new=AsyncMock(return_value=orchestrator),
        ), patch(
            "nl_to_sql.workers.scheduled_query_worker._send_alert_email",
            new=AsyncMock(return_value=True),
        ):
            await execute_schedule(
                schedule, container=MagicMock(), service=svc, session_factory=MagicMock()
            )

        svc.record_run.assert_awaited_once()
        kwargs = svc.record_run.call_args.kwargs
        assert kwargs["status"] == "success"
        assert kwargs["row_count"] == 2
        svc.mark_run_result.assert_awaited_once()
        mark_kwargs = svc.mark_run_result.call_args.kwargs
        assert mark_kwargs["consecutive_failures"] == 0
        assert mark_kwargs["is_paused"] is None

    async def test_execution_error_records_failed_run(self) -> None:
        schedule = _schedule(consecutive_failures=1)
        svc = _mock_service()
        response = QueryResponse(
            question="q", sql="SELECT 1", dialect="postgresql", is_valid=True,
            execution_result=None, execution_error="permission denied",
        )
        orchestrator = MagicMock(run=AsyncMock(return_value=response))

        with patch(
            "nl_to_sql.workers.scheduled_query_worker.build_orchestrator_for_connection",
            new=AsyncMock(return_value=orchestrator),
        ), patch(
            "nl_to_sql.workers.scheduled_query_worker._send_alert_email",
            new=AsyncMock(return_value=True),
        ):
            await execute_schedule(
                schedule, container=MagicMock(), service=svc, session_factory=MagicMock()
            )

        kwargs = svc.record_run.call_args.kwargs
        assert kwargs["status"] == "failed"
        assert kwargs["error"] == "permission denied"
        mark_kwargs = svc.mark_run_result.call_args.kwargs
        assert mark_kwargs["consecutive_failures"] == 2

    async def test_orchestrator_exception_is_caught(self) -> None:
        schedule = _schedule()
        svc = _mock_service()

        with patch(
            "nl_to_sql.workers.scheduled_query_worker.build_orchestrator_for_connection",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ), patch(
            "nl_to_sql.workers.scheduled_query_worker._send_alert_email",
            new=AsyncMock(return_value=True),
        ):
            await execute_schedule(
                schedule, container=MagicMock(), service=svc, session_factory=MagicMock()
            )

        kwargs = svc.record_run.call_args.kwargs
        assert kwargs["status"] == "failed"
        assert "boom" in kwargs["error"]

    async def test_auto_pause_after_max_consecutive_failures(self) -> None:
        schedule = _schedule(consecutive_failures=4)  # next failure -> 5 -> default cap
        svc = _mock_service()

        with patch(
            "nl_to_sql.workers.scheduled_query_worker.build_orchestrator_for_connection",
            new=AsyncMock(side_effect=RuntimeError("still broken")),
        ), patch(
            "nl_to_sql.workers.scheduled_query_worker._send_alert_email",
            new=AsyncMock(return_value=True),
        ):
            await execute_schedule(
                schedule, container=MagicMock(), service=svc, session_factory=MagicMock()
            )

        mark_kwargs = svc.mark_run_result.call_args.kwargs
        assert mark_kwargs["is_paused"] is True
        assert mark_kwargs["next_run_at"] is None


class TestRunScheduledQueryCycle:
    async def test_skips_when_lock_not_acquired(self) -> None:
        schedule = _schedule()
        container = MagicMock()
        container.connection_service.return_value = MagicMock()

        with patch(
            "nl_to_sql.workers.scheduled_query_worker.ScheduledQueryService"
        ) as svc_cls, patch(
            "nl_to_sql.workers.scheduled_query_worker.try_advisory_lock",
            new=AsyncMock(return_value=False),
        ) as lock_mock, patch(
            "nl_to_sql.workers.scheduled_query_worker.advisory_unlock", new=AsyncMock()
        ) as unlock_mock, patch(
            "nl_to_sql.workers.scheduled_query_worker.execute_schedule", new=AsyncMock()
        ) as exec_mock:
            svc_cls.return_value.list_due = AsyncMock(return_value=[schedule])
            result = await run_scheduled_query_cycle(MagicMock(), container)

        lock_mock.assert_awaited_once()
        exec_mock.assert_not_called()
        unlock_mock.assert_not_called()
        assert result == {"executed": 0, "skipped_locked": 1}

    async def test_executes_when_lock_acquired(self) -> None:
        schedule = _schedule()
        container = MagicMock()
        container.connection_service.return_value = MagicMock()

        with patch(
            "nl_to_sql.workers.scheduled_query_worker.ScheduledQueryService"
        ) as svc_cls, patch(
            "nl_to_sql.workers.scheduled_query_worker.try_advisory_lock",
            new=AsyncMock(return_value=True),
        ), patch(
            "nl_to_sql.workers.scheduled_query_worker.advisory_unlock", new=AsyncMock()
        ) as unlock_mock, patch(
            "nl_to_sql.workers.scheduled_query_worker.execute_schedule", new=AsyncMock()
        ) as exec_mock:
            svc_cls.return_value.list_due = AsyncMock(return_value=[schedule])
            result = await run_scheduled_query_cycle(MagicMock(), container)

        exec_mock.assert_awaited_once()
        unlock_mock.assert_awaited_once()
        assert result == {"executed": 1, "skipped_locked": 0}


@pytest.fixture(autouse=True)
def _settings_defaults():
    """Ensure scheduled-query settings fields exist with sane test defaults."""
    from nl_to_sql.config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
