"""Unit tests for StarterContentService.

Covers the idempotency contract that makes this service safe to call on every
list-read: seeding runs exactly once per domain per user (gated by a marker in
OnboardingState.completed_items, not by "is the list empty"), and a failure in
any one domain's seeding is logged and swallowed rather than raised — the
service is called from the hot list-read path of four routes, so it must never
turn a real list-fetch into a 500.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nl_to_sql.infrastructure.database.models import OnboardingState
from nl_to_sql.services.dashboard_service import DashboardService
from nl_to_sql.services.metrics_service import MetricsService
from nl_to_sql.services.scheduled_query_service import ScheduledQueryService
from nl_to_sql.services.starter_content_service import (
    MARKER_DASHBOARDS,
    MARKER_METRICS,
    MARKER_SCHEDULES,
    MARKER_TEMPLATES,
    StarterContentService,
)

USER = "user-1"
CONN = "conn-1"


def _mock_session(onboarding_row: OnboardingState | None = None) -> AsyncMock:
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()

    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=onboarding_row)
    db.execute = AsyncMock(return_value=result)
    return db


def _service(
    db: AsyncMock,
    *,
    metrics_service: MagicMock | None = None,
    scheduled_query_service: MagicMock | None = None,
    dashboard_service: MagicMock | None = None,
) -> StarterContentService:
    factory = MagicMock(return_value=db)
    return StarterContentService(
        session_factory=factory,
        metrics_service=metrics_service or MagicMock(spec=MetricsService),
        scheduled_query_service=scheduled_query_service or MagicMock(spec=ScheduledQueryService),
        dashboard_service=dashboard_service or MagicMock(spec=DashboardService),
    )


class TestEnsureTemplatesSeeded:
    async def test_seeds_three_templates_when_no_marker(self) -> None:
        db = _mock_session(onboarding_row=None)
        svc = _service(db)

        await svc.ensure_templates_seeded(USER)

        # 3 QueryTemplate inserts + 1 OnboardingState insert (no existing row).
        assert db.add.call_count == 4
        assert db.commit.await_count >= 1

    async def test_skips_when_marker_already_present(self) -> None:
        row = OnboardingState(user_id=USER, completed_items=[MARKER_TEMPLATES])
        db = _mock_session(onboarding_row=row)
        svc = _service(db)

        await svc.ensure_templates_seeded(USER)

        db.add.assert_not_called()

    async def test_swallows_failure(self) -> None:
        db = _mock_session(onboarding_row=None)
        db.commit = AsyncMock(side_effect=RuntimeError("db down"))
        svc = _service(db)

        # Must not raise.
        await svc.ensure_templates_seeded(USER)


class TestEnsureMetricsSeeded:
    async def test_seeds_and_certifies_three_metrics(self) -> None:
        db = _mock_session(onboarding_row=None)
        metrics_service = MagicMock(spec=MetricsService)
        metrics_service.create = AsyncMock(
            side_effect=[MagicMock(metric_id=f"m{i}") for i in range(3)]
        )
        metrics_service.certify = AsyncMock()
        svc = _service(db, metrics_service=metrics_service)

        await svc.ensure_metrics_seeded(USER, CONN)

        assert metrics_service.create.await_count == 3
        assert metrics_service.certify.await_count == 3
        for _, kwargs in metrics_service.create.await_args_list:
            assert kwargs.get("is_builtin") is True

    async def test_skips_when_marker_already_present(self) -> None:
        row = OnboardingState(user_id=USER, completed_items=[MARKER_METRICS])
        db = _mock_session(onboarding_row=row)
        metrics_service = MagicMock(spec=MetricsService)
        metrics_service.create = AsyncMock()
        svc = _service(db, metrics_service=metrics_service)

        await svc.ensure_metrics_seeded(USER, CONN)

        metrics_service.create.assert_not_called()

    async def test_swallows_failure(self) -> None:
        db = _mock_session(onboarding_row=None)
        metrics_service = MagicMock(spec=MetricsService)
        metrics_service.create = AsyncMock(side_effect=RuntimeError("boom"))
        svc = _service(db, metrics_service=metrics_service)

        await svc.ensure_metrics_seeded(USER, CONN)


class TestEnsureSchedulesSeeded:
    async def test_seeds_three_schedules_already_paused(self) -> None:
        db = _mock_session(onboarding_row=None)
        scheduled_query_service = MagicMock(spec=ScheduledQueryService)
        scheduled_query_service.create = AsyncMock(
            side_effect=[MagicMock(id=f"s{i}") for i in range(3)]
        )
        svc = _service(db, scheduled_query_service=scheduled_query_service)

        await svc.ensure_schedules_seeded(USER, CONN)

        # is_paused=True must be passed directly into create() (one atomic
        # INSERT) — never create-then-pause, which leaves a race window where
        # the worker could pick up an unpaused row before a follow-up pause().
        assert scheduled_query_service.create.await_count == 3
        assert scheduled_query_service.pause.await_count == 0
        for _, kwargs in scheduled_query_service.create.await_args_list:
            assert kwargs.get("is_builtin") is True
            assert kwargs.get("is_paused") is True

    async def test_skips_when_marker_already_present(self) -> None:
        row = OnboardingState(user_id=USER, completed_items=[MARKER_SCHEDULES])
        db = _mock_session(onboarding_row=row)
        scheduled_query_service = MagicMock(spec=ScheduledQueryService)
        scheduled_query_service.create = AsyncMock()
        svc = _service(db, scheduled_query_service=scheduled_query_service)

        await svc.ensure_schedules_seeded(USER, CONN)

        scheduled_query_service.create.assert_not_called()


class TestEnsureDashboardsSeeded:
    async def test_seeds_three_dashboards(self) -> None:
        db = _mock_session(onboarding_row=None)
        dashboard_service = MagicMock(spec=DashboardService)
        dashboard_service.create = AsyncMock()
        svc = _service(db, dashboard_service=dashboard_service)

        await svc.ensure_dashboards_seeded(USER)

        assert dashboard_service.create.await_count == 3
        for _, kwargs in dashboard_service.create.await_args_list:
            assert kwargs.get("is_builtin") is True

    async def test_skips_when_marker_already_present(self) -> None:
        row = OnboardingState(user_id=USER, completed_items=[MARKER_DASHBOARDS])
        db = _mock_session(onboarding_row=row)
        dashboard_service = MagicMock(spec=DashboardService)
        dashboard_service.create = AsyncMock()
        svc = _service(db, dashboard_service=dashboard_service)

        await svc.ensure_dashboards_seeded(USER)

        dashboard_service.create.assert_not_called()

    async def test_swallows_failure(self) -> None:
        db = _mock_session(onboarding_row=None)
        dashboard_service = MagicMock(spec=DashboardService)
        dashboard_service.create = AsyncMock(side_effect=RuntimeError("boom"))
        svc = _service(db, dashboard_service=dashboard_service)

        await svc.ensure_dashboards_seeded(USER)
