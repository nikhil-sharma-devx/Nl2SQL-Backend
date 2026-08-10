"""ScheduledQueryService — per-user recurring NL queries with email alerting.

Owns the ``scheduled_queries`` + ``scheduled_query_runs`` tables (Scheduled
Queries & Alerts feature). Every read/write is scoped by ``user_id``; a
cross-user lookup raises ``ScheduleNotFoundError`` (never confirming that
another user's schedule exists) — mirrors ``ConnectionService._get_owned``.

The session factory is constructor-injected (mirrors ``DashboardService``) —
this service never instantiates infrastructure. ``ConnectionService`` is also
injected, used *only* to validate that ``connection_id`` belongs to the
caller before a schedule is created/updated against it — this service never
fetches a DSN or builds a live DB client itself (that's the worker's job at
execution time, via the same ``ConnectionService``).

``run_now`` and the worker's per-tick execution share one execution path
(``services.scheduled_query_worker.execute_schedule``) so manual and
scheduled runs can never behave differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from nl_to_sql.core.exceptions import (
    ConnectionNotFoundError,
    ScheduleNotFoundError,
    ScheduleValidationError,
)
from nl_to_sql.infrastructure.database.models import ScheduledQuery, ScheduledQueryRun
from nl_to_sql.services.connection_service import ConnectionService
from nl_to_sql.services.cron_utils import compute_next_run, parse_schedule_text

logger = structlog.get_logger(__name__)

_MAX_SCHEDULES_PER_USER = 25
_MAX_HISTORY_PER_SCHEDULE = 50
_VALID_NOTIFY_CONDITIONS = frozenset({"always", "on_results", "on_change"})


@dataclass
class ScheduleInfo:
    """Public view of a schedule."""

    id: str
    connection_id: str
    name: str
    nl_prompt: str
    cron_expr: str
    raw_schedule_text: str | None
    timezone: str
    is_paused: bool
    notify_email: bool
    notify_in_app: bool
    notify_condition: str
    is_builtin: bool
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_status: str | None
    consecutive_failures: int
    created_at: datetime
    updated_at: datetime


class ScheduledQueryService:
    """Per-user scheduled-query CRUD + pause/resume/history."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        connection_service: ConnectionService,
    ) -> None:
        self._session_factory = session_factory
        self._connection_service = connection_service
        self._log = logger.bind(service="ScheduledQuery")

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Expose the session factory read-only, for the ``run-now`` route.

        Lets the route call ``workers.scheduled_query_worker.execute_schedule``
        (the same execution path the worker tick uses) without reaching into a
        private attribute.
        """
        return self._session_factory

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _get_owned(
        self,
        db: AsyncSession,
        user_id: str,
        schedule_id: str,
    ) -> ScheduledQuery:
        conditions = [
            ScheduledQuery.id == schedule_id,
            ScheduledQuery.user_id == user_id,
        ]
        row = (await db.execute(select(ScheduledQuery).where(*conditions))).scalar_one_or_none()
        if row is None:
            raise ScheduleNotFoundError("Schedule not found.")
        return row

    async def _assert_connection_owned(self, user_id: str, connection_id: str) -> None:
        try:
            await self._connection_service.assert_owned(user_id, connection_id)
        except ConnectionNotFoundError as exc:
            raise ScheduleValidationError("Unknown connection.") from exc

    @staticmethod
    def _to_info(row: ScheduledQuery) -> ScheduleInfo:
        return ScheduleInfo(
            id=row.id,
            connection_id=row.connection_id,
            name=row.name,
            nl_prompt=row.nl_prompt,
            cron_expr=row.cron_expr,
            raw_schedule_text=row.raw_schedule_text,
            timezone=row.timezone,
            is_paused=row.is_paused,
            notify_email=row.notify_email,
            notify_in_app=row.notify_in_app,
            notify_condition=row.notify_condition,
            is_builtin=row.is_builtin,
            next_run_at=row.next_run_at,
            last_run_at=row.last_run_at,
            last_status=row.last_status,
            consecutive_failures=row.consecutive_failures,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _validate_notify_condition(value: str) -> str:
        if value not in _VALID_NOTIFY_CONDITIONS:
            raise ScheduleValidationError(
                f"notify_condition must be one of {sorted(_VALID_NOTIFY_CONDITIONS)}."
            )
        return value

    # ── CRUD ─────────────────────────────────────────────────────────────────────

    async def list_schedules(
        self,
        user_id: str,
        connection_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ScheduleInfo], int]:
        """Return ``(schedules, total)`` for a user, oldest first.

        Medium: this endpoint had no limit/offset at all (unbounded list),
        unlike every sibling list endpoint (dashboards, connections, ...).
        """
        async with self._session_factory() as db:
            conditions = [ScheduledQuery.user_id == user_id]
            if connection_id is not None:
                conditions.append(ScheduledQuery.connection_id == connection_id)

            total = (
                await db.execute(
                    select(func.count()).select_from(ScheduledQuery).where(*conditions)
                )
            ).scalar_one()

            rows = (
                (
                    await db.execute(
                        select(ScheduledQuery)
                        .where(*conditions)
                        .order_by(ScheduledQuery.created_at)
                        .limit(limit)
                        .offset(offset)
                    )
                )
                .scalars()
                .all()
            )
        return [self._to_info(r) for r in rows], int(total)

    async def get(self, user_id: str, schedule_id: str) -> ScheduleInfo:
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, schedule_id)
        return self._to_info(row)

    async def create(
        self,
        user_id: str,
        connection_id: str,
        name: str,
        nl_prompt: str,
        schedule_text: str,
        timezone: str = "UTC",
        notify_email: bool = True,
        notify_condition: str = "always",
        is_builtin: bool = False,
        is_paused: bool = False,
    ) -> ScheduleInfo:
        name = name.strip()
        nl_prompt = nl_prompt.strip()
        if not name:
            raise ScheduleValidationError("Schedule name cannot be empty.")
        if not nl_prompt:
            raise ScheduleValidationError("Schedule query cannot be empty.")
        notify_condition = self._validate_notify_condition(notify_condition)

        await self._assert_connection_owned(user_id, connection_id)
        cron_expr = parse_schedule_text(schedule_text)
        next_run_at = compute_next_run(cron_expr, timezone)

        async with self._session_factory() as db:
            count = (
                await db.execute(
                    select(func.count())
                    .select_from(ScheduledQuery)
                    .where(ScheduledQuery.user_id == user_id)
                )
            ).scalar_one()
            if count >= _MAX_SCHEDULES_PER_USER:
                raise ScheduleValidationError(
                    f"Maximum of {_MAX_SCHEDULES_PER_USER} schedules per user reached."
                )

            row = ScheduledQuery(
                user_id=user_id,
                connection_id=connection_id,
                name=name[:200],
                nl_prompt=nl_prompt,
                cron_expr=cron_expr,
                raw_schedule_text=schedule_text[:500],
                timezone=timezone,
                notify_email=notify_email,
                notify_condition=notify_condition,
                next_run_at=next_run_at,
                is_builtin=is_builtin,
                is_paused=is_paused,
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)

        self._log.info(
            "schedule created", user_id=user_id, schedule_id=row.id, connection_id=connection_id
        )
        return self._to_info(row)

    async def update(
        self,
        user_id: str,
        schedule_id: str,
        *,
        name: str | None = None,
        nl_prompt: str | None = None,
        schedule_text: str | None = None,
        timezone: str | None = None,
        notify_email: bool | None = None,
        notify_condition: str | None = None,
    ) -> ScheduleInfo:
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, schedule_id)

            if name is not None:
                name = name.strip()
                if not name:
                    raise ScheduleValidationError("Schedule name cannot be empty.")
                row.name = name[:200]
            if nl_prompt is not None:
                nl_prompt = nl_prompt.strip()
                if not nl_prompt:
                    raise ScheduleValidationError("Schedule query cannot be empty.")
                row.nl_prompt = nl_prompt
            if notify_email is not None:
                row.notify_email = notify_email
            if notify_condition is not None:
                row.notify_condition = self._validate_notify_condition(notify_condition)

            recompute = False
            if schedule_text is not None:
                row.cron_expr = parse_schedule_text(schedule_text)
                row.raw_schedule_text = schedule_text[:500]
                recompute = True
            if timezone is not None:
                row.timezone = timezone
                recompute = True
            if recompute:
                row.next_run_at = compute_next_run(row.cron_expr, row.timezone)

            row.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(row)

        self._log.info("schedule updated", user_id=user_id, schedule_id=schedule_id)
        return self._to_info(row)

    async def delete(self, user_id: str, schedule_id: str) -> None:
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, schedule_id)
            await db.delete(row)
            await db.commit()
        self._log.info("schedule deleted", user_id=user_id, schedule_id=schedule_id)

    async def pause(self, user_id: str, schedule_id: str) -> ScheduleInfo:
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, schedule_id)
            row.is_paused = True
            row.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(row)
        self._log.info("schedule paused", user_id=user_id, schedule_id=schedule_id)
        return self._to_info(row)

    async def resume(self, user_id: str, schedule_id: str) -> ScheduleInfo:
        """Resume a paused schedule, recomputing ``next_run_at`` from now.

        Recomputing (rather than reusing a stale ``next_run_at``) avoids firing
        a backlog of missed runs for a schedule that was paused for a while.
        """
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, schedule_id)
            row.is_paused = False
            row.next_run_at = compute_next_run(row.cron_expr, row.timezone)
            row.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(row)
        self._log.info("schedule resumed", user_id=user_id, schedule_id=schedule_id)
        return self._to_info(row)

    # ── History ──────────────────────────────────────────────────────────────────

    async def list_history(
        self, user_id: str, schedule_id: str, limit: int = 20, offset: int = 0
    ) -> tuple[list[ScheduledQueryRun], int]:
        async with self._session_factory() as db:
            await self._get_owned(db, user_id, schedule_id)  # ownership check
            total = (
                await db.execute(
                    select(func.count())
                    .select_from(ScheduledQueryRun)
                    .where(ScheduledQueryRun.schedule_id == schedule_id)
                )
            ).scalar_one()
            rows = (
                (
                    await db.execute(
                        select(ScheduledQueryRun)
                        .where(ScheduledQueryRun.schedule_id == schedule_id)
                        .order_by(ScheduledQueryRun.started_at.desc())
                        .limit(limit)
                        .offset(offset)
                    )
                )
                .scalars()
                .all()
            )
        return list(rows), int(total)

    async def record_run(
        self,
        schedule_id: str,
        *,
        status: str,
        started_at: datetime,
        finished_at: datetime | None,
        row_count: int | None,
        generated_sql: str | None,
        error: str | None,
        notified: bool,
        duration_ms: int | None,
    ) -> ScheduledQueryRun:
        """Persist one execution record and trim history to the retention cap.

        Called by the worker (and by ``run_now``, via the shared execution
        helper) after every attempt, success or failure.
        """
        async with self._session_factory() as db:
            run = ScheduledQueryRun(
                schedule_id=schedule_id,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                row_count=row_count,
                generated_sql=generated_sql,
                error=error,
                notified=notified,
                duration_ms=duration_ms,
            )
            db.add(run)
            await db.commit()
            await db.refresh(run)

            stale_ids = (
                (
                    await db.execute(
                        select(ScheduledQueryRun.id)
                        .where(ScheduledQueryRun.schedule_id == schedule_id)
                        .order_by(ScheduledQueryRun.started_at.desc())
                        .offset(_MAX_HISTORY_PER_SCHEDULE)
                    )
                )
                .scalars()
                .all()
            )
            if stale_ids:
                await db.execute(
                    ScheduledQueryRun.__table__.delete().where(ScheduledQueryRun.id.in_(stale_ids))
                )
                await db.commit()
        return run

    async def get_owned_row(self, user_id: str, schedule_id: str) -> ScheduledQuery:
        """Fetch the ORM row directly (owned check) — used by ``run_now``."""
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, schedule_id)
            db.expunge(row)
        return row

    async def mark_run_result(
        self,
        schedule_id: str,
        *,
        status: str,
        next_run_at: datetime | None,
        row_count: int | None,
        result_hash: str | None,
        consecutive_failures: int,
        is_paused: bool | None = None,
    ) -> None:
        """Update the parent schedule's rollup fields after an execution attempt."""
        async with self._session_factory() as db:
            row = (
                await db.execute(select(ScheduledQuery).where(ScheduledQuery.id == schedule_id))
            ).scalar_one_or_none()
            if row is None:
                return
            row.last_status = status
            row.last_run_at = datetime.utcnow()
            row.next_run_at = next_run_at
            if row_count is not None:
                row.last_row_count = row_count
            if result_hash is not None:
                row.last_result_hash = result_hash
            row.consecutive_failures = consecutive_failures
            if is_paused is not None:
                row.is_paused = is_paused
            await db.commit()

    # ── Worker support ───────────────────────────────────────────────────────────

    async def list_due(self, before: datetime) -> list[ScheduledQuery]:
        """Return non-paused schedules whose ``next_run_at`` has passed.

        Used by the worker's tick — not user-scoped (cross-user by design; the
        worker processes every due schedule regardless of owner).
        """
        async with self._session_factory() as db:
            rows = (
                (
                    await db.execute(
                        select(ScheduledQuery)
                        .options(selectinload(ScheduledQuery.runs))
                        .where(
                            ScheduledQuery.is_paused.is_(False),
                            ScheduledQuery.next_run_at.is_not(None),
                            ScheduledQuery.next_run_at <= before,
                        )
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)
