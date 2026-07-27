"""Scheduled-query worker — executes due ``ScheduledQuery`` rows and alerts.

Driven by a dedicated, independent asyncio loop (``scheduled_query_scheduler_loop``,
started in the app lifespan alongside — but separate from — the daily
``maintenance_scheduler_loop``, since schedules need much finer-grained
polling than once-a-day maintenance jobs).

Dedup across multiple worker processes uses a **per-row** Postgres advisory
lock (``sq:<schedule_id>``), not one global key — otherwise all schedules
would serialize behind each other. ``execute_schedule`` is the single
execution path shared by both the tick and the ``POST /schedules/{id}/run-now``
route (which calls it directly, without the per-row lock, since a manual
run is user-initiated and not competing with other worker processes for the
same row), so manual and scheduled runs can never behave differently.

Alerting is email-only for v1 (see ``ScheduledQuery.notify_in_app`` docstring
in ``infrastructure/database/models.py``) — reuses the already-built SMTP path
(``services/digest_service.send_digest_email``).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.models.query import QueryRequest
from nl_to_sql.infrastructure.database.models import ScheduledQuery, User
from nl_to_sql.services.cron_utils import compute_next_run
from nl_to_sql.services.digest_service import send_digest_email
from nl_to_sql.services.orchestrator_factory import build_orchestrator_for_connection
from nl_to_sql.services.scheduled_query_service import ScheduledQueryService
from nl_to_sql.workers.scheduler import advisory_unlock, try_advisory_lock

if TYPE_CHECKING:
    from nl_to_sql.config.container import ApplicationContainer

logger = structlog.get_logger(__name__)


def _lock_key(schedule_id: str) -> str:
    return f"sq:{schedule_id}"


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _should_notify(schedule: ScheduledQuery, status: str, row_count: int | None, result_hash: str | None) -> bool:
    """Decide whether this run's outcome warrants an alert email.

    Failures always notify (the caller still gates on ``notify_email``).
    Successes are gated by ``notify_condition``.
    """
    if status != "success":
        return True
    if schedule.notify_condition == "on_results":
        return bool(row_count)
    if schedule.notify_condition == "on_change":
        return result_hash != schedule.last_result_hash
    return True  # 'always'


async def _send_alert_email(
    session_factory: async_sessionmaker[AsyncSession],
    schedule: ScheduledQuery,
    *,
    status: str,
    row_count: int | None,
    error: str | None,
    auto_paused: bool,
) -> bool:
    """Send a "schedule fired" email via the existing SMTP path. Returns True on send."""
    settings = get_settings()
    if not (settings.smtp_username and settings.smtp_password):
        logger.info("scheduled_query: SMTP not configured — skipping alert email")
        return False

    async with session_factory() as db:
        user = (
            await db.execute(select(User).where(User.id == schedule.user_id))
        ).scalar_one_or_none()
    if user is None or not user.email:
        return False

    app_url = settings.app_base_url.strip().rstrip("/") or "http://localhost:5173"

    if auto_paused:
        subject = f"[NL2SQL] Schedule '{schedule.name}' auto-disabled after repeated failures"
        text = (
            f"Your scheduled query '{schedule.name}' has failed "
            f"{schedule.consecutive_failures + 1} times in a row and has been "
            f"automatically paused.\n\nLast error: {error}\n\n"
            f"View and resume it: {app_url}/schedules"
        )
    elif status != "success":
        subject = f"[NL2SQL] Scheduled query '{schedule.name}' failed"
        text = f"Your scheduled query '{schedule.name}' failed.\n\nError: {error}\n\nView it: {app_url}/schedules"
    else:
        subject = f"[NL2SQL] Scheduled query '{schedule.name}' — {row_count or 0} row(s)"
        text = (
            f"Your scheduled query '{schedule.name}' ran successfully with "
            f"{row_count or 0} row(s).\n\nQuestion: {schedule.nl_prompt}\n\n"
            f"View it: {app_url}/schedules"
        )
    html = f"<p>{text.replace(chr(10), '<br>')}</p>"

    return await send_digest_email(user.email, subject, text, html)


async def execute_schedule(
    schedule: ScheduledQuery,
    *,
    container: ApplicationContainer,
    service: ScheduledQueryService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Execute one schedule end-to-end: orchestrator run -> history -> rollup -> alert.

    Never raises — every failure mode (timeout, orchestrator error) is caught
    and recorded as a failed run so the worker tick can keep processing other
    due schedules.
    """
    settings = get_settings()
    started_at = datetime.utcnow()
    status = "failed"
    row_count: int | None = None
    generated_sql: str | None = None
    error: str | None = None
    result_hash: str | None = None

    try:
        orchestrator = await build_orchestrator_for_connection(
            container, settings, schedule.user_id, schedule.connection_id
        )
        request = QueryRequest(question=schedule.nl_prompt, execute=True)
        response = await asyncio.wait_for(
            orchestrator.run(request),
            timeout=settings.scheduled_query_timeout_seconds,
        )
        generated_sql = response.sql
        if response.execution_error:
            status = "failed"
            error = response.execution_error
        else:
            status = "success"
            rows = response.execution_result or []
            row_count = len(rows)
            result_hash = _hash_rows(rows)
    except TimeoutError:
        status = "timeout"
        error = f"Execution exceeded {settings.scheduled_query_timeout_seconds}s timeout."
    except Exception as exc:
        status = "failed"
        error = str(exc)

    finished_at = datetime.utcnow()
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)

    consecutive_failures = 0 if status == "success" else schedule.consecutive_failures + 1
    auto_pause = consecutive_failures >= settings.scheduled_query_max_consecutive_failures

    should_notify = _should_notify(schedule, status, row_count, result_hash)
    notified = False
    if should_notify and schedule.notify_email:
        try:
            notified = await _send_alert_email(
                session_factory,
                schedule,
                status=status,
                row_count=row_count,
                error=error,
                auto_paused=auto_pause,
            )
        except Exception as exc:
            logger.error("scheduled_query: alert email failed", schedule_id=schedule.id, error=str(exc))

    next_run_at = (
        None if auto_pause else compute_next_run(schedule.cron_expr, schedule.timezone, after=finished_at)
    )

    await service.record_run(
        schedule.id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        row_count=row_count,
        generated_sql=generated_sql,
        error=error,
        notified=notified,
        duration_ms=duration_ms,
    )
    await service.mark_run_result(
        schedule.id,
        status=status,
        next_run_at=next_run_at,
        row_count=row_count,
        result_hash=result_hash,
        consecutive_failures=consecutive_failures,
        is_paused=True if auto_pause else None,
    )

    if auto_pause:
        logger.warning(
            "scheduled_query: auto-paused after repeated failures",
            schedule_id=schedule.id,
            consecutive_failures=consecutive_failures,
        )


async def run_scheduled_query_cycle(
    session_factory: async_sessionmaker[AsyncSession],
    container: ApplicationContainer,
) -> dict[str, int]:
    """Execute every due, non-paused schedule. Returns a summary.

    Dedup is per-row (``sq:<schedule_id>``) so schedules never serialize behind
    each other, unlike the job-level locks in ``workers/scheduler.py``.
    """
    service = ScheduledQueryService(
        session_factory=session_factory, connection_service=container.connection_service()
    )
    due = await service.list_due(before=datetime.utcnow())

    executed = 0
    skipped_locked = 0
    for schedule in due:
        lock_key = _lock_key(schedule.id)
        acquired = await try_advisory_lock(session_factory, lock_key)
        if not acquired:
            skipped_locked += 1
            continue
        try:
            await execute_schedule(
                schedule, container=container, service=service, session_factory=session_factory
            )
            executed += 1
        finally:
            await advisory_unlock(session_factory, lock_key)

    if executed or skipped_locked:
        logger.info(
            "scheduled_query: cycle complete", executed=executed, skipped_locked=skipped_locked
        )
    return {"executed": executed, "skipped_locked": skipped_locked}


async def scheduled_query_scheduler_loop(
    session_factory: async_sessionmaker[AsyncSession],
    container: ApplicationContainer,
    interval_seconds: int,
) -> None:
    """Run ``run_scheduled_query_cycle`` every ``interval_seconds`` until cancelled.

    Independent of ``maintenance_scheduler_loop`` — that loop's default
    interval (once a day) is far too coarse for "daily at 9"-style schedules.
    """
    logger.info("scheduled_query: scheduler loop started", interval_seconds=interval_seconds)
    try:
        await asyncio.sleep(min(30, interval_seconds))
        while True:
            await run_scheduled_query_cycle(session_factory, container)
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("scheduled_query: scheduler loop stopped")
        raise
