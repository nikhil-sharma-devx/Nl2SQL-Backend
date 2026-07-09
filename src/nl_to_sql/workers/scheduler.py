"""Background job scheduler — runs the idempotent maintenance workers on a loop.

Wires the previously-orphaned ``purge_worker`` (account grace-period purge) and
``retention_worker`` (per-user data-retention cleanup) into the application
lifespan so they actually run. Both workers are idempotent, so running them on a
simple ``asyncio`` interval loop is sufficient — no external cron is required.

When ``WEB_CONCURRENCY > 1`` several workers would otherwise run the jobs at the
same time. A PostgreSQL advisory lock ensures only one process executes a given
tick; the others skip it and wait for the next interval.
"""
from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.workers.purge_worker import run_account_purge
from nl_to_sql.workers.retention_worker import run_retention_cleanup

logger = structlog.get_logger(__name__)

# Distinct advisory-lock key per job so purge and retention don't block each other.
_PURGE_LOCK_KEY = "nl2sql_account_purge"
_RETENTION_LOCK_KEY = "nl2sql_retention_cleanup"


async def _run_with_advisory_lock(
    session_factory: async_sessionmaker[AsyncSession],
    lock_key: str,
    coro_factory: object,
) -> None:
    """Run ``coro_factory`` only if this process wins the advisory lock.

    The lock is held for the duration of the job and released afterwards. On a
    non-PostgreSQL backend (or if the lock call fails) we fall back to running
    the job unconditionally — single-process deployments are the common case.
    """
    acquired = False
    try:
        async with session_factory() as db:
            result = await db.execute(
                text("SELECT pg_try_advisory_lock(hashtext(:k))"),
                {"k": lock_key},
            )
            acquired = bool(result.scalar())
    except Exception:
        acquired = True  # Non-PostgreSQL or lock unavailable — proceed anyway.

    if not acquired:
        logger.info("scheduler: another worker holds the lock — skipping", job=lock_key)
        return

    try:
        summary = await coro_factory()  # type: ignore[operator]
        logger.info("scheduler: job complete", job=lock_key, **(summary or {}))
    except Exception as exc:
        logger.error("scheduler: job failed", job=lock_key, error=str(exc))
    finally:
        try:
            async with session_factory() as db:
                await db.execute(
                    text("SELECT pg_advisory_unlock(hashtext(:k))"),
                    {"k": lock_key},
                )
        except Exception:
            pass


async def run_maintenance_jobs(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Run one tick of every maintenance job, each guarded by its advisory lock."""
    await _run_with_advisory_lock(
        session_factory,
        _PURGE_LOCK_KEY,
        lambda: run_account_purge(session_factory),
    )
    await _run_with_advisory_lock(
        session_factory,
        _RETENTION_LOCK_KEY,
        lambda: run_retention_cleanup(session_factory),
    )


async def maintenance_scheduler_loop(
    session_factory: async_sessionmaker[AsyncSession],
    interval_seconds: int,
) -> None:
    """Run the maintenance jobs every ``interval_seconds`` until cancelled.

    A short initial delay lets the rest of startup settle before the first tick.
    ``asyncio.CancelledError`` propagates so the lifespan can shut the loop down
    cleanly.
    """
    logger.info("scheduler: maintenance loop started", interval_seconds=interval_seconds)
    try:
        # Let startup finish (model warm-up, schema ingest) before the first run.
        await asyncio.sleep(min(60, interval_seconds))
        while True:
            await run_maintenance_jobs(session_factory)
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("scheduler: maintenance loop stopped")
        raise
