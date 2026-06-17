"""Retention worker: enforce per-user data retention settings.

Run daily (or on demand) — idempotent.
Deletes chat messages older than each user's configured retention window.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nl_to_sql.infrastructure.database.models import ChatMessage, ChatSession, UserSettings

logger = structlog.get_logger(__name__)


async def run_retention_cleanup(session_factory: async_sessionmaker) -> dict:
    """Run data retention cleanup for all users with non-forever settings."""
    purged_users = 0
    purged_messages = 0

    async with session_factory() as db:
        result = await db.execute(
            select(UserSettings).where(UserSettings.data_retention != "forever")
        )
        settings_list = result.scalars().all()

    for s in settings_list:
        count = await _purge_user_history(s.user_id, s.data_retention, session_factory)
        if count > 0:
            purged_users += 1
            purged_messages += count

    logger.info(
        "retention_worker: cleanup complete",
        purged_users=purged_users,
        purged_messages=purged_messages,
    )
    return {"purged_users": purged_users, "purged_messages": purged_messages}


async def _purge_user_history(
    user_id: str, retention: str, session_factory: async_sessionmaker
) -> int:
    cutoff: datetime | None = None
    now = datetime.utcnow()

    if retention == "7d":
        cutoff = now - timedelta(days=7)
    elif retention == "30d":
        cutoff = now - timedelta(days=30)
    elif retention == "none":
        cutoff = now
    else:
        return 0

    async with session_factory() as db:
        r = await db.execute(select(ChatSession.id).where(ChatSession.user_id == user_id))
        session_ids = [row[0] for row in r.all()]

        if not session_ids:
            return 0

        result = await db.execute(
            delete(ChatMessage)
            .where(ChatMessage.session_id.in_(session_ids))
            .where(ChatMessage.timestamp < cutoff)
        )
        count = result.rowcount
        await db.commit()

    if count > 0:
        logger.info("retention_worker: purged messages", user_id=user_id, retention=retention, count=count)
    return count
