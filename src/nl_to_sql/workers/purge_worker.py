"""Account purge worker: hard-delete accounts after grace period.

Run daily — idempotent. Processes AccountDeletion rows with status='scheduled'
whose purge_after timestamp has elapsed.
"""
from __future__ import annotations

from datetime import datetime

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nl_to_sql.infrastructure.database.models import AccountDeletion, User

logger = structlog.get_logger(__name__)


async def run_account_purge(session_factory: async_sessionmaker) -> dict:
    """Process pending account deletions whose grace period has elapsed."""
    purged = 0
    now = datetime.utcnow()

    async with session_factory() as db:
        result = await db.execute(
            select(AccountDeletion).where(
                AccountDeletion.status == "scheduled",
                AccountDeletion.purge_after <= now,
            )
        )
        pending = result.scalars().all()

    for deletion in pending:
        try:
            await _hard_delete_user(deletion.user_id, session_factory)
            async with session_factory() as db:
                r = await db.execute(
                    select(AccountDeletion).where(AccountDeletion.user_id == deletion.user_id)
                )
                d = r.scalar_one_or_none()
                if d:
                    d.status = "purged"
                    await db.commit()
            purged += 1
            logger.info("purge_worker: account purged", user_id=deletion.user_id)
        except Exception as exc:
            logger.error("purge_worker: failed to purge account", user_id=deletion.user_id, error=str(exc))

    return {"purged_accounts": purged}


async def _hard_delete_user(user_id: str, session_factory: async_sessionmaker) -> None:
    """Hard-delete a user. FK cascades handle child rows where configured."""
    async with session_factory() as db:
        await db.execute(delete(User).where(User.id == user_id))
        await db.commit()
    logger.info("purge_worker: hard deleted user", user_id=user_id)
