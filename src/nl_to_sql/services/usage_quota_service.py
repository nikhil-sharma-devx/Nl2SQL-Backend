"""UsageQuotaService — per-user daily AI-query limits.

Each AI query generation (POST /query, /query/stream) consumes one unit from
the user's allowance for the current UTC day. The allowance is the user's own
``users.daily_query_limit`` if an admin set one, else the platform default
(``DEFAULT_DAILY_QUERY_LIMIT``; 0 = unlimited). Admins are never limited.

Counts live in ``user_daily_usage`` (not chat history), so deleting chats does
not give the allowance back. This service is the only writer of that table
and of ``users.daily_query_limit``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.core.exceptions import DailyQuotaExceededError, SessionNotFoundError
from nl_to_sql.infrastructure.database.models import User, UserDailyUsage

logger = structlog.get_logger(__name__)

MAX_DAILY_LIMIT = 100_000


def _today() -> date:
    return datetime.now(UTC).date()


def _seconds_until_reset() -> int:
    now = datetime.now(UTC)
    midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    return max(1, int((midnight - now).total_seconds()))


class UsageQuotaService:
    """Daily allowance lookup, atomic consumption, and admin-set limits.

    SOLID:
      S — only the daily query allowance.
      D — session factory and platform default constructor-injected.
    """

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], default_limit: int = 0
    ) -> None:
        self._session_factory = session_factory
        self._default_limit = max(0, int(default_limit))

    def _effective(self, own_limit: int | None, is_admin: bool) -> int | None:
        """The limit that applies: None = unlimited."""
        if is_admin:
            return None
        limit = self._default_limit if own_limit is None else own_limit
        return limit or None

    async def limit_for(self, user_id: str, is_admin: bool) -> int | None:
        async with self._session_factory() as db:
            own = (
                await db.execute(select(User.daily_query_limit).where(User.id == user_id))
            ).scalar_one_or_none()
        return self._effective(own, is_admin)

    async def effective_limit(self, user_id: str) -> int | None:
        """The limit that applies to ``user_id`` right now (reads their admin flag)."""
        async with self._session_factory() as db:
            row = (
                await db.execute(
                    select(User.daily_query_limit, User.is_admin).where(User.id == user_id)
                )
            ).first()
        if row is None:
            raise SessionNotFoundError("User not found.")
        return self._effective(row[0], bool(row[1]))

    async def used_today(self, user_id: str) -> int:
        async with self._session_factory() as db:
            n = (
                await db.execute(
                    select(UserDailyUsage.query_count).where(
                        UserDailyUsage.user_id == user_id, UserDailyUsage.day == _today()
                    )
                )
            ).scalar_one_or_none()
        return int(n or 0)

    async def _ensure_row(self, db: AsyncSession, user_id: str, day: date) -> None:
        dialect = db.get_bind().dialect.name
        values: dict[str, Any] = {"user_id": user_id, "day": day, "query_count": 0}
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            await db.execute(pg_insert(UserDailyUsage).values(**values).on_conflict_do_nothing())
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            await db.execute(
                sqlite_insert(UserDailyUsage).values(**values).on_conflict_do_nothing()
            )
        else:
            exists = (
                await db.execute(
                    select(UserDailyUsage.user_id).where(
                        UserDailyUsage.user_id == user_id, UserDailyUsage.day == day
                    )
                )
            ).first()
            if exists is None:
                try:
                    async with db.begin_nested():
                        db.add(UserDailyUsage(**values))
                except IntegrityError:
                    pass  # a concurrent request created it

    async def consume(self, user_id: str, is_admin: bool) -> None:
        """Count one AI query, or raise ``DailyQuotaExceededError`` at the limit.

        The increment is a single conditional UPDATE (``query_count < limit``),
        so concurrent requests can never push a user past their allowance.
        Unlimited users are still counted (for usage display).
        """
        limit = await self.limit_for(user_id, is_admin)
        day = _today()
        async with self._session_factory() as db:
            await self._ensure_row(db, user_id, day)
            stmt = (
                update(UserDailyUsage)
                .where(UserDailyUsage.user_id == user_id, UserDailyUsage.day == day)
                .values(query_count=UserDailyUsage.query_count + 1)
            )
            if limit is not None:
                stmt = stmt.where(UserDailyUsage.query_count < limit)
            result = await db.execute(stmt)
            await db.commit()
        if limit is not None and result.rowcount == 0:  # type: ignore[attr-defined]
            logger.info("Daily query limit reached", user_id=user_id, limit=limit)
            raise DailyQuotaExceededError(
                f"You've used all {limit} queries for today. Your allowance resets at "
                "midnight UTC — ask an admin if you need more.",
                retry_after=_seconds_until_reset(),
            )

    async def set_limit(self, user_id: str, limit: int | None) -> tuple[str, int | None]:
        """Set (or clear, with None) a user's own daily limit. Returns (email, old)."""
        if limit is not None and not 0 <= limit <= MAX_DAILY_LIMIT:
            raise ValueError(f"Limit must be between 0 and {MAX_DAILY_LIMIT}.")
        async with self._session_factory() as db:
            user = await db.get(User, user_id)
            if user is None:
                raise SessionNotFoundError("User not found.")
            old = user.daily_query_limit
            user.daily_query_limit = limit
            user.updated_at = datetime.utcnow()
            await db.commit()
            email = user.email
        logger.info("Daily query limit changed", user_id=user_id, old=old, new=limit)
        return email, old
