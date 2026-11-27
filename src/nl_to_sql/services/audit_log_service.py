"""AuditLogService — append-only record of admin actions.

Every admin route that changes something records who did what to whom
(``admin_audit_events``). Recording is best-effort: a failed write is logged
and never undoes or fails the admin action itself. Reading a user's data is not
recorded — only changes. This service is the only writer of the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.infrastructure.database.models import AdminAuditEvent

logger = structlog.get_logger(__name__)

# The actions the admin routes record (the Audit tab's filter list).
AUDIT_ACTIONS = (
    "owner_claimed",
    "admin_granted",
    "admin_revoked",
    "user_deactivated",
    "user_reactivated",
    "reactivation_approved",
    "reactivation_rejected",
    "chat_deleted",
    "connection_deleted",
    "workspace_deleted",
    "query_limit_changed",
)


@dataclass
class AuditEventView:
    id: str
    actor_email: str
    action: str
    target_type: str
    target_id: str | None
    target_label: str | None
    details: dict[str, Any] | None
    created_at: datetime


class AuditLogService:
    """Write and page through admin audit events.

    SOLID:
      S — only the audit log.
      D — session factory constructor-injected.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record(
        self,
        *,
        actor_id: str | None,
        actor_email: str,
        action: str,
        target_type: str,
        target_id: str | None = None,
        target_label: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append one event. Never raises (the action already happened)."""
        try:
            async with self._session_factory() as db:
                db.add(
                    AdminAuditEvent(
                        actor_id=actor_id,
                        actor_email=actor_email,
                        action=action,
                        target_type=target_type,
                        target_id=target_id,
                        target_label=target_label,
                        details=details,
                    )
                )
                await db.commit()
        except Exception as exc:
            logger.error(
                "Audit event could not be recorded",
                action=action,
                actor_id=actor_id,
                target_id=target_id,
                error=str(exc),
            )

    async def list_events(
        self, limit: int = 50, offset: int = 0, action: str | None = None
    ) -> tuple[list[AuditEventView], int]:
        """Newest first, optionally filtered to one action."""
        conditions = [AdminAuditEvent.action == action] if action else []
        async with self._session_factory() as db:
            total = (
                await db.execute(
                    select(func.count()).select_from(AdminAuditEvent).where(*conditions)
                )
            ).scalar_one()
            rows = (
                (
                    await db.execute(
                        select(AdminAuditEvent)
                        .where(*conditions)
                        .order_by(AdminAuditEvent.created_at.desc())
                        .limit(limit)
                        .offset(offset)
                    )
                )
                .scalars()
                .all()
            )
        return (
            [
                AuditEventView(
                    id=r.id,
                    actor_email=r.actor_email,
                    action=r.action,
                    target_type=r.target_type,
                    target_id=r.target_id,
                    target_label=r.target_label,
                    details=r.details,
                    created_at=r.created_at,
                )
                for r in rows
            ],
            int(total),
        )
