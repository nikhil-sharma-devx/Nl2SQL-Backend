"""AdminService — platform-wide view and control for admin accounts.

Admins (emails in ``ADMIN_EMAILS``) can list every account, open any user's
chats, connections and teams, delete chats / connections / workspaces, and
deactivate or reactivate accounts. Every route that uses this service is
behind ``require_admin``; this service itself does no permission checks, so
it must never be wired to a non-admin route.

Deliberately NOT here: reading decrypted connection strings or API keys —
admins see that a connection exists and its masked host, never the secret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.core.exceptions import SessionNotFoundError, WorkspaceValidationError
from nl_to_sql.infrastructure.database.models import (
    ChatMessage,
    ChatSession,
    RefreshToken,
    SqlVersion,
    User,
    UserDatabaseConnection,
    UserLoginSession,
    Workspace,
    WorkspaceMember,
)

logger = structlog.get_logger(__name__)


@dataclass
class AdminUserSummary:
    id: str
    email: str
    full_name: str | None
    is_active: bool
    is_verified: bool
    auth_provider: str
    created_at: datetime
    chat_count: int
    connection_count: int
    workspace_count: int


@dataclass
class AdminSessionSummary:
    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


@dataclass
class AdminUserDetail:
    user: AdminUserSummary
    sessions: list[AdminSessionSummary]
    connections: list[dict[str, Any]] = field(default_factory=list)
    workspaces: list[dict[str, Any]] = field(default_factory=list)


class AdminService:
    """Cross-user read and control for admins.

    SOLID:
      S — admin-only cross-user operations; no permission logic (routes do it).
      D — session factory constructor-injected.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ── Users ────────────────────────────────────────────────────────────────

    async def list_users(
        self, query: str = "", limit: int = 50, offset: int = 0
    ) -> tuple[list[AdminUserSummary], int]:
        q = " ".join(query.split()).lower()
        conditions = []
        if q:
            pattern = f"%{q}%"
            conditions.append(
                or_(
                    func.lower(User.email).like(pattern),
                    func.lower(func.coalesce(User.full_name, "")).like(pattern),
                )
            )
        async with self._session_factory() as db:
            total = (
                await db.execute(select(func.count()).select_from(User).where(*conditions))
            ).scalar_one()
            users = (
                (
                    await db.execute(
                        select(User)
                        .where(*conditions)
                        .order_by(User.created_at.desc())
                        .limit(limit)
                        .offset(offset)
                    )
                )
                .scalars()
                .all()
            )
            summaries = [await self._summary(db, u) for u in users]
        return summaries, int(total)

    @staticmethod
    async def _summary(db: AsyncSession, u: User) -> AdminUserSummary:
        chats = (
            await db.execute(
                select(func.count()).select_from(ChatSession).where(ChatSession.user_id == u.id)
            )
        ).scalar_one()
        conns = (
            await db.execute(
                select(func.count())
                .select_from(UserDatabaseConnection)
                .where(UserDatabaseConnection.user_id == u.id)
            )
        ).scalar_one()
        teams = (
            await db.execute(
                select(func.count())
                .select_from(WorkspaceMember)
                .where(WorkspaceMember.user_id == u.id)
            )
        ).scalar_one()
        return AdminUserSummary(
            id=u.id,
            email=u.email,
            full_name=u.full_name,
            is_active=bool(u.is_active),
            is_verified=bool(u.is_verified),
            auth_provider=u.auth_provider,
            created_at=u.created_at,
            chat_count=int(chats),
            connection_count=int(conns),
            workspace_count=int(teams),
        )

    async def _get_user(self, db: AsyncSession, user_id: str) -> User:
        user = await db.get(User, user_id)
        if user is None:
            raise SessionNotFoundError("User not found.")
        return user

    async def get_user(self, user_id: str) -> AdminUserDetail:
        async with self._session_factory() as db:
            user = await self._get_user(db, user_id)
            summary = await self._summary(db, user)
            session_rows = (
                await db.execute(
                    select(ChatSession, func.count(ChatMessage.id))
                    .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
                    .where(ChatSession.user_id == user_id)
                    .group_by(ChatSession.id)
                    .order_by(ChatSession.updated_at.desc())
                    .limit(500)
                )
            ).all()
            conn_rows = (
                await db.execute(
                    select(UserDatabaseConnection, Workspace.name)
                    .outerjoin(Workspace, Workspace.id == UserDatabaseConnection.workspace_id)
                    .where(UserDatabaseConnection.user_id == user_id)
                    .order_by(UserDatabaseConnection.created_at)
                )
            ).all()
            ws_rows = (
                await db.execute(
                    select(Workspace, WorkspaceMember.role)
                    .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
                    .where(WorkspaceMember.user_id == user_id)
                    .order_by(Workspace.name)
                )
            ).all()
        return AdminUserDetail(
            user=summary,
            sessions=[
                AdminSessionSummary(s.id, s.title, int(n), s.created_at, s.updated_at)
                for s, n in session_rows
            ],
            connections=[
                {
                    "connection_id": c.connection_id,
                    "name": c.name,
                    "db_type": c.db_type,
                    "has_dsn": c.encrypted_url is not None,
                    "is_default": bool(c.is_default),
                    "workspace_name": ws_name,
                    "allowed_tables": c.allowed_tables,
                    "restricted_columns": list(c.restricted_columns or []),
                }
                for c, ws_name in conn_rows
            ],
            workspaces=[{"id": w.id, "name": w.name, "role": role} for w, role in ws_rows],
        )

    async def set_active(self, admin_id: str, user_id: str, active: bool) -> AdminUserSummary:
        """Deactivate (locks the account out immediately) or reactivate a user."""
        if admin_id == user_id and not active:
            raise WorkspaceValidationError("You can't deactivate your own account.")
        now = datetime.utcnow()
        async with self._session_factory() as db:
            user = await self._get_user(db, user_id)
            user.is_active = active
            user.updated_at = now
            if not active:
                # get_current_user already rejects inactive users on every
                # request; revoking sessions/refresh tokens also ends them cleanly.
                await db.execute(
                    update(UserLoginSession)
                    .where(
                        UserLoginSession.user_id == user_id, UserLoginSession.revoked_at.is_(None)
                    )
                    .values(revoked_at=now)
                )
                await db.execute(
                    update(RefreshToken)
                    .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
                    .values(revoked_at=now)
                )
            await db.commit()
            await db.refresh(user)
            summary = await self._summary(db, user)
        logger.info(
            "Admin changed account status", admin_id=admin_id, user_id=user_id, active=active
        )
        return summary

    # ── Chats ────────────────────────────────────────────────────────────────

    async def get_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        async with self._session_factory() as db:
            session = (
                await db.execute(
                    select(ChatSession).where(
                        ChatSession.id == session_id, ChatSession.user_id == user_id
                    )
                )
            ).scalar_one_or_none()
            if session is None:
                raise SessionNotFoundError("Chat not found.")
            messages = (
                (
                    await db.execute(
                        select(ChatMessage)
                        .where(ChatMessage.session_id == session_id)
                        .order_by(ChatMessage.timestamp)
                    )
                )
                .scalars()
                .all()
            )
        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": [
                {
                    "id": m.id,
                    "timestamp": m.timestamp,
                    "question": m.question,
                    "sql": m.sql,
                    "is_valid": m.is_valid,
                    "message": m.message,
                    "execution_error": m.execution_error,
                    "row_count": len(m.execution_result)
                    if isinstance(m.execution_result, list)
                    else None,
                    "execution_result": (m.execution_result or [])[:50]
                    if isinstance(m.execution_result, list)
                    else None,
                    "answer": (m.trust or {}).get("answer") if isinstance(m.trust, dict) else None,
                    "confidence": (m.trust or {}).get("confidence")
                    if isinstance(m.trust, dict)
                    else None,
                }
                for m in messages
            ],
        }

    async def delete_session(self, admin_id: str, user_id: str, session_id: str) -> None:
        async with self._session_factory() as db:
            session = (
                await db.execute(
                    select(ChatSession).where(
                        ChatSession.id == session_id, ChatSession.user_id == user_id
                    )
                )
            ).scalar_one_or_none()
            if session is None:
                raise SessionNotFoundError("Chat not found.")
            message_ids = select(ChatMessage.id).where(ChatMessage.session_id == session_id)
            await db.execute(delete(SqlVersion).where(SqlVersion.message_id.in_(message_ids)))
            await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
            await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
            await db.commit()
        logger.info(
            "Admin deleted a chat", admin_id=admin_id, user_id=user_id, session_id=session_id
        )
