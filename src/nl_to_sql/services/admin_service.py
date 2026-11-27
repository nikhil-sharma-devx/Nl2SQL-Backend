"""AdminService — platform-wide view and control for admin accounts.

Admins (``users.is_admin``) can list every account, open any user's chats,
connections and teams, delete chats / connections / workspaces, deactivate or
reactivate accounts, and grant admin to other accounts. Only the single owner
(``users.is_owner``) can revoke admin or deactivate another admin. Every route
that uses this service is behind ``require_admin`` (or ``require_owner``) —
except the one-time owner setup, which is gated by ``ADMIN_SETUP_TOKEN`` here.
Beyond the owner/admin rules below, this service does no permission checks,
so it must never be wired to a non-admin route.

This service is the only writer of ``users.is_admin`` / ``users.is_owner``.

Deliberately NOT here: reading decrypted connection strings or API keys —
admins see that a connection exists and its masked host, never the secret.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from pydantic import SecretStr
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.core.exceptions import (
    AdminConflictError,
    AdminPermissionError,
    SessionNotFoundError,
    WorkspaceValidationError,
)
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
    is_admin: bool
    is_owner: bool
    created_at: datetime
    chat_count: int
    connection_count: int
    workspace_count: int


@dataclass
class AdminAccount:
    """One platform admin, as listed on the Admins tab."""

    id: str
    email: str
    full_name: str | None
    is_owner: bool
    is_active: bool


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
      S — admin-only cross-user operations; route deps gate admin/owner, the
          owner-only invariants (who may revoke / deactivate an admin) live here.
      D — session factory and setup token constructor-injected.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        setup_token: SecretStr | str = "",
    ) -> None:
        self._session_factory = session_factory
        self._setup_token = (
            setup_token.get_secret_value() if isinstance(setup_token, SecretStr) else setup_token
        )

    # ── Admin accounts ───────────────────────────────────────────────────────

    @property
    def setup_enabled(self) -> bool:
        """Whether the server has an ADMIN_SETUP_TOKEN configured."""
        return bool(self._setup_token)

    async def needs_setup(self) -> bool:
        """True while no owner exists — the setup page is only usable then."""
        async with self._session_factory() as db:
            owner = (
                await db.execute(select(User.id).where(User.is_owner.is_(True)).limit(1))
            ).scalar_one_or_none()
        return owner is None

    async def claim_owner(self, user_id: str, token: str) -> AdminAccount:
        """One-time setup: make ``user_id`` the owner (and an admin).

        Allowed only while no owner exists and only with the server's
        ADMIN_SETUP_TOKEN. The single-owner unique index turns a race between
        two simultaneous claims into a conflict for the loser.
        """
        if not await self.needs_setup():
            raise AdminConflictError("Admin setup is already complete.")
        if not self._setup_token:
            raise AdminPermissionError(
                "Admin setup is disabled. Set ADMIN_SETUP_TOKEN on the server and restart it."
            )
        if not hmac.compare_digest(token.encode(), self._setup_token.encode()):
            logger.warning("Admin setup rejected: bad token", user_id=user_id)
            raise AdminPermissionError("That setup token is not correct.")
        async with self._session_factory() as db:
            user = await self._get_user(db, user_id)
            user.is_admin = True
            user.is_owner = True
            user.updated_at = datetime.utcnow()
            try:
                await db.commit()
            except IntegrityError as exc:
                await db.rollback()
                raise AdminConflictError("Admin setup is already complete.") from exc
            await db.refresh(user)
            account = self._account(user)
        logger.info("Admin setup complete: owner claimed", user_id=user_id)
        return account

    @staticmethod
    def _account(u: User) -> AdminAccount:
        return AdminAccount(
            id=u.id,
            email=u.email,
            full_name=u.full_name,
            is_owner=bool(u.is_owner),
            is_active=bool(u.is_active),
        )

    async def list_admins(self) -> list[AdminAccount]:
        """Every admin, owner first then by email."""
        async with self._session_factory() as db:
            rows = (
                (
                    await db.execute(
                        select(User)
                        .where(User.is_admin.is_(True))
                        .order_by(User.is_owner.desc(), User.email)
                    )
                )
                .scalars()
                .all()
            )
        return [self._account(u) for u in rows]

    async def grant_admin(self, admin_id: str, user_id: str) -> AdminAccount:
        """Any admin may make another active, verified account an admin."""
        async with self._session_factory() as db:
            user = await self._get_user(db, user_id)
            if user.is_admin:
                raise AdminConflictError(f"{user.email} is already an admin.")
            if not user.is_active:
                raise WorkspaceValidationError("Reactivate this account before making it an admin.")
            if not user.is_verified:
                raise WorkspaceValidationError(
                    "This account hasn't verified its email yet, so it can't be an admin."
                )
            user.is_admin = True
            user.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(user)
            account = self._account(user)
        logger.info("Admin granted", admin_id=admin_id, user_id=user_id)
        return account

    async def revoke_admin(self, owner_id: str, user_id: str) -> None:
        """Owner-only (enforced by ``require_owner`` on the route): remove admin."""
        async with self._session_factory() as db:
            user = await self._get_user(db, user_id)
            if user.is_owner:
                raise AdminConflictError("The owner's admin access can't be removed.")
            if not user.is_admin:
                raise AdminConflictError(f"{user.email} is not an admin.")
            user.is_admin = False
            user.updated_at = datetime.utcnow()
            await db.commit()
        logger.info("Admin revoked", owner_id=owner_id, user_id=user_id)

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
            is_admin=bool(u.is_admin),
            is_owner=bool(u.is_owner),
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

    async def set_active(
        self, admin_id: str, user_id: str, active: bool, *, actor_is_owner: bool = False
    ) -> AdminUserSummary:
        """Deactivate (locks the account out immediately) or reactivate a user.

        Deactivating an admin would take their admin access away, so — like
        revoking admin — only the owner may do it.
        """
        if admin_id == user_id and not active:
            raise WorkspaceValidationError("You can't deactivate your own account.")
        now = datetime.utcnow()
        async with self._session_factory() as db:
            user = await self._get_user(db, user_id)
            if not active and user.is_admin and not actor_is_owner:
                raise AdminPermissionError("Only the owner can deactivate an admin.")
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
