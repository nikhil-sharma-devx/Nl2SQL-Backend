"""WorkspaceService — teams that share database connections and their knowledge.

Design (a deliberately smaller rebuild of the Phase-6a feature removed in
migration 0017, aimed at the reasons it failed):

* **Members are existing users, linked by id.** An admin invites someone by
  the email of their Vectrix account (or picks them from the directory); the
  lookup resolves it to ``users.id`` at that moment, so the invitation can't
  get lost. The invitee sees it in the app and accepts or declines; only
  acceptance creates the membership row (``workspace_invitations`` is a
  separate table, so no access check ever has to filter out "pending").
* **Directory scope:** a user sees verified accounts on their own email
  domain (everyone @gmail.com counts as one company) and can reach anyone else
  only by typing their exact email; admins see everyone.
* **Three fixed roles, checked in exactly one place** (:meth:`require_role`):
  ``admin`` > ``editor`` > ``member``. Every route calls through this service;
  there is no second permission matrix to drift out of sync.
* **The last admin can't be removed or demoted**, so a workspace can never be
  left unmanageable.
* **Non-members get "not found"**, never "forbidden", so workspace existence
  can't be probed. Members without the needed role get a permission error.

Only connection-level knowledge is shared (connections, their certified
metrics, workspace glossary terms). Chats, saved queries, dashboards and
schedules stay personal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, cast

import structlog
from sqlalchemy import ColumnElement, CompoundSelect, Select, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.core.exceptions import (
    ConnectionNotFoundError,
    WorkspaceConflictError,
    WorkspaceNotFoundError,
    WorkspacePermissionError,
    WorkspaceValidationError,
)
from nl_to_sql.infrastructure.database.models import (
    GlossaryEntry,
    User,
    UserDatabaseConnection,
    Workspace,
    WorkspaceInvitation,
    WorkspaceMember,
)

logger = structlog.get_logger(__name__)

Role = Literal["admin", "editor", "member"]
ROLES: tuple[Role, ...] = ("member", "editor", "admin")
_RANK: dict[str, int] = {r: i for i, r in enumerate(ROLES)}
_MAX_WORKSPACES_PER_USER = 20
_MAX_MEMBERS = 200


@dataclass
class WorkspaceSummary:
    id: str
    name: str
    role: Role
    member_count: int
    shared_connection_count: int
    created_at: datetime


@dataclass
class MemberInfo:
    user_id: str
    email: str
    full_name: str | None
    role: Role
    added_at: datetime


@dataclass
class SharedConnectionInfo:
    connection_id: str
    name: str
    db_type: str
    owner_user_id: str
    owner_email: str


@dataclass
class InvitationInfo:
    id: str
    workspace_id: str
    workspace_name: str
    role: Role
    invitee_user_id: str
    invitee_email: str
    invitee_name: str | None
    invited_by_email: str | None
    created_at: datetime


@dataclass
class DirectoryUser:
    user_id: str
    email: str
    full_name: str | None
    # Relation to the workspace passed to directory(), if any.
    status: Literal["member", "invited"] | None = None


@dataclass
class WorkspaceDetail:
    id: str
    name: str
    role: Role
    created_at: datetime
    members: list[MemberInfo]
    connections: list[SharedConnectionInfo]
    pending_invitations: list[InvitationInfo] = field(default_factory=list)


def email_domain(email: str) -> str:
    return email.strip().lower().rpartition("@")[2]


def _check_role(role: str) -> Role:
    if role not in _RANK:
        raise WorkspaceValidationError(f"Unknown role '{role}'. Use admin, editor or member.")
    return cast(Role, role)


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards so a query is matched literally."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _clean_name(name: str) -> str:
    cleaned = " ".join(name.split())
    if not cleaned:
        raise WorkspaceValidationError("Workspace name cannot be empty.")
    if len(cleaned) > 120:
        raise WorkspaceValidationError("Workspace name must be 120 characters or fewer.")
    return cleaned


class WorkspaceService:
    """Workspace membership, roles, and connection sharing.

    SOLID:
      S — owns team membership + the role rules; nothing else.
      D — session factory is constructor-injected (mirrors MetricsService).
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ── Role checks (the single source of truth) ─────────────────────────────

    @staticmethod
    async def _role(db: AsyncSession, user_id: str, workspace_id: str) -> Role | None:
        role = (
            await db.execute(
                select(WorkspaceMember.role).where(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == user_id,
                )
            )
        ).scalar_one_or_none()
        return cast(Role, role) if role in _RANK else None

    async def _require(
        self, db: AsyncSession, user_id: str, workspace_id: str, minimum: Role
    ) -> Role:
        role = await self._role(db, user_id, workspace_id)
        if role is None:
            raise WorkspaceNotFoundError("Workspace not found.")
        if _RANK[role] < _RANK[minimum]:
            raise WorkspacePermissionError(
                f"This needs the {minimum} role in this workspace; you are a {role}."
            )
        return role

    async def role_of(self, user_id: str, workspace_id: str) -> Role | None:
        async with self._session_factory() as db:
            return await self._role(db, user_id, workspace_id)

    async def require_role(self, user_id: str, workspace_id: str, minimum: Role) -> Role:
        """Raise unless ``user_id`` holds at least ``minimum`` in ``workspace_id``."""
        async with self._session_factory() as db:
            return await self._require(db, user_id, workspace_id, minimum)

    # ── Workspaces ───────────────────────────────────────────────────────────

    async def create(self, user_id: str, name: str) -> WorkspaceSummary:
        cleaned = _clean_name(name)
        async with self._session_factory() as db:
            owned = (
                await db.execute(
                    select(func.count())
                    .select_from(Workspace)
                    .where(Workspace.created_by == user_id)
                )
            ).scalar_one()
            if owned >= _MAX_WORKSPACES_PER_USER:
                raise WorkspaceValidationError(
                    f"You can create at most {_MAX_WORKSPACES_PER_USER} workspaces."
                )
            ws = Workspace(name=cleaned, created_by=user_id)
            db.add(ws)
            await db.flush()
            db.add(
                WorkspaceMember(workspace_id=ws.id, user_id=user_id, role="admin", added_by=user_id)
            )
            await db.commit()
            await db.refresh(ws)
        logger.info("Workspace created", workspace_id=ws.id, user_id=user_id)
        return WorkspaceSummary(ws.id, ws.name, "admin", 1, 0, ws.created_at)

    async def list_for_user(self, user_id: str) -> list[WorkspaceSummary]:
        async with self._session_factory() as db:
            rows = (
                await db.execute(
                    select(Workspace, WorkspaceMember.role)
                    .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
                    .where(WorkspaceMember.user_id == user_id)
                    .order_by(Workspace.name)
                )
            ).all()
            ids = [ws.id for ws, _ in rows]
            members: dict[str, int] = {}
            shared: dict[str, int] = {}
            if ids:
                member_rows = (
                    await db.execute(
                        select(WorkspaceMember.workspace_id, func.count())
                        .where(WorkspaceMember.workspace_id.in_(ids))
                        .group_by(WorkspaceMember.workspace_id)
                    )
                ).all()
                members = {ws_id: int(n) for ws_id, n in member_rows}
                shared_rows = (
                    await db.execute(
                        select(UserDatabaseConnection.workspace_id, func.count())
                        .where(UserDatabaseConnection.workspace_id.in_(ids))
                        .group_by(UserDatabaseConnection.workspace_id)
                    )
                ).all()
                shared = {ws_id: int(n) for ws_id, n in shared_rows if ws_id is not None}
        return [
            WorkspaceSummary(
                ws.id,
                ws.name,
                cast(Role, role),
                members.get(ws.id, 0),
                shared.get(ws.id, 0),
                ws.created_at,
            )
            for ws, role in rows
        ]

    async def get(self, user_id: str, workspace_id: str) -> WorkspaceDetail:
        async with self._session_factory() as db:
            role = await self._require(db, user_id, workspace_id, "member")
            ws = await db.get(Workspace, workspace_id)
            if ws is None:
                raise WorkspaceNotFoundError("Workspace not found.")
            member_rows = (
                await db.execute(
                    select(WorkspaceMember, User)
                    .join(User, User.id == WorkspaceMember.user_id)
                    .where(WorkspaceMember.workspace_id == workspace_id)
                    .order_by(User.email)
                )
            ).all()
            conn_rows = (
                await db.execute(
                    select(UserDatabaseConnection, User.email)
                    .join(User, User.id == UserDatabaseConnection.user_id)
                    .where(UserDatabaseConnection.workspace_id == workspace_id)
                    .order_by(UserDatabaseConnection.name)
                )
            ).all()
            pending = (
                await self._invitations(db, workspace_id=workspace_id) if role == "admin" else []
            )
        return WorkspaceDetail(
            id=ws.id,
            name=ws.name,
            role=role,
            created_at=ws.created_at,
            members=[
                MemberInfo(u.id, u.email, u.full_name, cast(Role, m.role), m.created_at)
                for m, u in member_rows
            ],
            connections=[
                SharedConnectionInfo(c.connection_id, c.name, c.db_type, c.user_id, email)
                for c, email in conn_rows
            ],
            pending_invitations=pending,
        )

    async def rename(self, user_id: str, workspace_id: str, name: str) -> None:
        cleaned = _clean_name(name)
        async with self._session_factory() as db:
            await self._require(db, user_id, workspace_id, "admin")
            await db.execute(
                update(Workspace)
                .where(Workspace.id == workspace_id)
                .values(name=cleaned, updated_at=datetime.utcnow())
            )
            await db.commit()

    async def delete(self, user_id: str, workspace_id: str) -> None:
        async with self._session_factory() as db:
            await self._require(db, user_id, workspace_id, "admin")
            await self._delete_rows(db, workspace_id)
            await db.commit()
        logger.info("Workspace deleted", workspace_id=workspace_id, user_id=user_id)

    async def force_delete(self, platform_admin_id: str, workspace_id: str) -> None:
        """Delete any workspace (platform admins only — the route enforces it)."""
        async with self._session_factory() as db:
            if await db.get(Workspace, workspace_id) is None:
                raise WorkspaceNotFoundError("Workspace not found.")
            await self._delete_rows(db, workspace_id)
            await db.commit()
        logger.info(
            "Workspace deleted by platform admin",
            workspace_id=workspace_id,
            admin_id=platform_admin_id,
        )

    @staticmethod
    async def _delete_rows(db: AsyncSession, workspace_id: str) -> None:
        # Explicit (not just FK cascade) so behaviour is identical on SQLite
        # dev databases, which can't add FKs via ALTER.
        await db.execute(
            update(UserDatabaseConnection)
            .where(UserDatabaseConnection.workspace_id == workspace_id)
            .values(workspace_id=None)
        )
        await db.execute(delete(GlossaryEntry).where(GlossaryEntry.workspace_id == workspace_id))
        await db.execute(
            delete(WorkspaceInvitation).where(WorkspaceInvitation.workspace_id == workspace_id)
        )
        await db.execute(
            delete(WorkspaceMember).where(WorkspaceMember.workspace_id == workspace_id)
        )
        await db.execute(delete(Workspace).where(Workspace.id == workspace_id))

    # ── Members ──────────────────────────────────────────────────────────────

    async def invite(
        self, admin_id: str, workspace_id: str, email: str, role: str = "member"
    ) -> InvitationInfo:
        """Invite an existing, verified Vectrix account to the workspace (admin).

        Creates a ``pending`` invitation; the person becomes a member only when
        they accept it. Any verified account can be invited by its exact
        email — the directory scope limits what is *browsable*, not who can be
        invited by someone who already knows the address.
        """
        checked = _check_role(role)
        normalised = email.strip().lower()
        if not normalised or "@" not in normalised:
            raise WorkspaceValidationError("Enter the email address of a Vectrix account.")
        async with self._session_factory() as db:
            await self._require(db, admin_id, workspace_id, "admin")
            user = (
                await db.execute(select(User).where(func.lower(User.email) == normalised))
            ).scalar_one_or_none()
            if user is None or not user.is_active or not user.is_verified:
                raise WorkspaceValidationError(
                    "No verified Vectrix account uses that email. Ask them to finish signing up first."
                )
            user_id, user_email, user_name = user.id, user.email, user.full_name
            if user_id == admin_id:
                raise WorkspaceValidationError("You're already in this workspace.")
            if await self._role(db, user_id, workspace_id) is not None:
                raise WorkspaceConflictError(f"{user_email} is already in this workspace.")
            already = (
                await db.execute(
                    select(WorkspaceInvitation.id).where(
                        WorkspaceInvitation.workspace_id == workspace_id,
                        WorkspaceInvitation.invitee_user_id == user_id,
                        WorkspaceInvitation.status == "pending",
                    )
                )
            ).first()
            if already:
                raise WorkspaceConflictError(f"{user_email} already has a pending invitation.")
            count = (
                await db.execute(
                    select(func.count())
                    .select_from(WorkspaceMember)
                    .where(WorkspaceMember.workspace_id == workspace_id)
                )
            ).scalar_one()
            if count >= _MAX_MEMBERS:
                raise WorkspaceValidationError(
                    f"A workspace can have at most {_MAX_MEMBERS} members."
                )
            inv = WorkspaceInvitation(
                workspace_id=workspace_id,
                invitee_user_id=user_id,
                invited_by=admin_id,
                role=checked,
                status="pending",
            )
            db.add(inv)
            await db.commit()
            await db.refresh(inv)
            ws_name = (
                await db.execute(select(Workspace.name).where(Workspace.id == workspace_id))
            ).scalar_one()
            inviter = (
                await db.execute(select(User.email).where(User.id == admin_id))
            ).scalar_one_or_none()
        logger.info(
            "Workspace invitation sent", workspace_id=workspace_id, invitee=user_id, role=checked
        )
        return InvitationInfo(
            inv.id,
            workspace_id,
            ws_name,
            checked,
            user_id,
            user_email,
            user_name,
            inviter,
            inv.created_at,
        )

    @staticmethod
    async def _invitations(
        db: AsyncSession, *, workspace_id: str | None = None, invitee_id: str | None = None
    ) -> list[InvitationInfo]:
        """Pending invitations for a workspace, or addressed to one user."""
        invitee = User.__table__.alias("invitee")
        inviter = User.__table__.alias("inviter")
        stmt = (
            select(
                WorkspaceInvitation,
                Workspace.name,
                invitee.c.email,
                invitee.c.full_name,
                inviter.c.email,
            )
            .join(Workspace, Workspace.id == WorkspaceInvitation.workspace_id)
            .join(invitee, invitee.c.id == WorkspaceInvitation.invitee_user_id)
            .outerjoin(inviter, inviter.c.id == WorkspaceInvitation.invited_by)
            .where(WorkspaceInvitation.status == "pending")
            .order_by(WorkspaceInvitation.created_at.desc())
        )
        if workspace_id is not None:
            stmt = stmt.where(WorkspaceInvitation.workspace_id == workspace_id)
        if invitee_id is not None:
            stmt = stmt.where(WorkspaceInvitation.invitee_user_id == invitee_id)
        rows = (await db.execute(stmt)).all()
        return [
            InvitationInfo(
                inv.id,
                inv.workspace_id,
                ws_name,
                cast(Role, inv.role),
                inv.invitee_user_id,
                inv_email,
                inv_name,
                by_email,
                inv.created_at,
            )
            for inv, ws_name, inv_email, inv_name, by_email in rows
        ]

    async def my_invitations(self, user_id: str) -> list[InvitationInfo]:
        """Pending invitations addressed to ``user_id`` (drives the notification)."""
        async with self._session_factory() as db:
            return await self._invitations(db, invitee_id=user_id)

    async def respond(self, user_id: str, invitation_id: str, accept: bool) -> str | None:
        """Accept or decline an invitation addressed to ``user_id``.

        Returns the workspace id on accept. Someone else's invitation, or one
        that is no longer pending, is "not found".
        """
        async with self._session_factory() as db:
            inv = (
                await db.execute(
                    select(WorkspaceInvitation).where(
                        WorkspaceInvitation.id == invitation_id,
                        WorkspaceInvitation.invitee_user_id == user_id,
                        WorkspaceInvitation.status == "pending",
                    )
                )
            ).scalar_one_or_none()
            if inv is None:
                raise WorkspaceNotFoundError("Invitation not found.")
            workspace_id = inv.workspace_id
            inv.status = "accepted" if accept else "declined"
            inv.responded_at = datetime.utcnow()
            if accept and await self._role(db, user_id, workspace_id) is None:
                db.add(
                    WorkspaceMember(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        role=_check_role(inv.role),
                        added_by=inv.invited_by,
                    )
                )
            await db.commit()
        logger.info(
            "Workspace invitation answered",
            workspace_id=workspace_id,
            user_id=user_id,
            accepted=accept,
        )
        return workspace_id if accept else None

    async def cancel_invitation(self, admin_id: str, workspace_id: str, invitation_id: str) -> None:
        async with self._session_factory() as db:
            await self._require(db, admin_id, workspace_id, "admin")
            inv = (
                await db.execute(
                    select(WorkspaceInvitation).where(
                        WorkspaceInvitation.id == invitation_id,
                        WorkspaceInvitation.workspace_id == workspace_id,
                        WorkspaceInvitation.status == "pending",
                    )
                )
            ).scalar_one_or_none()
            if inv is None:
                raise WorkspaceNotFoundError("Invitation not found.")
            inv.status = "cancelled"
            inv.responded_at = datetime.utcnow()
            await db.commit()

    # ── Directory (who can be invited) ───────────────────────────────────────

    async def directory(
        self,
        requester_id: str,
        requester_email: str,
        query: str = "",
        *,
        is_admin: bool = False,
        workspace_id: str | None = None,
        limit: int = 50,
    ) -> list[DirectoryUser]:
        """Verified, active accounts the requester may pick when inviting.

        Scope: same email domain as the requester (admins: everyone). A query
        filters by name or email within that scope; a query that is a full
        email address also matches that exact account on any domain.
        """
        q = " ".join(query.split()).lower()
        base = [User.is_active.is_(True), User.is_verified.is_(True), User.id != requester_id]
        scope: list[ColumnElement[bool]] = []
        if not is_admin:
            scope.append(
                func.lower(User.email).like(
                    f"%@{_escape_like(email_domain(requester_email))}", escape="\\"
                )
            )
        if q:
            pattern = f"%{_escape_like(q)}%"
            scope.append(
                func.lower(User.email).like(pattern, escape="\\")
                | func.lower(func.coalesce(User.full_name, "")).like(pattern, escape="\\")
            )
        conditions = [*base, *scope]
        async with self._session_factory() as db:
            scoped = select(User.id, User.email, User.full_name).where(*conditions)
            if q and "@" in q and "." in q.rpartition("@")[2]:
                # A full address also reaches that exact account on any domain.
                exact = select(User.id, User.email, User.full_name).where(
                    *base, func.lower(User.email) == q
                )
                stmt: (
                    Select[tuple[str, str, str | None]]
                    | CompoundSelect[tuple[str, str, str | None]]
                ) = scoped.union(exact)
            else:
                stmt = scoped
            rows = (await db.execute(stmt.order_by("email").limit(min(limit, 200)))).all()
            status: dict[str, Literal["member", "invited"]] = {}
            if workspace_id and rows:
                ids = [r[0] for r in rows]
                for uid in (
                    await db.execute(
                        select(WorkspaceMember.user_id).where(
                            WorkspaceMember.workspace_id == workspace_id,
                            WorkspaceMember.user_id.in_(ids),
                        )
                    )
                ).scalars():
                    status[uid] = "member"
                for uid in (
                    await db.execute(
                        select(WorkspaceInvitation.invitee_user_id).where(
                            WorkspaceInvitation.workspace_id == workspace_id,
                            WorkspaceInvitation.status == "pending",
                            WorkspaceInvitation.invitee_user_id.in_(ids),
                        )
                    )
                ).scalars():
                    status.setdefault(uid, "invited")
        return [DirectoryUser(uid, email, name, status.get(uid)) for uid, email, name in rows]

    async def _admin_count(self, db: AsyncSession, workspace_id: str) -> int:
        return (
            await db.execute(
                select(func.count())
                .select_from(WorkspaceMember)
                .where(
                    WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.role == "admin"
                )
            )
        ).scalar_one()

    async def _get_member(
        self, db: AsyncSession, workspace_id: str, user_id: str
    ) -> WorkspaceMember:
        member = (
            await db.execute(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id
                )
            )
        ).scalar_one_or_none()
        if member is None:
            raise WorkspaceValidationError("That person isn't a member of this workspace.")
        return member

    async def change_role(
        self, admin_id: str, workspace_id: str, member_user_id: str, role: str
    ) -> Role:
        checked = _check_role(role)
        async with self._session_factory() as db:
            await self._require(db, admin_id, workspace_id, "admin")
            member = await self._get_member(db, workspace_id, member_user_id)
            if (
                member.role == "admin"
                and checked != "admin"
                and await self._admin_count(db, workspace_id) <= 1
            ):
                raise WorkspaceValidationError(
                    "A workspace needs at least one admin. Make someone else an admin first."
                )
            member.role = checked
            member.updated_at = datetime.utcnow()
            await db.commit()
        logger.info(
            "Workspace role changed",
            workspace_id=workspace_id,
            user_id=member_user_id,
            role=checked,
        )
        return checked

    async def remove_member(self, actor_id: str, workspace_id: str, member_user_id: str) -> None:
        """Admins remove anyone; any member may remove themselves (leave)."""
        async with self._session_factory() as db:
            minimum: Role = "member" if actor_id == member_user_id else "admin"
            await self._require(db, actor_id, workspace_id, minimum)
            member = await self._get_member(db, workspace_id, member_user_id)
            if member.role == "admin" and await self._admin_count(db, workspace_id) <= 1:
                raise WorkspaceValidationError(
                    "The last admin can't leave. Make someone else an admin, or delete the workspace."
                )
            # Their own connections leave the workspace with them.
            await db.execute(
                update(UserDatabaseConnection)
                .where(
                    UserDatabaseConnection.workspace_id == workspace_id,
                    UserDatabaseConnection.user_id == member_user_id,
                )
                .values(workspace_id=None)
            )
            await db.delete(member)
            await db.commit()
        logger.info("Workspace member removed", workspace_id=workspace_id, user_id=member_user_id)

    # ── Connection sharing ───────────────────────────────────────────────────

    async def set_connection_workspace(
        self, user_id: str, connection_id: str, workspace_id: str | None
    ) -> None:
        """Share the caller's own connection into a workspace (admin) or unshare it."""
        async with self._session_factory() as db:
            row = (
                await db.execute(
                    select(UserDatabaseConnection).where(
                        UserDatabaseConnection.connection_id == connection_id,
                        UserDatabaseConnection.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise ConnectionNotFoundError("Connection not found.")
            if workspace_id is not None:
                await self._require(db, user_id, workspace_id, "admin")
                if row.encrypted_url is None:
                    raise WorkspaceValidationError(
                        "The built-in Server Default connection can't be shared. "
                        "Add your own database connection and share that."
                    )
            row.workspace_id = workspace_id
            row.updated_at = datetime.utcnow()
            await db.commit()
        logger.info(
            "Connection sharing changed",
            connection_id=connection_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
