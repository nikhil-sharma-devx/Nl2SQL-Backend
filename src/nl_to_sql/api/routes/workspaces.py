"""Workspace (team) routes — invitations, membership, roles, the workspace glossary.

Adding someone sends an invitation (``POST /{id}/invitations``); the invitee
sees it via ``GET /invitations`` (which drives the in-app notification) and
accepts or declines it. ``GET /directory`` lists who can be picked: verified
accounts on the caller's email domain, plus an exact-email match on any
domain; admins see everyone.

Every rule about who may do what lives in ``WorkspaceService``; these routes
only parse, delegate, and shape responses. Non-members get 404 for a
workspace (existence is never confirmed); members without the needed role
get 403 ``WORKSPACE_FORBIDDEN``.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from nl_to_sql.api.dependencies import (
    get_current_user,
    get_session_service,
    get_workspace_service,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import GlossaryEntry
from nl_to_sql.services.chat_session_service import ChatSessionService
from nl_to_sql.services.workspace_service import (
    DirectoryUser,
    InvitationInfo,
    MemberInfo,
    WorkspaceDetail,
    WorkspaceService,
    WorkspaceSummary,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/workspaces", tags=["Workspaces"])

RoleField = Field(pattern="^(admin|editor|member)$")


# ── Models ───────────────────────────────────────────────────────────────────


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceRename(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceOut(BaseModel):
    id: str
    name: str
    role: str
    member_count: int
    shared_connection_count: int
    created_at: datetime


class MemberOut(BaseModel):
    user_id: str
    email: str
    full_name: str | None
    role: str
    added_at: datetime


class SharedConnectionOut(BaseModel):
    connection_id: str
    name: str
    db_type: str
    owner_user_id: str
    owner_email: str


class InvitationOut(BaseModel):
    id: str
    workspace_id: str
    workspace_name: str
    role: str
    invitee_user_id: str
    invitee_email: str
    invitee_name: str | None
    invited_by_email: str | None
    created_at: datetime


class WorkspaceDetailOut(BaseModel):
    id: str
    name: str
    role: str
    created_at: datetime
    members: list[MemberOut]
    connections: list[SharedConnectionOut]
    pending_invitations: list[InvitationOut] = Field(
        default_factory=list, description="Visible to admins only"
    )


class InvitationCreate(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    role: str = RoleField
    model_config = {
        "json_schema_extra": {"examples": [{"email": "ana@company.com", "role": "member"}]}
    }


class DirectoryUserOut(BaseModel):
    user_id: str
    email: str
    full_name: str | None
    status: str | None = Field(
        default=None, description="'member' or 'invited' for the given workspace"
    )


class MemberRoleUpdate(BaseModel):
    role: str = RoleField


class WorkspaceGlossaryIn(BaseModel):
    term: str = Field(min_length=1, max_length=200)
    definition: str = Field(min_length=1, max_length=4000)


class WorkspaceGlossaryPatch(BaseModel):
    term: str | None = Field(default=None, min_length=1, max_length=200)
    definition: str | None = Field(default=None, min_length=1, max_length=4000)


class WorkspaceGlossaryOut(BaseModel):
    id: int
    term: str
    definition: str
    author_user_id: str
    updated_at: datetime


def _summary_out(w: WorkspaceSummary) -> WorkspaceOut:
    return WorkspaceOut(
        id=w.id,
        name=w.name,
        role=w.role,
        member_count=w.member_count,
        shared_connection_count=w.shared_connection_count,
        created_at=w.created_at,
    )


def _member_out(m: MemberInfo) -> MemberOut:
    return MemberOut(
        user_id=m.user_id, email=m.email, full_name=m.full_name, role=m.role, added_at=m.added_at
    )


def _invitation_out(i: InvitationInfo) -> InvitationOut:
    return InvitationOut(
        id=i.id,
        workspace_id=i.workspace_id,
        workspace_name=i.workspace_name,
        role=i.role,
        invitee_user_id=i.invitee_user_id,
        invitee_email=i.invitee_email,
        invitee_name=i.invitee_name,
        invited_by_email=i.invited_by_email,
        created_at=i.created_at,
    )


def _detail_out(d: WorkspaceDetail) -> WorkspaceDetailOut:
    return WorkspaceDetailOut(
        id=d.id,
        name=d.name,
        role=d.role,
        created_at=d.created_at,
        members=[_member_out(m) for m in d.members],
        connections=[
            SharedConnectionOut(
                connection_id=c.connection_id,
                name=c.name,
                db_type=c.db_type,
                owner_user_id=c.owner_user_id,
                owner_email=c.owner_email,
            )
            for c in d.connections
        ],
        pending_invitations=[_invitation_out(i) for i in d.pending_invitations],
    )


def _glossary_out(e: GlossaryEntry) -> WorkspaceGlossaryOut:
    return WorkspaceGlossaryOut(
        id=e.id,
        term=e.term,
        definition=e.definition,
        author_user_id=e.user_id,
        updated_at=e.updated_at,
    )


# ── Workspaces ───────────────────────────────────────────────────────────────


@router.get("", response_model=list[WorkspaceOut], summary="List workspaces you belong to")
async def list_workspaces(
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> list[WorkspaceOut]:
    return [_summary_out(w) for w in await svc.list_for_user(current_user.id)]


@router.post(
    "",
    response_model=WorkspaceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a workspace",
)
@limiter.limit("10/minute")
async def create_workspace(
    request: Request,
    body: WorkspaceCreate,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceOut:
    return _summary_out(await svc.create(current_user.id, body.name))


@router.get(
    "/invitations",
    response_model=list[InvitationOut],
    summary="Your pending invitations (drives the in-app notification)",
)
async def my_invitations(
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> list[InvitationOut]:
    return [_invitation_out(i) for i in await svc.my_invitations(current_user.id)]


@router.post(
    "/invitations/{invitation_id}/accept",
    response_model=WorkspaceOut,
    summary="Accept an invitation and join the workspace",
)
@limiter.limit("30/minute")
async def accept_invitation(
    request: Request,
    invitation_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceOut:
    workspace_id = await svc.respond(current_user.id, invitation_id, accept=True)
    for w in await svc.list_for_user(current_user.id):
        if w.id == workspace_id:
            return _summary_out(w)
    raise HTTPException(status_code=404, detail="Workspace not found.")


@router.post(
    "/invitations/{invitation_id}/decline",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Decline an invitation",
)
@limiter.limit("30/minute")
async def decline_invitation(
    request: Request,
    invitation_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> None:
    await svc.respond(current_user.id, invitation_id, accept=False)


@router.get(
    "/directory",
    response_model=list[DirectoryUserOut],
    summary="People you can invite",
    description=(
        "Verified, active accounts on your email domain (everyone on the same domain "
        "counts as one company), filtered by `q`. A `q` that is a full email address "
        "also matches that exact account on any domain. Admins see every account. "
        "Pass `workspace_id` to mark who is already a member or invited."
    ),
)
@limiter.limit("60/minute")
async def directory(
    request: Request,
    q: str = Query(default="", max_length=255),
    workspace_id: str | None = Query(default=None, max_length=36),
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> list[DirectoryUserOut]:
    users: list[DirectoryUser] = await svc.directory(
        current_user.id,
        current_user.email,
        q,
        is_admin=current_user.is_admin,
        workspace_id=workspace_id,
    )
    return [
        DirectoryUserOut(user_id=u.user_id, email=u.email, full_name=u.full_name, status=u.status)
        for u in users
    ]


@router.get(
    "/{workspace_id}", response_model=WorkspaceDetailOut, summary="Members and shared connections"
)
async def get_workspace(
    workspace_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceDetailOut:
    return _detail_out(await svc.get(current_user.id, workspace_id))


@router.patch(
    "/{workspace_id}", response_model=WorkspaceDetailOut, summary="Rename a workspace (admin)"
)
@limiter.limit("20/minute")
async def rename_workspace(
    request: Request,
    workspace_id: str,
    body: WorkspaceRename,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceDetailOut:
    await svc.rename(current_user.id, workspace_id, body.name)
    return _detail_out(await svc.get(current_user.id, workspace_id))


@router.delete(
    "/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a workspace (admin)"
)
@limiter.limit("10/minute")
async def delete_workspace(
    request: Request,
    workspace_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> None:
    await svc.delete(current_user.id, workspace_id)


# ── Members ──────────────────────────────────────────────────────────────────


@router.post(
    "/{workspace_id}/invitations",
    response_model=InvitationOut,
    status_code=status.HTTP_201_CREATED,
    summary="Invite a verified Vectrix account by email (admin)",
    description="The person joins only after accepting the invitation in the app.",
)
@limiter.limit("30/minute")
async def invite_member(
    request: Request,
    workspace_id: str,
    body: InvitationCreate,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> InvitationOut:
    return _invitation_out(await svc.invite(current_user.id, workspace_id, body.email, body.role))


@router.delete(
    "/{workspace_id}/invitations/{invitation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a pending invitation (admin)",
)
@limiter.limit("30/minute")
async def cancel_invitation(
    request: Request,
    workspace_id: str,
    invitation_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> None:
    await svc.cancel_invitation(current_user.id, workspace_id, invitation_id)


@router.patch(
    "/{workspace_id}/members/{user_id}",
    response_model=MemberOut,
    summary="Change a member's role (admin)",
)
@limiter.limit("30/minute")
async def change_member_role(
    request: Request,
    workspace_id: str,
    user_id: str,
    body: MemberRoleUpdate,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> MemberOut:
    await svc.change_role(current_user.id, workspace_id, user_id, body.role)
    detail = await svc.get(current_user.id, workspace_id)
    for m in detail.members:
        if m.user_id == user_id:
            return _member_out(m)
    raise HTTPException(status_code=404, detail="Member not found.")


@router.delete(
    "/{workspace_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a member (admin), or leave (yourself)",
)
@limiter.limit("30/minute")
async def remove_member(
    request: Request,
    workspace_id: str,
    user_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
) -> None:
    await svc.remove_member(current_user.id, workspace_id, user_id)


# ── Workspace glossary (read: member, write: editor) ────────────────────────


@router.get(
    "/{workspace_id}/glossary",
    response_model=list[WorkspaceGlossaryOut],
    summary="Workspace glossary terms (members)",
)
async def list_workspace_glossary(
    workspace_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
    session_service: ChatSessionService = Depends(get_session_service),
) -> list[WorkspaceGlossaryOut]:
    await svc.require_role(current_user.id, workspace_id, "member")
    async with session_service._session_factory() as db:
        rows = (
            (
                await db.execute(
                    select(GlossaryEntry)
                    .where(GlossaryEntry.workspace_id == workspace_id)
                    .order_by(GlossaryEntry.term.asc())
                    .limit(500)
                )
            )
            .scalars()
            .all()
        )
    return [_glossary_out(e) for e in rows]


@router.post(
    "/{workspace_id}/glossary",
    response_model=WorkspaceGlossaryOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add a workspace glossary term (editor)",
)
@limiter.limit("60/minute")
async def create_workspace_glossary_entry(
    request: Request,
    workspace_id: str,
    body: WorkspaceGlossaryIn,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
    session_service: ChatSessionService = Depends(get_session_service),
) -> WorkspaceGlossaryOut:
    await svc.require_role(current_user.id, workspace_id, "editor")
    async with session_service._session_factory() as db:
        existing = (
            await db.execute(
                select(GlossaryEntry.id).where(
                    GlossaryEntry.workspace_id == workspace_id,
                    GlossaryEntry.term.ilike(body.term.strip()),
                )
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=409, detail=f"'{body.term}' is already defined in this workspace."
            )
        entry = GlossaryEntry(
            user_id=current_user.id,
            workspace_id=workspace_id,
            term=body.term.strip(),
            definition=body.definition.strip(),
        )
        db.add(entry)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            # uq_user_glossary_term spans (user_id, term): the author already
            # has a personal term with this name.
            raise HTTPException(
                status_code=409,
                detail=f"You already have a personal term '{body.term}'. Rename or delete it first.",
            ) from None
        await db.refresh(entry)
    logger.info("Workspace glossary term added", workspace_id=workspace_id, user_id=current_user.id)
    return _glossary_out(entry)


async def _get_workspace_entry(db: AsyncSession, workspace_id: str, entry_id: int) -> GlossaryEntry:
    entry = (
        await db.execute(
            select(GlossaryEntry).where(
                GlossaryEntry.id == entry_id, GlossaryEntry.workspace_id == workspace_id
            )
        )
    ).scalar_one_or_none()
    if entry is None:
        raise HTTPException(status_code=404, detail="Glossary term not found.")
    return entry


@router.patch(
    "/{workspace_id}/glossary/{entry_id}",
    response_model=WorkspaceGlossaryOut,
    summary="Edit a workspace glossary term (editor)",
)
@limiter.limit("60/minute")
async def update_workspace_glossary_entry(
    request: Request,
    workspace_id: str,
    entry_id: int,
    body: WorkspaceGlossaryPatch,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
    session_service: ChatSessionService = Depends(get_session_service),
) -> WorkspaceGlossaryOut:
    await svc.require_role(current_user.id, workspace_id, "editor")
    async with session_service._session_factory() as db:
        entry = await _get_workspace_entry(db, workspace_id, entry_id)
        if body.term is not None:
            entry.term = body.term.strip()
        if body.definition is not None:
            entry.definition = body.definition.strip()
        entry.updated_at = datetime.utcnow()
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=409, detail="A term with that name already exists."
            ) from None
        await db.refresh(entry)
    return _glossary_out(entry)


@router.delete(
    "/{workspace_id}/glossary/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a workspace glossary term (editor)",
)
@limiter.limit("60/minute")
async def delete_workspace_glossary_entry(
    request: Request,
    workspace_id: str,
    entry_id: int,
    current_user: UserPublic = Depends(get_current_user),
    svc: WorkspaceService = Depends(get_workspace_service),
    session_service: ChatSessionService = Depends(get_session_service),
) -> None:
    await svc.require_role(current_user.id, workspace_id, "editor")
    async with session_service._session_factory() as db:
        entry = await _get_workspace_entry(db, workspace_id, entry_id)
        await db.delete(entry)
        await db.commit()
