"""Platform admin routes — every account, every chat, full control.

Every route requires ``require_admin`` (``users.is_admin``) except the
one-time setup pair (``/setup``), which any signed-in user may call but which
only succeeds with ``ADMIN_SETUP_TOKEN`` while no owner exists. Admins can
list and search users, open any user's chats / connections / teams, delete
chats, connections and workspaces, deactivate or reactivate accounts, and
grant admin. Revoking admin is owner-only (``require_owner``). Connection
secrets are never returned — only names and types.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    auth_cache_invalidate_user,
    get_account_reactivation_service,
    get_admin_service,
    get_connection_service,
    get_current_user,
    get_workspace_service,
    require_admin,
    require_owner,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.account_reactivation_service import (
    AccountReactivationService,
    ReactivationRequestView,
)
from nl_to_sql.services.admin_service import AdminAccount, AdminService, AdminUserSummary
from nl_to_sql.services.connection_service import ConnectionService
from nl_to_sql.services.workspace_service import WorkspaceService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


class AdminUserOut(BaseModel):
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


class AdminUserList(BaseModel):
    items: list[AdminUserOut]
    total: int


class AdminSessionOut(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class AdminUserDetailOut(BaseModel):
    user: AdminUserOut
    sessions: list[AdminSessionOut]
    connections: list[dict[str, Any]]
    workspaces: list[dict[str, Any]]


class AdminChatMessageOut(BaseModel):
    id: int
    timestamp: datetime
    question: str
    sql: str | None
    is_valid: bool | None
    message: str | None
    execution_error: str | None
    row_count: int | None
    execution_result: list[dict[str, Any]] | None
    answer: str | None
    confidence: str | None


class AdminChatOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[AdminChatMessageOut]


class AdminUserStatusUpdate(BaseModel):
    is_active: bool = Field(description="false locks the account out immediately")


class AdminSetupStatus(BaseModel):
    needs_setup: bool = Field(description="true while no owner exists")
    enabled: bool = Field(description="false when the server has no ADMIN_SETUP_TOKEN")


class AdminSetupRequest(BaseModel):
    token: str = Field(min_length=1, max_length=256)


class AdminAccountOut(BaseModel):
    id: str
    email: str
    full_name: str | None
    is_owner: bool
    is_active: bool


class AdminAccountList(BaseModel):
    items: list[AdminAccountOut]


class AdminGrantRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=36)


def _user_out(u: AdminUserSummary) -> AdminUserOut:
    return AdminUserOut(**u.__dict__)


def _account_out(a: AdminAccount) -> AdminAccountOut:
    return AdminAccountOut(**a.__dict__)


# ── One-time setup (any signed-in user; token-gated) ─────────────────────────


@router.get("/setup", response_model=AdminSetupStatus, summary="Whether owner setup is pending")
async def get_setup_status(
    _user: UserPublic = Depends(get_current_user),
    svc: AdminService = Depends(get_admin_service),
) -> AdminSetupStatus:
    return AdminSetupStatus(needs_setup=await svc.needs_setup(), enabled=svc.setup_enabled)


@router.post(
    "/setup",
    response_model=AdminAccountOut,
    summary="Claim the owner account with the server's setup token (once)",
)
@limiter.limit("5/minute")
async def claim_owner(
    request: Request,
    body: AdminSetupRequest,
    current_user: UserPublic = Depends(get_current_user),
    svc: AdminService = Depends(get_admin_service),
) -> AdminAccountOut:
    account = await svc.claim_owner(current_user.id, body.token)
    auth_cache_invalidate_user(current_user.id)
    return _account_out(account)


# ── Admin accounts ───────────────────────────────────────────────────────────


@router.get("/admins", response_model=AdminAccountList, summary="List every admin")
async def list_admins(
    _admin: UserPublic = Depends(require_admin),
    svc: AdminService = Depends(get_admin_service),
) -> AdminAccountList:
    return AdminAccountList(items=[_account_out(a) for a in await svc.list_admins()])


@router.post(
    "/admins",
    response_model=AdminAccountOut,
    status_code=status.HTTP_201_CREATED,
    summary="Make an account an admin (any admin)",
)
@limiter.limit("30/minute")
async def grant_admin(
    request: Request,
    body: AdminGrantRequest,
    admin: UserPublic = Depends(require_admin),
    svc: AdminService = Depends(get_admin_service),
) -> AdminAccountOut:
    account = await svc.grant_admin(admin.id, body.user_id)
    auth_cache_invalidate_user(body.user_id)
    return _account_out(account)


@router.delete(
    "/admins/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove an account's admin access (owner only)",
)
@limiter.limit("30/minute")
async def revoke_admin(
    request: Request,
    user_id: str,
    owner: UserPublic = Depends(require_owner),
    svc: AdminService = Depends(get_admin_service),
) -> None:
    await svc.revoke_admin(owner.id, user_id)
    auth_cache_invalidate_user(user_id)


# ── Reactivation requests ───────────────────────────────────────────────────


class ReactivationRequestOut(BaseModel):
    id: str
    user_id: str
    email: str
    full_name: str | None
    message: str | None
    status: Literal["pending", "approved", "rejected"]
    created_at: datetime
    decided_at: datetime | None
    decided_by_email: str | None


class ReactivationRequestList(BaseModel):
    items: list[ReactivationRequestOut]
    total: int
    pending: int = Field(description="pending count regardless of the status filter")


def _request_out(r: ReactivationRequestView) -> ReactivationRequestOut:
    return ReactivationRequestOut(**r.__dict__)


@router.get(
    "/reactivation-requests",
    response_model=ReactivationRequestList,
    summary="Deactivated accounts asking to be reactivated",
)
async def list_reactivation_requests(
    status_filter: Literal["pending", "approved", "rejected", "all"] = Query(
        default="pending", alias="status"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _admin: UserPublic = Depends(require_admin),
    svc: AccountReactivationService = Depends(get_account_reactivation_service),
) -> ReactivationRequestList:
    items, total = await svc.list_requests(
        None if status_filter == "all" else status_filter, limit, offset
    )
    return ReactivationRequestList(
        items=[_request_out(r) for r in items], total=total, pending=await svc.count_pending()
    )


@router.post(
    "/reactivation-requests/{request_id}/approve",
    response_model=ReactivationRequestOut,
    summary="Approve: reactivate the account",
)
@limiter.limit("60/minute")
async def approve_reactivation(
    request: Request,
    request_id: str,
    admin: UserPublic = Depends(require_admin),
    svc: AccountReactivationService = Depends(get_account_reactivation_service),
) -> ReactivationRequestOut:
    view = await svc.decide(admin.id, request_id, approve=True)
    auth_cache_invalidate_user(view.user_id)
    return _request_out(view)


@router.post(
    "/reactivation-requests/{request_id}/reject",
    response_model=ReactivationRequestOut,
    summary="Reject: the account stays deactivated",
)
@limiter.limit("60/minute")
async def reject_reactivation(
    request: Request,
    request_id: str,
    admin: UserPublic = Depends(require_admin),
    svc: AccountReactivationService = Depends(get_account_reactivation_service),
) -> ReactivationRequestOut:
    return _request_out(await svc.decide(admin.id, request_id, approve=False))


# ── Users ────────────────────────────────────────────────────────────────────


@router.get("/users", response_model=AdminUserList, summary="List and search every account")
async def list_users(
    q: str = Query(default="", max_length=255),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _admin: UserPublic = Depends(require_admin),
    svc: AdminService = Depends(get_admin_service),
) -> AdminUserList:
    items, total = await svc.list_users(q, limit, offset)
    return AdminUserList(items=[_user_out(u) for u in items], total=total)


@router.get(
    "/users/{user_id}",
    response_model=AdminUserDetailOut,
    summary="A user's chats, connections and teams",
)
async def get_user(
    user_id: str,
    _admin: UserPublic = Depends(require_admin),
    svc: AdminService = Depends(get_admin_service),
) -> AdminUserDetailOut:
    d = await svc.get_user(user_id)
    return AdminUserDetailOut(
        user=_user_out(d.user),
        sessions=[AdminSessionOut(**s.__dict__) for s in d.sessions],
        connections=d.connections,
        workspaces=d.workspaces,
    )


@router.patch(
    "/users/{user_id}", response_model=AdminUserOut, summary="Deactivate or reactivate an account"
)
@limiter.limit("30/minute")
async def set_user_status(
    request: Request,
    user_id: str,
    body: AdminUserStatusUpdate,
    admin: UserPublic = Depends(require_admin),
    svc: AdminService = Depends(get_admin_service),
    reactivation: AccountReactivationService = Depends(get_account_reactivation_service),
) -> AdminUserOut:
    summary = await svc.set_active(admin.id, user_id, body.is_active, actor_is_owner=admin.is_owner)
    auth_cache_invalidate_user(user_id)
    if body.is_active:
        await reactivation.close_pending_on_reactivate(admin.id, user_id)
    return _user_out(summary)


@router.get(
    "/users/{user_id}/chats/{session_id}",
    response_model=AdminChatOut,
    summary="Read any user's chat",
)
async def get_user_chat(
    user_id: str,
    session_id: str,
    _admin: UserPublic = Depends(require_admin),
    svc: AdminService = Depends(get_admin_service),
) -> AdminChatOut:
    return AdminChatOut(**await svc.get_session(user_id, session_id))


@router.delete(
    "/users/{user_id}/chats/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete any user's chat",
)
@limiter.limit("60/minute")
async def delete_user_chat(
    request: Request,
    user_id: str,
    session_id: str,
    admin: UserPublic = Depends(require_admin),
    svc: AdminService = Depends(get_admin_service),
) -> None:
    await svc.delete_session(admin.id, user_id, session_id)


@router.delete(
    "/users/{user_id}/connections/{connection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete any user's database connection",
)
@limiter.limit("30/minute")
async def delete_user_connection(
    request: Request,
    user_id: str,
    connection_id: str,
    admin: UserPublic = Depends(require_admin),
    conns: ConnectionService = Depends(get_connection_service),
) -> None:
    # Same path the owner's own delete uses: ownership-scoped delete, then
    # best-effort cleanup of the derived catalog and vectors.
    await conns.delete(user_id, connection_id)
    try:
        from nl_to_sql.api.dependencies import _get_container

        container = _get_container()
        await container.schema_catalog_service().delete_catalog(connection_id)
        vector_store = container.vector_store()
        if hasattr(vector_store, "delete_by_connection"):
            await vector_store.delete_by_connection(connection_id)
    except Exception as exc:
        logger.warning("Post-delete cleanup failed", connection_id=connection_id, error=str(exc))
    logger.info(
        "Admin deleted a connection",
        admin_id=admin.id,
        user_id=user_id,
        connection_id=connection_id,
    )


@router.delete(
    "/workspaces/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete any workspace",
)
@limiter.limit("30/minute")
async def delete_workspace(
    request: Request,
    workspace_id: str,
    admin: UserPublic = Depends(require_admin),
    workspaces: WorkspaceService = Depends(get_workspace_service),
) -> None:
    await workspaces.force_delete(admin.id, workspace_id)
