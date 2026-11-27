"""Platform admin routes — every account, every chat, full control.

Every route requires ``require_admin`` (emails listed in ``ADMIN_EMAILS``).
Admins can list and search users, open any user's chats / connections /
teams, delete chats, connections and workspaces, and deactivate or reactivate
accounts. Connection secrets are never returned — only names and types.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    get_admin_service,
    get_connection_service,
    get_workspace_service,
    require_admin,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.admin_service import AdminService, AdminUserSummary
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


def _user_out(u: AdminUserSummary) -> AdminUserOut:
    return AdminUserOut(**u.__dict__)


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
) -> AdminUserOut:
    return _user_out(await svc.set_active(admin.id, user_id, body.is_active))


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
