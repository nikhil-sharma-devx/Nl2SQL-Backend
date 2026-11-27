"""Connection management routes — per-user multiple database connections (BYOD).

Every endpoint is authenticated and scoped to the current user; a connection
the user can't access is reported as *not found* (404) so cross-user existence
cannot be probed. Responses never contain a decrypted DSN — only a masked
``url_preview``, and only for the connection's owner.

Connections shared into a workspace the user belongs to are listed alongside
their own (``access`` = their workspace role) and can be selected as the
active connection. Only the owner can edit, delete, test, re-share or change
the access policy of a connection.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    get_connection_service,
    get_current_user,
    get_workspace_service,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.connection_service import ConnectionInfo, ConnectionService
from nl_to_sql.services.workspace_service import WorkspaceService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/connections", tags=["Connections"])


# ── Models ───────────────────────────────────────────────────────────────────


class ConnectionOut(BaseModel):
    connection_id: str
    name: str
    db_type: str
    is_default: bool
    has_dsn: bool
    url_preview: str | None = None
    created_at: datetime
    updated_at: datetime
    access: str = Field(default="owner", description="owner, or your role in the sharing workspace")
    workspace_id: str | None = None
    workspace_name: str | None = None
    owner_email: str | None = Field(default=None, description="Set only for connections shared with you")
    allowed_tables: list[str] | None = Field(
        default=None, description="Tables the AI may use (null = every table)"
    )
    restricted_columns: list[str] = Field(
        default_factory=list, description="table.column entries whose values never leave the database"
    )


class ConnectionPolicyUpdate(BaseModel):
    allowed_tables: list[str] | None = Field(
        default=None, max_length=1000, description="null = allow every table"
    )
    restricted_columns: list[str] = Field(default_factory=list, max_length=2000)


class ConnectionShareUpdate(BaseModel):
    workspace_id: str | None = Field(
        default=None, max_length=36, description="Workspace to share into; null to make it private"
    )


class ConnectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    database_url: str = Field(min_length=1, max_length=4000)
    db_type: str | None = Field(default=None, max_length=20)


class ConnectionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    database_url: str | None = Field(default=None, min_length=1, max_length=4000)


class ConnectionTestResponse(BaseModel):
    ok: bool
    message: str


class ConnectionDeleteResponse(BaseModel):
    message: str


def _to_out(info: ConnectionInfo) -> ConnectionOut:
    return ConnectionOut(
        connection_id=info.connection_id,
        name=info.name,
        db_type=info.db_type,
        is_default=info.is_default,
        has_dsn=info.has_dsn,
        url_preview=info.url_preview,
        created_at=info.created_at,
        updated_at=info.updated_at,
        access=info.access,
        workspace_id=info.workspace_id,
        workspace_name=info.workspace_name,
        owner_email=info.owner_email,
        allowed_tables=info.allowed_tables,
        restricted_columns=info.restricted_columns,
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("", response_model=list[ConnectionOut], summary="List the user's database connections")
@limiter.limit("30/minute")
async def list_connections(
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
    svc: ConnectionService = Depends(get_connection_service),
) -> list[ConnectionOut]:
    infos = await svc.list_connections(current_user.id)
    return [_to_out(i) for i in infos]


@router.post("", response_model=ConnectionOut, summary="Add a new database connection")
@limiter.limit("10/minute")
async def create_connection(
    request: Request,
    body: ConnectionCreate,
    current_user: UserPublic = Depends(get_current_user),
    svc: ConnectionService = Depends(get_connection_service),
) -> ConnectionOut:
    info = await svc.create(current_user.id, body.name, body.database_url, body.db_type)
    return _to_out(info)


@router.put(
    "/{connection_id}", response_model=ConnectionOut, summary="Rename or update a connection"
)
@limiter.limit("10/minute")
async def update_connection(
    request: Request,
    connection_id: str,
    body: ConnectionUpdate,
    current_user: UserPublic = Depends(get_current_user),
    svc: ConnectionService = Depends(get_connection_service),
) -> ConnectionOut:
    info = await svc.update(
        current_user.id,
        connection_id,
        name=body.name,
        raw_url=body.database_url,
    )
    return _to_out(info)


@router.delete(
    "/{connection_id}", response_model=ConnectionDeleteResponse, summary="Delete a connection"
)
@limiter.limit("10/minute")
async def delete_connection(
    request: Request,
    connection_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: ConnectionService = Depends(get_connection_service),
) -> ConnectionDeleteResponse:
    await svc.delete(current_user.id, connection_id)
    # Best-effort: drop the connection's derived schema catalog + vectors.
    try:
        from nl_to_sql.api.dependencies import _get_container

        container = _get_container()
        await container.schema_catalog_service().delete_catalog(connection_id)
        vector_store = container.vector_store()
        if hasattr(vector_store, "delete_by_connection"):
            await vector_store.delete_by_connection(connection_id)
    except Exception as exc:
        logger.warning("Post-delete cleanup failed", connection_id=connection_id, error=str(exc))
    return ConnectionDeleteResponse(message="Connection deleted.")


@router.post(
    "/{connection_id}/test",
    response_model=ConnectionTestResponse,
    summary="Test connectivity of a connection",
)
@limiter.limit("10/minute")
async def test_connection(
    request: Request,
    connection_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: ConnectionService = Depends(get_connection_service),
) -> ConnectionTestResponse:
    await svc.test(current_user.id, connection_id)
    return ConnectionTestResponse(ok=True, message="Connection is reachable.")


@router.post(
    "/{connection_id}/select",
    response_model=ConnectionOut,
    summary="Set a connection as the active/default connection",
)
@limiter.limit("10/minute")
async def select_connection(
    request: Request,
    connection_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: ConnectionService = Depends(get_connection_service),
) -> ConnectionOut:
    info = await svc.set_default(current_user.id, connection_id)
    return _to_out(info)


@router.put(
    "/{connection_id}/policy",
    response_model=ConnectionOut,
    summary="Set which tables the AI may use and which columns are restricted",
    description=(
        "Owner only. `allowed_tables: null` allows every table. Restricted columns "
        "(written `table.column`) are hidden from the model, rejected in SQL wherever "
        "they appear, and redacted from results. Applies to every member using the "
        "connection, on every execution path."
    ),
)
@limiter.limit("20/minute")
async def update_connection_policy(
    request: Request,
    connection_id: str,
    body: ConnectionPolicyUpdate,
    current_user: UserPublic = Depends(get_current_user),
    svc: ConnectionService = Depends(get_connection_service),
) -> ConnectionOut:
    for value in [*(body.allowed_tables or []), *body.restricted_columns]:
        if len(value) > 300:
            raise HTTPException(status_code=422, detail="Table and column names must be 300 characters or fewer.")
    # No explicit cache flush needed: the orchestrator's cache scope includes a
    # fingerprint of the policy, so answers cached under the old rules miss.
    info = await svc.update_policy(
        current_user.id, connection_id, body.allowed_tables, body.restricted_columns
    )
    return _to_out(info)


@router.put(
    "/{connection_id}/workspace",
    response_model=ConnectionOut,
    summary="Share a connection into a workspace, or make it private",
    description="Owner only, and the owner must be an admin of the target workspace.",
)
@limiter.limit("10/minute")
async def share_connection(
    request: Request,
    connection_id: str,
    body: ConnectionShareUpdate,
    current_user: UserPublic = Depends(get_current_user),
    svc: ConnectionService = Depends(get_connection_service),
    workspaces: WorkspaceService = Depends(get_workspace_service),
) -> ConnectionOut:
    await workspaces.set_connection_workspace(current_user.id, connection_id, body.workspace_id)
    for info in await svc.list_connections(current_user.id):
        if info.connection_id == connection_id:
            return _to_out(info)
    raise HTTPException(status_code=404, detail="Connection not found.")
