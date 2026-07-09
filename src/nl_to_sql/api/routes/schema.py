"""Schema management routes — the per-user schema catalog + vector store.

The catalog (Supabase) is the source of truth the Schema page reads from;
Qdrant is a derived index rebuilt from the catalog by ``SchemaCatalogService``.
Legacy vector-store endpoints (``/status``, ``/refresh``, ``/visualize``) are
kept for backward compatibility.
"""
import json
from datetime import datetime
from typing import Any, Literal, cast

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    get_current_user,
    get_db_client,
    get_ingestion_pipeline,
    get_schema_catalog,
    get_user_db_service,
    get_vector_store,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.rag.ingestion.pipeline import IngestionPipeline
from nl_to_sql.services.schema_catalog_service import SchemaCatalogService
from nl_to_sql.services.user_db_service import UserDbConnectionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/schema", tags=["Schema"])

_BLOCKED_SCHEMAS = frozenset({"information_schema", "pg_catalog", "pg_toast", "pg_temp"})

# ── Upload hardening caps (Phase 2 §8) ──────────────────────────────────────────
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
_MAX_TABLES = 2000
_MAX_COLUMNS_PER_TABLE = 1000
_MAX_STRING_LEN = 2000


def _validate_schema_name(schema_name: str) -> None:
    if schema_name.lower() in _BLOCKED_SCHEMAS:
        raise HTTPException(
            status_code=400,
            detail=f"Schema '{schema_name}' is not accessible.",
        )


def _validate_upload_shape(raw: Any) -> dict[str, Any]:
    """Reject oversized / malformed uploaded schema JSON before processing."""
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="Schema JSON must be an object.")
    tables = raw.get("tables")
    if not isinstance(tables, list):
        raise HTTPException(status_code=400, detail="Schema JSON must contain a 'tables' array.")
    if len(tables) > _MAX_TABLES:
        raise HTTPException(status_code=400, detail=f"Too many tables (max {_MAX_TABLES}).")
    for t in tables:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            raise HTTPException(status_code=400, detail="Each table needs a string 'name'.")
        if len(t["name"]) > _MAX_STRING_LEN:
            raise HTTPException(status_code=400, detail="Table name too long.")
        cols = t.get("columns", [])
        if not isinstance(cols, list):
            raise HTTPException(status_code=400, detail=f"Columns for '{t['name']}' must be an array.")
        if len(cols) > _MAX_COLUMNS_PER_TABLE:
            raise HTTPException(
                status_code=400,
                detail=f"Too many columns on '{t['name']}' (max {_MAX_COLUMNS_PER_TABLE}).",
            )
        for c in cols:
            if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                raise HTTPException(status_code=400, detail="Each column needs a string 'name'.")
    return cast(dict[str, Any], raw)


async def _resolve_user_db_client(
    user_id: str,
    user_db_service: UserDbConnectionService,
    server_client: AsyncDatabaseClient,
) -> AsyncDatabaseClient:
    """Return the user's BYOD client, falling back to the server default."""
    try:
        client = await user_db_service.get_client(user_id)
        if client is not None:
            return client
    except Exception as exc:
        logger.warning("Failed to resolve per-user DB client — using server default", error=str(exc))
    return server_client


# ── Response / request models ───────────────────────────────────────────────────


class IngestResponse(BaseModel):
    message: str
    chunks_ingested: int


class SchemaStatusResponse(BaseModel):
    chunks_stored: int
    vector_store_ready: bool


class SchemaRefreshResponse(BaseModel):
    """Response after refreshing schema from the live database."""
    message: str
    tables_found: int
    chunks_ingested: int


class CatalogColumn(BaseModel):
    name: str
    data_type: str = ""
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None
    description: str | None = None


class CatalogTable(BaseModel):
    id: int
    schema_name: str
    table_name: str
    source: Literal["reflected", "uploaded"]
    columns: list[CatalogColumn]
    description: str | None = None  # effective = user_description or reflected_description
    pinned: bool = False
    is_new: bool = False
    last_seen_at: datetime | None = None


class SchemaTablesResponse(BaseModel):
    database_name: str | None = None
    dialect: str = "postgresql"
    source: Literal["reflected", "uploaded", "merged"] = "reflected"
    last_synced_at: datetime | None = None
    tables: list[CatalogTable]


class SyncResponse(BaseModel):
    message: str
    changed: bool
    new_tables: list[str]
    total_tables: int


class TableDescriptionPatch(BaseModel):
    user_description: str | None = Field(default=None, max_length=_MAX_STRING_LEN)


class MarkSeenRequest(BaseModel):
    table_ids: list[int] = Field(default_factory=list)


class MarkSeenResponse(BaseModel):
    updated: int


# ── Catalog read model ──────────────────────────────────────────────────────────


@router.get(
    "/tables",
    response_model=SchemaTablesResponse,
    summary="List the user's catalog tables (with columns, source, pin + new flags)",
)
async def list_schema_tables(
    schema_name: str | None = None,
    current_user: UserPublic = Depends(get_current_user),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> SchemaTablesResponse:
    """Read model for the Schema page — the per-user catalog."""
    if schema_name:
        _validate_schema_name(schema_name)
    data = await catalog.get_catalog(current_user.id, schema_name)
    return SchemaTablesResponse.model_validate(data)


@router.get(
    "/tables/{table_id}",
    response_model=CatalogTable,
    summary="Single-table detail (columns, source, description)",
)
async def get_schema_table(
    table_id: int,
    current_user: UserPublic = Depends(get_current_user),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> CatalogTable:
    data = await catalog.get_catalog(current_user.id)
    for t in data["tables"]:
        if t["id"] == table_id:
            return CatalogTable.model_validate(t)
    raise HTTPException(status_code=404, detail="Table not found.")


@router.patch(
    "/tables/{table_id}",
    response_model=CatalogTable,
    summary="Set a table's user description (sticky, survives re-reflection)",
)
async def patch_schema_table(
    table_id: int,
    body: TableDescriptionPatch,
    current_user: UserPublic = Depends(get_current_user),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> CatalogTable:
    result = await catalog.set_table_description(current_user.id, table_id, body.user_description)
    if result is None:
        raise HTTPException(status_code=404, detail="Table not found.")
    # Re-read to include pinned status.
    data = await catalog.get_catalog(current_user.id)
    for t in data["tables"]:
        if t["id"] == table_id:
            return CatalogTable.model_validate(t)
    return CatalogTable.model_validate({**result, "pinned": False})


@router.post(
    "/tables/seen",
    response_model=MarkSeenResponse,
    summary="Clear the 'new table' badge for the given tables",
)
async def mark_tables_seen(
    body: MarkSeenRequest,
    current_user: UserPublic = Depends(get_current_user),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> MarkSeenResponse:
    updated = await catalog.mark_seen(current_user.id, body.table_ids)
    return MarkSeenResponse(updated=updated)


# ── Sync (auto-generate) + upload (BYOS) ────────────────────────────────────────


@router.post(
    "/sync",
    response_model=SyncResponse,
    summary="Reflect the user's live DB into the catalog and re-embed",
)
@limiter.limit("10/minute")
async def sync_schema(
    request: Request,
    schema_name: str = "public",
    current_user: UserPublic = Depends(get_current_user),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
    user_db_service: UserDbConnectionService = Depends(get_user_db_service),
    server_client: AsyncDatabaseClient = Depends(get_db_client),
) -> SyncResponse:
    _validate_schema_name(schema_name)
    db_client = await _resolve_user_db_client(current_user.id, user_db_service, server_client)
    try:
        result = await catalog.sync_from_live_db(current_user.id, db_client, schema_name)
    except Exception as exc:
        logger.error("Schema sync failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Schema sync failed: {exc}") from exc
    return SyncResponse(
        message="Schema synced from the live database.",
        changed=result.changed,
        new_tables=result.new_tables,
        total_tables=result.total_tables,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload a schema JSON — overlays (or replaces) the catalog",
    description=(
        "Upload a JSON file describing the database schema. It is overlaid onto "
        "the per-user catalog (source='uploaded') and re-embedded. Pass "
        "`reset=true` to replace the whole catalog instead of merging."
    ),
)
@limiter.limit("10/minute")
async def ingest_schema(
    request: Request,
    file: UploadFile = File(..., description="Schema JSON file (see /docs for format)"),
    reset: bool = False,
    current_user: UserPublic = Depends(get_current_user),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> IngestResponse:
    """Ingest an uploaded schema JSON file into the user's catalog."""
    # Size guard — read at most _MAX_UPLOAD_BYTES + 1 to detect overflow.
    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Schema file too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    raw = _validate_upload_shape(raw)
    result = await catalog.apply_upload(current_user.id, raw, replace=reset)
    return IngestResponse(
        message=(
            f"{'Replaced' if reset else 'Merged'} {result.tables} table(s) into your schema catalog."
        ),
        chunks_ingested=result.tables,
    )


# ── Legacy vector-store endpoints (kept for compatibility) ──────────────────────


@router.get(
    "/status",
    response_model=SchemaStatusResponse,
    summary="Get vector store status",
    description="Returns how many schema chunks are currently stored.",
)
async def schema_status(
    vector_store: IVectorStore = Depends(get_vector_store),
) -> SchemaStatusResponse:
    """Return the current state of the vector store."""
    count = await vector_store.count()
    healthy = await vector_store.health_check()
    return SchemaStatusResponse(chunks_stored=count, vector_store_ready=healthy)


@router.post(
    "/refresh",
    response_model=SchemaRefreshResponse,
    summary="Refresh schema from live database (legacy global path)",
    description=(
        "Reflects the current schema tables from the active database connection "
        "and re-ingests them into the vector store. Prefer POST /schema/sync, "
        "which writes to the per-user catalog."
    ),
)
@limiter.limit("10/minute")
async def refresh_schema(
    request: Request,
    schema_name: str = "public",
    current_user: UserPublic = Depends(get_current_user),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> SchemaRefreshResponse:
    """Reflect schema from the live DB and ingest into the vector store."""
    _validate_schema_name(schema_name)
    try:
        chunks_ingested = await pipeline.run(schema_name=schema_name, reset=True)
    except Exception as exc:
        logger.error("Schema refresh failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Schema refresh failed: {exc}",
        ) from exc

    return SchemaRefreshResponse(
        message="Schema refreshed successfully from the live database.",
        tables_found=chunks_ingested,
        chunks_ingested=chunks_ingested,
    )


@router.get(
    "/visualize",
    summary="Get schema visualization data",
    description="Returns the full schema structure including tables, columns, and foreign keys for graph rendering.",
)
async def visualize_schema(
    schema_name: str = "public",
    current_user: UserPublic = Depends(get_current_user),
    db_client: AsyncDatabaseClient = Depends(get_db_client),
) -> dict[str, Any]:
    """Reflect schema from the live DB and return it for visualization."""
    _validate_schema_name(schema_name)
    try:
        schema_def = await db_client.reflect_schema(schema_name=schema_name)
        return cast(dict[str, Any], schema_def)
    except Exception as exc:
        logger.error("Schema visualization failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Schema visualization failed: {exc}",
        ) from exc
