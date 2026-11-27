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
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    get_active_connection_id,
    get_connection_service,
    get_current_user,
    get_db_client,
    get_schema_catalog,
    get_schema_doc_service,
    get_vector_store,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.services.connection_service import ConnectionService
from nl_to_sql.services.schema_catalog_service import SchemaCatalogService, SyncResult
from nl_to_sql.services.schema_doc_service import SchemaDocService, SchemaExplanation

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
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> SchemaTablesResponse:
    """Read model for the Schema page — the active connection's catalog."""
    if schema_name:
        _validate_schema_name(schema_name)
    data = await catalog.get_catalog(current_user.id, connection_id, schema_name)
    return SchemaTablesResponse.model_validate(data)


@router.get(
    "/tables/{table_id}",
    response_model=CatalogTable,
    summary="Single-table detail (columns, source, description)",
)
async def get_schema_table(
    table_id: int,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> CatalogTable:
    data = await catalog.get_catalog(current_user.id, connection_id)
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
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> CatalogTable:
    result = await catalog.set_table_description(
        current_user.id, connection_id, table_id, body.user_description
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Table not found.")
    # Re-read to include pinned status.
    data = await catalog.get_catalog(current_user.id, connection_id)
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
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> MarkSeenResponse:
    updated = await catalog.mark_seen(current_user.id, connection_id, body.table_ids)
    return MarkSeenResponse(updated=updated)


# ── RAG-powered explanations ────────────────────────────────────────────────────


@router.get(
    "/explain",
    response_model=SchemaExplanation,
    summary="RAG-powered explanation for a table or column",
    description=(
        "Generates a structured explanation (description, business meaning, "
        "relationships, example usage, common joins, example SQL) for a table "
        "or column from the user's schema documentation. Results are cached."
    ),
)
@limiter.limit("30/minute")
async def explain_schema(
    request: Request,
    table: str,
    column: str | None = None,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: SchemaDocService = Depends(get_schema_doc_service),
) -> SchemaExplanation:
    """Return a cached-or-generated explanation for a schema table/column."""
    if not table or len(table) > _MAX_STRING_LEN:
        raise HTTPException(status_code=400, detail="Invalid table name.")
    if column is not None and len(column) > _MAX_STRING_LEN:
        raise HTTPException(status_code=400, detail="Invalid column name.")
    return await svc.explain(
        user_id=current_user.id, connection_id=connection_id, table=table, column=column
    )


# ── Sync (auto-generate) + upload (BYOS) ────────────────────────────────────────


_MAX_SYNC_SCHEMAS = 200


class SchemaInfo(BaseModel):
    name: str
    table_count: int = Field(description="tables, views and other queryable relations")
    synced: bool = Field(description="currently in this connection's catalog")


class SchemaListResponse(BaseModel):
    schemas: list[SchemaInfo]


async def _resolve_sync_targets(
    connection_id: str,
    schema_name: str | None,
    schemas: list[str] | None,
    catalog: SchemaCatalogService,
    db_client: AsyncDatabaseClient,
) -> tuple[str, list[str] | None]:
    """Decide what a sync reflects: ``(schema_name, schemas)`` for the catalog.

    - ``schemas`` given: exactly those (the user's ticked set).
    - ``schema_name`` given: that one schema only (legacy callers).
    - neither: the schemas already in the catalog, or — first sync of a
      connection — every user schema in the database that has tables.
    """
    if schemas:
        cleaned = list(dict.fromkeys(n.strip() for n in schemas if n and n.strip()))
        if len(cleaned) > _MAX_SYNC_SCHEMAS:
            raise HTTPException(status_code=400, detail=f"Pick at most {_MAX_SYNC_SCHEMAS} schemas.")
        for name in cleaned:
            _validate_schema_name(name)
        return "public", cleaned
    if schema_name:
        _validate_schema_name(schema_name)
        return schema_name, None
    already = await catalog.synced_schemas(connection_id)
    if already:
        return "public", already
    try:
        found = [s["name"] for s in await db_client.list_schemas() if s["table_count"] > 0]
    except Exception as exc:  # e.g. a non-PostgreSQL database
        logger.warning("Listing schemas failed; syncing public", error=str(exc))
        found = []
    return "public", (found or None)


async def _sync_active_connection(
    user_id: str,
    connection_id: str,
    schema_name: str | None,
    schemas: list[str] | None,
    catalog: SchemaCatalogService,
    conns: ConnectionService,
    default_client: AsyncDatabaseClient,
) -> SyncResult:
    """Reflect the active connection's live DB into its catalog.

    The client is resolved from the SAME ``connection_id`` the catalog is
    written under, in strict mode: a connection with a URL that can't be
    opened is an error, never a silent fallback to the platform database
    (which would file the default DB's tables under this connection).
    """
    db_client = await conns.get_scoped_client(user_id, connection_id, default_client, strict=True)
    name, targets = await _resolve_sync_targets(
        connection_id, schema_name, schemas, catalog, db_client
    )
    try:
        return await catalog.sync_from_live_db(
            user_id, connection_id, db_client, name, schemas=targets
        )
    except Exception as exc:
        logger.error("Schema sync failed", connection_id=connection_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Schema sync failed: {exc}") from exc


@router.get(
    "/schemas",
    response_model=SchemaListResponse,
    summary="Schemas in the active connection's database (to choose what to sync)",
)
async def list_schemas(
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
    conns: ConnectionService = Depends(get_connection_service),
    default_client: AsyncDatabaseClient = Depends(get_db_client),
) -> SchemaListResponse:
    db_client = await conns.get_scoped_client(
        current_user.id, connection_id, default_client, strict=True
    )
    try:
        found = await db_client.list_schemas()
    except Exception as exc:
        logger.error("Listing schemas failed", connection_id=connection_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Couldn't list schemas: {exc}") from exc
    synced = set(await catalog.synced_schemas(connection_id))
    return SchemaListResponse(
        schemas=[
            SchemaInfo(name=s["name"], table_count=s["table_count"], synced=s["name"] in synced)
            for s in found
        ]
    )


@router.post(
    "/sync",
    response_model=SyncResponse,
    summary="Reflect the active connection's live DB into the catalog and re-embed",
    description=(
        "Pass `schemas` (repeatable) to sync exactly those schemas; reflected tables "
        "from unlisted schemas are dropped. Without it, re-syncs the schemas already "
        "in the catalog, or on the first sync every schema that has tables. "
        "`schema_name` syncs a single schema (legacy)."
    ),
)
@limiter.limit("10/minute")
async def sync_schema(
    request: Request,
    schema_name: str | None = None,
    schemas: list[str] | None = Query(default=None),
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
    conns: ConnectionService = Depends(get_connection_service),
    default_client: AsyncDatabaseClient = Depends(get_db_client),
) -> SyncResponse:
    result = await _sync_active_connection(
        current_user.id, connection_id, schema_name, schemas, catalog, conns, default_client
    )
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
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
) -> IngestResponse:
    """Ingest an uploaded schema JSON file into the active connection's catalog."""
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
    result = await catalog.apply_upload(current_user.id, connection_id, raw, replace=reset)
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
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    vector_store: IVectorStore = Depends(get_vector_store),
) -> SchemaStatusResponse:
    """Return the vector-store state for the active connection."""
    count = await vector_store.count(connection_id=connection_id)
    healthy = await vector_store.health_check()
    return SchemaStatusResponse(chunks_stored=count, vector_store_ready=healthy)


@router.post(
    "/refresh",
    response_model=SchemaRefreshResponse,
    summary="Refresh schema from the active connection (alias of POST /schema/sync)",
    description=(
        "Kept for older clients. Reflects the ACTIVE connection's live database "
        "into that connection's catalog, exactly like POST /schema/sync. It used "
        "to re-ingest the server's default database globally, which showed the "
        "default tables no matter which connection was selected."
    ),
)
@limiter.limit("10/minute")
async def refresh_schema(
    request: Request,
    schema_name: str | None = None,
    schemas: list[str] | None = Query(default=None),
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
    conns: ConnectionService = Depends(get_connection_service),
    default_client: AsyncDatabaseClient = Depends(get_db_client),
) -> SchemaRefreshResponse:
    result = await _sync_active_connection(
        current_user.id, connection_id, schema_name, schemas, catalog, conns, default_client
    )
    return SchemaRefreshResponse(
        message="Schema refreshed from the active connection.",
        tables_found=result.total_tables,
        chunks_ingested=result.total_tables,
    )


def _sql_name(schema: str | None, table: str) -> str:
    """The name to write in SQL: bare in ``public``, ``schema.table`` elsewhere."""
    return table if schema in (None, "", "public") else f"{schema}.{table}"


@router.get(
    "/visualize",
    summary="Get schema visualization data",
    description=(
        "Tables, columns and foreign keys of the ACTIVE connection for the schema "
        "graph and visual query builder. Served from the synced catalog (every "
        "synced schema), so it matches what the AI sees; before the first sync it "
        "reads the live database's schemas instead. Each table's `name` is how to "
        "write it in SQL (`orders` in public, `sales.orders` elsewhere); "
        "`foreign_key` values use the same form plus `.column`. `schema_name` "
        "limits the result to one schema."
    ),
)
async def visualize_schema(
    schema_name: str | None = None,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    catalog: SchemaCatalogService = Depends(get_schema_catalog),
    conns: ConnectionService = Depends(get_connection_service),
    default_client: AsyncDatabaseClient = Depends(get_db_client),
) -> dict[str, Any]:
    if schema_name:
        _validate_schema_name(schema_name)

    data = await catalog.get_catalog(current_user.id, connection_id, schema_name)
    if data["tables"]:
        return {
            "database_name": data["database_name"],
            "dialect": data["dialect"],
            "source": "catalog",
            "last_synced_at": data["last_synced_at"],
            "tables": [
                {
                    "name": _sql_name(t["schema_name"], t["table_name"]),
                    "schema_name": t["schema_name"],
                    "table_name": t["table_name"],
                    "description": t["description"],
                    "columns": t["columns"],
                }
                for t in data["tables"]
            ],
        }

    # Nothing synced yet: read the live DB (strict client — never the platform
    # DB in place of this connection; read-only transaction).
    db_client = await conns.get_scoped_client(
        current_user.id, connection_id, default_client, strict=True
    )
    try:
        if schema_name:
            targets = [schema_name]
        else:
            targets = [x["name"] for x in await db_client.list_schemas() if x["table_count"] > 0]
        live = await db_client.reflect_schema(schemas=targets or ["public"])
    except Exception as exc:
        logger.error("Schema visualization failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Schema visualization failed: {exc}",
        ) from exc
    for t in live["tables"]:
        t["table_name"] = t["name"]
        t["name"] = _sql_name(t.get("schema_name"), t["name"])
    live["source"] = "live"
    return live
