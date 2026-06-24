"""Schema management routes — ingest and inspect the schema in the vector store."""
import json
from typing import Any, cast

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from nl_to_sql.api.dependencies import (
    get_current_user,
    get_db_client,
    get_ingestion_pipeline,
    get_schema_ingestion,
    get_vector_store,
)
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.rag.ingestion.pipeline import IngestionPipeline
from nl_to_sql.services.schema_ingestion import SchemaIngestionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/schema", tags=["Schema"])

_BLOCKED_SCHEMAS = frozenset({"information_schema", "pg_catalog", "pg_toast", "pg_temp"})


def _validate_schema_name(schema_name: str) -> None:
    if schema_name.lower() in _BLOCKED_SCHEMAS:
        raise HTTPException(
            status_code=400,
            detail=f"Schema '{schema_name}' is not accessible.",
        )


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


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a database schema into the vector store",
    description=(
        "Upload a JSON file describing the database schema. "
        "The service will embed each table description and store it in the vector store. "
        "Pass `reset=true` to wipe existing chunks before ingesting."
    ),
)
async def ingest_schema(
    file: UploadFile = File(..., description="Schema JSON file (see /docs for format)"),
    reset: bool = False,
    current_user: UserPublic = Depends(get_current_user),
    ingestion: SchemaIngestionService = Depends(get_schema_ingestion),
) -> IngestResponse:
    """Ingest an uploaded schema JSON file."""
    content = await file.read()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    schema = SchemaIngestionService.build_schema_from_dict(raw)
    count = await ingestion.ingest(schema, reset=reset)

    return IngestResponse(
        message=f"Successfully ingested schema for '{schema.database_name}'.",
        chunks_ingested=count,
    )


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
    summary="Refresh schema from live database",
    description=(
        "Reflects the current schema tables from the active database connection "
        "and re-ingests them into the vector store. This replaces all existing "
        "schema chunks with the freshly reflected data."
    ),
)
async def refresh_schema(
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
