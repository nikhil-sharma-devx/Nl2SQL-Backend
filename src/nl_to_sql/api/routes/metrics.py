"""Semantic Layer / Metrics Catalog routes — connection-scoped governed metrics.

Scoped to the caller's active connection via ``get_active_connection_id``
(same convention as ``api/routes/schema.py``'s connection-scoped catalog
reads). Reads are open to any user with access to the connection (metrics are
global-to-connection, mirroring the glossary feature's global-to-user
posture); writes (update/delete/certify/uncertify) are further scoped to the
metric's creator — cross-user/cross-connection access is reported as *not
found* (404) so existence cannot be probed.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    get_active_connection_id,
    get_current_user,
    get_metrics_service,
    get_starter_content_service,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.metrics_service import MetricInfo, MetricsService
from nl_to_sql.services.sql_validator import SQLValidatorService
from nl_to_sql.services.starter_content_service import StarterContentService

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])


# ── Models ───────────────────────────────────────────────────────────────────


class MetricOut(BaseModel):
    metric_id: str
    connection_id: str
    name: str
    description: str | None = None
    sql_definition: str
    dimensions: list[str]
    tags: list[str]
    owner: str | None = None
    certified: bool
    is_builtin: bool
    validation_errors: list[str]
    created_at: datetime
    updated_at: datetime


class MetricListResponse(BaseModel):
    items: list[MetricOut]
    total: int


class MetricCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    sql_definition: str = Field(min_length=1, max_length=4000)
    dimensions: list[str] = Field(default_factory=list, max_length=50)
    tags: list[str] = Field(default_factory=list, max_length=50)
    owner: str | None = Field(default=None, max_length=200)


class MetricUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    sql_definition: str | None = Field(default=None, min_length=1, max_length=4000)
    dimensions: list[str] | None = Field(default=None, max_length=50)
    tags: list[str] | None = Field(default=None, max_length=50)
    owner: str | None = Field(default=None, max_length=200)


class MetricDeleteResponse(BaseModel):
    message: str


class MetricPreviewResponse(BaseModel):
    ok: bool
    row_count: int | None = None
    rows: list[dict] | None = None
    estimated_rows: int | None = None
    estimated_cost: float | None = None
    message: str | None = None
    error: str | None = None


def _to_out(info: MetricInfo) -> MetricOut:
    return MetricOut(
        metric_id=info.metric_id,
        connection_id=info.connection_id,
        name=info.name,
        description=info.description,
        sql_definition=info.sql_definition,
        dimensions=info.dimensions,
        tags=info.tags,
        owner=info.owner,
        certified=info.certified,
        is_builtin=info.is_builtin,
        validation_errors=info.validation_errors,
        created_at=info.created_at,
        updated_at=info.updated_at,
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("", response_model=MetricListResponse, summary="List metrics for the active connection")
async def list_metrics(
    search: str | None = Query(default=None, max_length=200),
    tag: str | None = Query(default=None, max_length=100),
    certified_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: MetricsService = Depends(get_metrics_service),
    starter_content: StarterContentService = Depends(get_starter_content_service),
) -> MetricListResponse:
    await starter_content.ensure_metrics_seeded(current_user.id, connection_id)
    infos, total = await svc.list_metrics(
        current_user.id,
        connection_id,
        search=search,
        tag=tag,
        certified_only=certified_only,
        limit=limit,
        offset=offset,
    )
    return MetricListResponse(items=[_to_out(i) for i in infos], total=total)


@router.post("", response_model=MetricOut, summary="Create a metric for the active connection")
@limiter.limit("20/minute")
async def create_metric(
    request: Request,
    body: MetricCreate,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: MetricsService = Depends(get_metrics_service),
) -> MetricOut:
    info = await svc.create(
        current_user.id,
        connection_id,
        body.name,
        body.description,
        body.sql_definition,
        dimensions=body.dimensions,
        tags=body.tags,
        owner=body.owner,
    )
    return _to_out(info)


@router.get("/{metric_id}", response_model=MetricOut, summary="Get a metric")
async def get_metric(
    metric_id: str,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: MetricsService = Depends(get_metrics_service),
) -> MetricOut:
    info = await svc.get(current_user.id, connection_id, metric_id)
    return _to_out(info)


@router.put("/{metric_id}", response_model=MetricOut, summary="Update a metric")
async def update_metric(
    metric_id: str,
    body: MetricUpdate,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: MetricsService = Depends(get_metrics_service),
) -> MetricOut:
    info = await svc.update(
        current_user.id,
        connection_id,
        metric_id,
        name=body.name,
        description=body.description,
        sql_definition=body.sql_definition,
        dimensions=body.dimensions,
        tags=body.tags,
        owner=body.owner,
    )
    return _to_out(info)


@router.delete("/{metric_id}", response_model=MetricDeleteResponse, summary="Delete a metric")
async def delete_metric(
    metric_id: str,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: MetricsService = Depends(get_metrics_service),
) -> MetricDeleteResponse:
    await svc.delete(current_user.id, connection_id, metric_id)
    return MetricDeleteResponse(message="Metric deleted.")


@router.post("/{metric_id}/certify", response_model=MetricOut, summary="Certify a metric")
async def certify_metric(
    metric_id: str,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: MetricsService = Depends(get_metrics_service),
) -> MetricOut:
    info = await svc.certify(current_user.id, connection_id, metric_id)
    return _to_out(info)


@router.post("/{metric_id}/uncertify", response_model=MetricOut, summary="Uncertify a metric")
async def uncertify_metric(
    metric_id: str,
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: MetricsService = Depends(get_metrics_service),
) -> MetricOut:
    info = await svc.uncertify(current_user.id, connection_id, metric_id)
    return _to_out(info)


@router.post(
    "/{metric_id}/preview",
    response_model=MetricPreviewResponse,
    summary="Preview a metric's SQL definition (EXPLAIN by default)",
)
@limiter.limit("20/minute")
async def preview_metric(
    request: Request,
    metric_id: str,
    execute: bool = Query(
        default=False, description="If true, run a capped 50-row sample instead of EXPLAIN"
    ),
    current_user: UserPublic = Depends(get_current_user),
    connection_id: str = Depends(get_active_connection_id),
    svc: MetricsService = Depends(get_metrics_service),
) -> MetricPreviewResponse:
    """Preview a metric's SQL, read-only.

    EXPLAIN-only by default (cheapest/safest — no data returned). With
    ``?execute=true``, runs a capped 50-row sample via the connection's
    existing ``AsyncDatabaseClient`` (read-only + statement-timeout guarantees
    already enforced there — no new execution path).
    """
    from nl_to_sql.api.dependencies import _get_container

    info = await svc.get(current_user.id, connection_id, metric_id)

    # Defense-in-depth (C6): re-validate immediately before execution,
    # independent of the hard block already enforced when the metric's SQL
    # was written — catches anything persisted before that check existed.
    validator = SQLValidatorService(dialect=get_settings().sql_dialect)
    validation = validator.validate(info.sql_definition)
    if not validation.is_valid:
        return MetricPreviewResponse(
            ok=False, error=f"Metric SQL rejected by validation: {'; '.join(validation.errors)}"
        )

    container = _get_container()
    conn_svc = container.connection_service()
    db_client = await conn_svc.get_client(current_user.id, connection_id)
    if db_client is None:
        db_client = container.db_client()

    try:
        if execute:
            wrapped = (
                f"SELECT * FROM ({info.sql_definition.rstrip(';')}) AS _metric_preview LIMIT 50"  # noqa: S608
            )
            rows = await db_client.execute_sql(wrapped)
            return MetricPreviewResponse(ok=True, row_count=len(rows), rows=rows)
        plan = await db_client.explain(info.sql_definition)
        return MetricPreviewResponse(
            ok=bool(plan.get("supported")),
            estimated_rows=plan.get("estimated_rows"),
            estimated_cost=plan.get("estimated_cost"),
            message=plan.get("message"),
        )
    except Exception as exc:
        return MetricPreviewResponse(ok=False, error=str(exc))
