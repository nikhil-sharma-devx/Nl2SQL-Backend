"""Auto Charting & Dashboards routes.

Per-user dashboard CRUD (create / list / get / rename / duplicate / delete),
per-widget management (add / update / delete / reorder), a ``refresh`` endpoint
that re-runs every widget's SQL against the caller's own database connection,
and a stateless ``recommend-chart`` endpoint that maps a result set to the best
visualization.

Every owner action is scoped to ``current_user.id`` — a cross-user id returns
404 (never 403, so existence is not confirmed). Mutating and refresh endpoints
are rate-limited.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    get_current_user,
    get_dashboard_service,
    get_request_db_client,
    get_starter_content_service,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import Dashboard, DashboardWidget
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.services.dashboard_service import DashboardService
from nl_to_sql.services.starter_content_service import StarterContentService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/dashboards", tags=["Dashboards"])


# ── Request / response models ──────────────────────────────────────────────────


class WidgetCreate(BaseModel):
    title: str = Field(default="Untitled", min_length=1, max_length=200)
    nl_prompt: str | None = Field(default=None, max_length=4000)
    sql: str = Field(default="", max_length=100_000)
    chart_type: str = Field(default="table", max_length=20)
    chart_config: dict[str, Any] | None = None
    layout: dict[str, Any] | None = None
    position: int | None = Field(default=None, ge=0)


class WidgetPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    nl_prompt: str | None = Field(default=None, max_length=4000)
    sql: str | None = Field(default=None, max_length=100_000)
    chart_type: str | None = Field(default=None, max_length=20)
    chart_config: dict[str, Any] | None = None
    layout: dict[str, Any] | None = None
    position: int | None = Field(default=None, ge=0)


class WidgetOut(BaseModel):
    id: str
    dashboard_id: str
    title: str
    nl_prompt: str | None
    sql: str
    chart_type: str
    chart_config: dict[str, Any] | None
    layout: dict[str, Any] | None
    position: int
    created_at: datetime


class DashboardCreate(BaseModel):
    name: str = Field(default="Untitled Dashboard", min_length=1, max_length=200)
    widgets: list[WidgetCreate] | None = None


class DashboardRename(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class ReorderRequest(BaseModel):
    widget_ids: list[str] = Field(..., max_length=200)


class DashboardOut(BaseModel):
    id: str
    name: str
    is_builtin: bool
    created_at: datetime
    updated_at: datetime
    widgets: list[WidgetOut]


class DashboardSummary(BaseModel):
    id: str
    name: str
    is_builtin: bool
    created_at: datetime
    updated_at: datetime
    widget_count: int


class DashboardListResponse(BaseModel):
    items: list[DashboardSummary]
    total: int


class RecommendChartRequest(BaseModel):
    columns: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=100_000)


class ChartRecommendationOut(BaseModel):
    chart_type: str
    x_axis: str | None = None
    y_axis: str | None = None
    reason: str


class WidgetRefreshResult(BaseModel):
    widget_id: str
    title: str
    chart_type: str
    chart_config: dict[str, Any] | None
    rows: list[dict[str, Any]]
    row_count: int
    error: str | None = None


class DashboardRefreshResponse(BaseModel):
    dashboard_id: str
    widgets: list[WidgetRefreshResult]


# ── Shaping helpers ─────────────────────────────────────────────────────────────


def _widget_out(w: DashboardWidget) -> WidgetOut:
    return WidgetOut(
        id=w.id,
        dashboard_id=w.dashboard_id,
        title=w.title,
        nl_prompt=w.nl_prompt,
        sql=w.sql,
        chart_type=w.chart_type,
        chart_config=w.chart_config,
        layout=w.layout,
        position=w.position,
        created_at=w.created_at,
    )


def _dashboard_out(d: Dashboard) -> DashboardOut:
    return DashboardOut(
        id=d.id,
        name=d.name,
        is_builtin=d.is_builtin,
        created_at=d.created_at,
        updated_at=d.updated_at,
        widgets=[_widget_out(w) for w in d.widgets],
    )


def _summary(d: Dashboard) -> DashboardSummary:
    return DashboardSummary(
        id=d.id,
        name=d.name,
        is_builtin=d.is_builtin,
        created_at=d.created_at,
        updated_at=d.updated_at,
        widget_count=len(d.widgets),
    )


_NOT_FOUND = "Dashboard not found"


# ── Chart recommendation (stateless) ────────────────────────────────────────────


@router.post(
    "/recommend-chart",
    response_model=ChartRecommendationOut,
    summary="Recommend the best chart for a result set",
)
async def recommend_chart(
    body: RecommendChartRequest,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> ChartRecommendationOut:
    rec = svc.recommend_chart(body.columns, body.rows)
    return ChartRecommendationOut(**rec)


# ── Dashboard CRUD ──────────────────────────────────────────────────────────────


@router.get("", response_model=DashboardListResponse, summary="List dashboards")
async def list_dashboards(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
    starter_content: StarterContentService = Depends(get_starter_content_service),
) -> DashboardListResponse:
    await starter_content.ensure_dashboards_seeded(current_user.id)
    items, total = await svc.list_dashboards(current_user.id, limit=limit, offset=offset)
    return DashboardListResponse(items=[_summary(d) for d in items], total=total)


@router.post(
    "",
    response_model=DashboardOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a dashboard",
)
@limiter.limit("30/minute")
async def create_dashboard(
    request: Request,
    body: DashboardCreate,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    widgets = [w.model_dump() for w in body.widgets] if body.widgets else None
    dashboard = await svc.create(current_user.id, body.name, widgets)
    return _dashboard_out(dashboard)


@router.get("/{dashboard_id}", response_model=DashboardOut, summary="Get a dashboard")
async def get_dashboard(
    dashboard_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    dashboard = await svc.get(current_user.id, dashboard_id)
    if dashboard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return _dashboard_out(dashboard)


@router.patch("/{dashboard_id}", response_model=DashboardOut, summary="Rename a dashboard")
@limiter.limit("60/minute")
async def rename_dashboard(
    request: Request,
    dashboard_id: str,
    body: DashboardRename,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    dashboard = await svc.rename(current_user.id, dashboard_id, body.name)
    if dashboard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return _dashboard_out(dashboard)


@router.delete(
    "/{dashboard_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a dashboard",
)
@limiter.limit("60/minute")
async def delete_dashboard(
    request: Request,
    dashboard_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> None:
    deleted = await svc.delete(current_user.id, dashboard_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)


@router.post(
    "/{dashboard_id}/duplicate",
    response_model=DashboardOut,
    status_code=status.HTTP_201_CREATED,
    summary="Duplicate a dashboard (deep-copies widgets)",
)
@limiter.limit("20/minute")
async def duplicate_dashboard(
    request: Request,
    dashboard_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    dashboard = await svc.duplicate(current_user.id, dashboard_id)
    if dashboard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return _dashboard_out(dashboard)


# ── Refresh ─────────────────────────────────────────────────────────────────────


@router.post(
    "/{dashboard_id}/refresh",
    response_model=DashboardRefreshResponse,
    summary="Re-run every widget's SQL and return fresh results",
)
@limiter.limit("10/minute")
async def refresh_dashboard(
    request: Request,
    dashboard_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
    db_client: AsyncDatabaseClient = Depends(get_request_db_client),
) -> DashboardRefreshResponse:
    """Execute each widget's SQL against the caller's DB, one failure at a time.

    A single bad widget must never fail the whole refresh — per-widget SQL
    errors are captured and returned as ``error`` on that widget's result.
    """
    dashboard = await svc.get(current_user.id, dashboard_id)
    if dashboard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)

    results: list[WidgetRefreshResult] = []
    for widget in dashboard.widgets:
        rows: list[dict[str, Any]] = []
        error: str | None = None
        if widget.sql.strip():
            try:
                rows = await db_client.execute_sql(widget.sql)
            except Exception as exc:
                error = str(exc)
                logger.info(
                    "dashboard widget refresh failed",
                    user_id=current_user.id,
                    dashboard_id=dashboard_id,
                    widget_id=widget.id,
                    error=error,
                )
        else:
            error = "Widget has no SQL to run."
        results.append(
            WidgetRefreshResult(
                widget_id=widget.id,
                title=widget.title,
                chart_type=widget.chart_type,
                chart_config=widget.chart_config,
                rows=rows,
                row_count=len(rows),
                error=error,
            )
        )

    logger.info(
        "dashboard refreshed",
        user_id=current_user.id,
        dashboard_id=dashboard_id,
        widgets=len(results),
    )
    return DashboardRefreshResponse(dashboard_id=dashboard_id, widgets=results)


# ── Widget management ────────────────────────────────────────────────────────────


@router.post(
    "/{dashboard_id}/widgets",
    response_model=DashboardOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add a widget to a dashboard",
)
@limiter.limit("60/minute")
async def add_widget(
    request: Request,
    dashboard_id: str,
    body: WidgetCreate,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    dashboard = await svc.add_widget(current_user.id, dashboard_id, body.model_dump())
    if dashboard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return _dashboard_out(dashboard)


@router.post(
    "/{dashboard_id}/widgets/reorder",
    response_model=DashboardOut,
    summary="Reorder a dashboard's widgets",
)
@limiter.limit("60/minute")
async def reorder_widgets(
    request: Request,
    dashboard_id: str,
    body: ReorderRequest,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    dashboard = await svc.reorder_widgets(current_user.id, dashboard_id, body.widget_ids)
    if dashboard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return _dashboard_out(dashboard)


@router.patch(
    "/{dashboard_id}/widgets/{widget_id}",
    response_model=DashboardOut,
    summary="Update a widget",
)
@limiter.limit("120/minute")
async def update_widget(
    request: Request,
    dashboard_id: str,
    widget_id: str,
    body: WidgetPatch,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    updates = body.model_dump(exclude_unset=True)
    dashboard = await svc.update_widget(current_user.id, dashboard_id, widget_id, updates)
    if dashboard is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard or widget not found"
        )
    return _dashboard_out(dashboard)


@router.delete(
    "/{dashboard_id}/widgets/{widget_id}",
    response_model=DashboardOut,
    summary="Delete a widget",
)
@limiter.limit("60/minute")
async def delete_widget(
    request: Request,
    dashboard_id: str,
    widget_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    dashboard = await svc.delete_widget(current_user.id, dashboard_id, widget_id)
    if dashboard is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard or widget not found"
        )
    return _dashboard_out(dashboard)
