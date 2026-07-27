"""Scheduled Queries & Alerts routes — per-user recurring NL queries.

Every endpoint is authenticated and scoped to the current user; a schedule
owned by another user is reported as *not found* (404) so cross-user
existence cannot be probed (mirrors ``api/routes/connections.py``).
"""
from __future__ import annotations

from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    get_active_connection_id,
    get_current_user,
    get_scheduled_query_service,
    get_starter_content_service,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.scheduled_query_service import ScheduledQueryService, ScheduleInfo
from nl_to_sql.services.starter_content_service import StarterContentService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/schedules", tags=["Schedules"])


# ── Models ───────────────────────────────────────────────────────────────────


class ScheduleOut(BaseModel):
    id: str
    connection_id: str
    name: str
    nl_prompt: str
    cron_expr: str
    raw_schedule_text: str | None = None
    timezone: str
    is_paused: bool
    notify_email: bool
    notify_in_app: bool
    notify_condition: str
    is_builtin: bool
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_status: str | None = None
    consecutive_failures: int
    created_at: datetime
    updated_at: datetime


class ScheduleListResponse(BaseModel):
    items: list[ScheduleOut]


class ScheduleCreate(BaseModel):
    connection_id: str = Field(min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=200)
    nl_prompt: str = Field(min_length=1, max_length=2000)
    schedule_text: str = Field(min_length=1, max_length=500)
    timezone: str = Field(default="UTC", max_length=50)
    notify_email: bool = True
    notify_condition: str = Field(default="always", max_length=20)


class ScheduleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    nl_prompt: str | None = Field(default=None, min_length=1, max_length=2000)
    schedule_text: str | None = Field(default=None, min_length=1, max_length=500)
    timezone: str | None = Field(default=None, max_length=50)
    notify_email: bool | None = None
    notify_condition: str | None = Field(default=None, max_length=20)


class ScheduleRunOut(BaseModel):
    id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    row_count: int | None = None
    generated_sql: str | None = None
    error: str | None = None
    notified: bool
    duration_ms: int | None = None


class ScheduleHistoryResponse(BaseModel):
    items: list[ScheduleRunOut]
    total: int


class ScheduleDeleteResponse(BaseModel):
    message: str


def _to_out(info: ScheduleInfo) -> ScheduleOut:
    return ScheduleOut(
        id=info.id,
        connection_id=info.connection_id,
        name=info.name,
        nl_prompt=info.nl_prompt,
        cron_expr=info.cron_expr,
        raw_schedule_text=info.raw_schedule_text,
        timezone=info.timezone,
        is_paused=info.is_paused,
        notify_email=info.notify_email,
        notify_in_app=info.notify_in_app,
        notify_condition=info.notify_condition,
        is_builtin=info.is_builtin,
        next_run_at=info.next_run_at,
        last_run_at=info.last_run_at,
        last_status=info.last_status,
        consecutive_failures=info.consecutive_failures,
        created_at=info.created_at,
        updated_at=info.updated_at,
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("", response_model=ScheduleListResponse, summary="List the user's scheduled queries")
async def list_schedules(
    connection_id: str | None = Query(default=None),
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
    active_connection_id: str = Depends(get_active_connection_id),
    starter_content: StarterContentService = Depends(get_starter_content_service),
) -> ScheduleListResponse:
    await starter_content.ensure_schedules_seeded(current_user.id, active_connection_id)
    infos = await svc.list_schedules(current_user.id, connection_id=connection_id)
    return ScheduleListResponse(items=[_to_out(i) for i in infos])


@router.post("", response_model=ScheduleOut, summary="Create a scheduled query")
@limiter.limit("10/minute")
async def create_schedule(
    request: Request,
    body: ScheduleCreate,
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
) -> ScheduleOut:
    info = await svc.create(
        current_user.id,
        body.connection_id,
        body.name,
        body.nl_prompt,
        body.schedule_text,
        timezone=body.timezone,
        notify_email=body.notify_email,
        notify_condition=body.notify_condition,
    )
    return _to_out(info)


@router.get("/{schedule_id}", response_model=ScheduleOut, summary="Get a scheduled query")
async def get_schedule(
    schedule_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
) -> ScheduleOut:
    info = await svc.get(current_user.id, schedule_id)
    return _to_out(info)


@router.put("/{schedule_id}", response_model=ScheduleOut, summary="Update a scheduled query")
async def update_schedule(
    schedule_id: str,
    body: ScheduleUpdate,
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
) -> ScheduleOut:
    info = await svc.update(
        current_user.id,
        schedule_id,
        name=body.name,
        nl_prompt=body.nl_prompt,
        schedule_text=body.schedule_text,
        timezone=body.timezone,
        notify_email=body.notify_email,
        notify_condition=body.notify_condition,
    )
    return _to_out(info)


@router.delete(
    "/{schedule_id}", response_model=ScheduleDeleteResponse, summary="Delete a scheduled query"
)
async def delete_schedule(
    schedule_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
) -> ScheduleDeleteResponse:
    await svc.delete(current_user.id, schedule_id)
    return ScheduleDeleteResponse(message="Schedule deleted.")


@router.post("/{schedule_id}/pause", response_model=ScheduleOut, summary="Pause a scheduled query")
async def pause_schedule(
    schedule_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
) -> ScheduleOut:
    info = await svc.pause(current_user.id, schedule_id)
    return _to_out(info)


@router.post("/{schedule_id}/resume", response_model=ScheduleOut, summary="Resume a scheduled query")
async def resume_schedule(
    schedule_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
) -> ScheduleOut:
    info = await svc.resume(current_user.id, schedule_id)
    return _to_out(info)


@router.post(
    "/{schedule_id}/run-now",
    response_model=ScheduleRunOut,
    summary="Run a scheduled query immediately",
)
@limiter.limit("5/minute")
async def run_schedule_now(
    request: Request,
    schedule_id: str,
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
) -> ScheduleRunOut:
    """Execute a schedule immediately, via the same path the worker uses.

    Not per-row-lock-guarded like the worker tick — a manual, user-initiated
    run isn't competing with other worker processes for this row.
    """
    from nl_to_sql.api.dependencies import _get_container
    from nl_to_sql.workers.scheduled_query_worker import execute_schedule

    row = await svc.get_owned_row(current_user.id, schedule_id)
    container = _get_container()
    await execute_schedule(
        row,
        container=container,
        service=svc,
        session_factory=svc.session_factory,
    )
    history, _total = await svc.list_history(current_user.id, schedule_id, limit=1)
    latest = history[0]
    return ScheduleRunOut(
        id=latest.id,
        status=latest.status,
        started_at=latest.started_at,
        finished_at=latest.finished_at,
        row_count=latest.row_count,
        generated_sql=latest.generated_sql,
        error=latest.error,
        notified=latest.notified,
        duration_ms=latest.duration_ms,
    )


@router.get(
    "/{schedule_id}/history",
    response_model=ScheduleHistoryResponse,
    summary="List execution history for a scheduled query",
)
async def get_schedule_history(
    schedule_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserPublic = Depends(get_current_user),
    svc: ScheduledQueryService = Depends(get_scheduled_query_service),
) -> ScheduleHistoryResponse:
    runs, total = await svc.list_history(current_user.id, schedule_id, limit=limit, offset=offset)
    return ScheduleHistoryResponse(
        items=[
            ScheduleRunOut(
                id=r.id,
                status=r.status,
                started_at=r.started_at,
                finished_at=r.finished_at,
                row_count=r.row_count,
                generated_sql=r.generated_sql,
                error=r.error,
                notified=r.notified,
                duration_ms=r.duration_ms,
            )
            for r in runs
        ],
        total=total,
    )
