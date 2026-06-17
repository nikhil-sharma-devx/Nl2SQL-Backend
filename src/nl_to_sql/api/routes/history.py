"""History routes — GET /api/v1/history, DELETE /api/v1/history."""
import csv
import io
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from starlette.responses import StreamingResponse

from nl_to_sql.api.dependencies import get_current_user, get_query_history, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.core.models.query import QueryResponse
from nl_to_sql.infrastructure.database.models import ChatMessage, ChatSession
from nl_to_sql.services.chat_session_service import ChatSessionService
from nl_to_sql.services.query_history import HistoryEntry, QueryHistoryService

router = APIRouter(prefix="/api/v1", tags=["History"])


class HistoryEntryResponse(BaseModel):
    """Single history entry for API response."""

    id: int = Field(..., description="Unique entry identifier.")
    timestamp: datetime = Field(..., description="UTC timestamp when query was made.")
    query_response: QueryResponse = Field(..., description="The query response data.")


class HistoryListResponse(BaseModel):
    """Paginated list of history entries."""

    entries: list[HistoryEntryResponse] = Field(
        default_factory=list,
        description="List of history entries (newest first).",
    )
    total: int = Field(..., description="Total number of entries in history.")
    limit: int = Field(..., description="Maximum entries requested.")
    offset: int = Field(..., description="Number of entries skipped.")


class ClearHistoryResponse(BaseModel):
    """Response after clearing history."""

    message: str = Field(..., description="Status message.")
    previous_count: int = Field(..., description="Number of entries before clearing.")


@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="Get query history",
    description=(
        "Returns paginated query history entries ordered by newest first. "
        "Reads from chat_messages — the unified source for all query data."
    ),
)
async def get_history(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum entries to return."),
    offset: int = Query(default=0, ge=0, description="Number of entries to skip."),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> HistoryListResponse:
    """Retrieve paginated query history from chat_messages."""
    messages, total = await _gather_history(session_service, limit, offset)

    return HistoryListResponse(
        entries=[
            HistoryEntryResponse(
                id=msg.id,
                timestamp=msg.timestamp,
                query_response=QueryResponse(
                    question=msg.question,
                    sql=msg.sql or "",
                    dialect=msg.dialect,
                    is_valid=msg.is_valid,
                    validation_errors=msg.validation_errors or [],
                    retrieved_tables=msg.retrieved_tables or [],
                    used_tables=getattr(msg, 'used_tables', None) or [],
                    execution_result=msg.execution_result,
                    execution_error=msg.execution_error,
                    tokens_used=msg.tokens_used or 0,
                    cached=msg.cached or False,
                    message=msg.message,
                    intent_type=msg.intent_type,
                    suggested_chart=getattr(msg, 'suggested_chart', None),
                    follow_up_questions=getattr(msg, 'follow_up_questions', None) or [],
                ),
            )
            for msg in messages
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


async def _gather_history(session_service: ChatSessionService, limit: int, offset: int):
    """Fetch messages and total count concurrently."""
    import asyncio
    return await asyncio.gather(
        session_service.list_all_messages(limit=limit, offset=offset),
        session_service.count_all_messages(),
    )


@router.delete(
    "/history",
    response_model=ClearHistoryResponse,
    summary="Clear query history (deprecated table)",
    description="Removes all entries from the legacy query_history table. Chat sessions and messages are preserved.",
)
async def clear_history(
    history_service: QueryHistoryService = Depends(get_query_history),
) -> ClearHistoryResponse:
    """Clear legacy query_history records. Use DELETE /sessions to clear chat data."""
    previous_count = await history_service.count()
    await history_service.clear()

    return ClearHistoryResponse(
        message="Legacy query history cleared. Use DELETE /api/v1/sessions to clear chat session data.",
        previous_count=previous_count,
    )


# ── F5: Export history & soft-delete clear ────────────────────────────────────


class ClearHistoryRequest(BaseModel):
    confirm: str = Field(..., description="Must equal the user's email address")


class ClearHistoryV2Response(BaseModel):
    soft_deleted: int


@router.get(
    "/history/export",
    summary="Export query history as CSV or JSON",
)
async def export_history(
    format: str = Query(default="json", pattern="^(csv|json)$"),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> StreamingResponse:
    """Stream chat history as CSV or JSON download."""
    async with session_service._session_factory() as db:
        session_result = await db.execute(
            select(ChatSession.id).where(ChatSession.user_id == current_user.id)
        )
        session_ids = [row[0] for row in session_result.all()]

        rows = []
        if session_ids:
            msg_result = await db.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.session_id.in_(session_ids),
                    ChatMessage.deleted_at.is_(None),
                )
                .order_by(ChatMessage.timestamp.asc())
            )
            rows = msg_result.scalars().all()

    if format == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["timestamp", "question", "sql", "dialect", "is_valid", "tokens_used"])
        for msg in rows:
            writer.writerow([
                msg.timestamp.isoformat(),
                msg.question,
                msg.sql or "",
                msg.dialect,
                msg.is_valid,
                msg.tokens_used or 0,
            ])
        content = buffer.getvalue()
        media_type = "text/csv"
        filename = "history.csv"
    else:
        data = [
            {
                "timestamp": msg.timestamp.isoformat(),
                "question": msg.question,
                "sql": msg.sql or "",
                "dialect": msg.dialect,
                "is_valid": msg.is_valid,
                "tokens_used": msg.tokens_used or 0,
            }
            for msg in rows
        ]
        content = json.dumps(data, indent=2)
        media_type = "application/json"
        filename = "history.json"

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/history/clear",
    response_model=ClearHistoryV2Response,
    summary="Soft-delete all chat history (requires email confirmation)",
)
async def clear_history_v2(
    body: ClearHistoryRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> ClearHistoryV2Response:
    """Soft-delete all messages. User must type their email to confirm."""
    if body.confirm.lower().strip() != current_user.email.lower().strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Confirmation does not match your email address",
        )

    async with session_service._session_factory() as db:
        session_result = await db.execute(
            select(ChatSession.id).where(ChatSession.user_id == current_user.id)
        )
        session_ids = [row[0] for row in session_result.all()]

        soft_deleted = 0
        if session_ids:
            now = datetime.utcnow()
            result = await db.execute(
                update(ChatMessage)
                .where(
                    ChatMessage.session_id.in_(session_ids),
                    ChatMessage.deleted_at.is_(None),
                )
                .values(deleted_at=now)
            )
            soft_deleted = result.rowcount
        await db.commit()

    return ClearHistoryV2Response(soft_deleted=soft_deleted)
