"""F6 - Usage & Limits routes (GET /api/v1/usage)."""
from datetime import datetime, timedelta
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import ChatMessage, ChatSession
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Usage"])

# Rough cost estimate per token (Groq pricing, USD)
_COST_PER_TOKEN = 0.000_000_59


class UsageResponse(BaseModel):
    queries_used: int
    tokens_in: int
    tokens_out: int
    est_cost_usd: float
    period: str


@router.get("/usage", response_model=UsageResponse, summary="Get usage stats for the current user")
async def get_usage(
    period: Literal["today", "7d", "30d"] = Query(default="7d"),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> UsageResponse:
    """Return aggregated usage derived from chat_messages — always populated."""
    now = datetime.utcnow()
    if period == "today":
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "7d":
        since = now - timedelta(days=7)
    else:
        since = now - timedelta(days=30)

    async with session_service._session_factory() as db:
        result = await db.execute(
            select(
                func.count(ChatMessage.id),
                func.coalesce(func.sum(ChatMessage.tokens_used), 0),
            )
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .where(
                ChatSession.user_id == current_user.id,
                ChatMessage.timestamp >= since,
                ChatMessage.deleted_at.is_(None),
                ChatMessage.sql != "",
            )
        )
        row = result.one()

    queries_used = int(row[0])
    total_tokens = int(row[1])

    return UsageResponse(
        queries_used=queries_used,
        tokens_in=total_tokens,
        tokens_out=0,
        est_cost_usd=round(total_tokens * _COST_PER_TOKEN, 6),
        period=period,
    )
