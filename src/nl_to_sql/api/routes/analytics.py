"""Analytics routes â€” GET /api/v1/analytics/*."""
from typing import Any

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from nl_to_sql.api.dependencies import get_container, get_current_user, require_admin
from nl_to_sql.config.container import ApplicationContainer
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


# ── Response schemas (item 12 — typed API contract) ───────────────────────────
class AnalyticsSummary(BaseModel):
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    cached_queries: int
    cache_hit_rate: float
    cache_exact_hit_rate: float
    cache_semantic_hit_rate: float
    cache_layer_lookups: int
    avg_tokens_used: float
    avg_response_time_ms: float
    period_days: int


class PopularQuery(BaseModel):
    question: str
    count: int


class FailurePattern(BaseModel):
    errors: str | None = None
    count: int


class TableUsage(BaseModel):
    table_name: str
    usage_count: int


class IntentDistribution(BaseModel):
    intent_type: str | None = None
    count: int


class PromptVersionPerformance(BaseModel):
    prompt_version: str | None = None
    total_uses: int
    successful_queries: int
    success_rate: float


class CacheStats(BaseModel):
    exact_hits: int
    semantic_hits: int
    misses: int
    total_lookups: int
    exact_hit_rate: float
    semantic_hit_rate: float
    overall_hit_rate: float


class LatencyBreakdown(BaseModel):
    samples: int
    avg_stage_ms: dict[str, float]


async def get_analytics_service(
    container: ApplicationContainer = Depends(get_container),
) -> AnalyticsService:
    """Get analytics service instance."""
    return container.analytics_service()


@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    response: Response,
    days: int = Query(default=30, ge=1, le=365),
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """Get overall analytics summary."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_summary(days=days)  # type: ignore[no-any-return]


@router.get("/popular-queries", response_model=list[PopularQuery])
async def get_popular_queries(
    response: Response,
    limit: int = Query(default=10, ge=1, le=1000),
    days: int = Query(default=30, ge=1, le=365),
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> list[dict[str, Any]]:
    """Get most frequently asked queries."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_popular_queries(limit=limit, days=days)  # type: ignore[no-any-return]


@router.get("/failure-patterns", response_model=list[FailurePattern])
async def get_failure_patterns(
    response: Response,
    days: int = Query(default=30, ge=1, le=365),
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> list[dict[str, Any]]:
    """Get common failure patterns."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_failure_patterns(days=days)  # type: ignore[no-any-return]


@router.get("/table-usage", response_model=list[TableUsage])
async def get_table_usage(
    response: Response,
    limit: int = Query(default=20, ge=1, le=1000),
    days: int = Query(default=30, ge=1, le=365),
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> list[dict[str, Any]]:
    """Get most frequently retrieved tables."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_table_usage(limit=limit, days=days)  # type: ignore[no-any-return]


@router.get("/intent-distribution", response_model=list[IntentDistribution])
async def get_intent_distribution(
    response: Response,
    days: int = Query(default=30, ge=1, le=365),
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> list[dict[str, Any]]:
    """Get distribution of query intent types."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_intent_distribution(days=days)  # type: ignore[no-any-return]


@router.get("/prompt-versions", response_model=list[PromptVersionPerformance])
async def get_prompt_version_performance(
    response: Response,
    days: int = Query(default=30, ge=1, le=365),
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> list[dict[str, Any]]:
    """Get performance metrics for each prompt version."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_prompt_version_performance(days=days)  # type: ignore[no-any-return]


@router.get("/cache-stats", response_model=CacheStats)
async def get_cache_stats(
    response: Response,
    current_user: UserPublic = Depends(get_current_user),
) -> dict[str, Any]:
    """Per-layer cache hit rates (L1 exact vs L2 semantic) since process start."""
    from nl_to_sql.infrastructure.cache.cache_metrics import get_cache_metrics
    if response:
        response.headers["Cache-Control"] = "private, max-age=30"
    stats: dict[str, Any] = get_cache_metrics().snapshot()
    return stats


@router.get("/latency-breakdown", response_model=LatencyBreakdown)
async def get_latency_breakdown(
    response: Response,
    current_user: UserPublic = Depends(get_current_user),
) -> dict[str, Any]:
    """Rolling average per-stage pipeline latency (ms) since process start."""
    from nl_to_sql.infrastructure.cache.cache_metrics import get_stage_metrics
    if response:
        response.headers["Cache-Control"] = "private, max-age=30"
    breakdown: dict[str, Any] = get_stage_metrics().snapshot()
    return breakdown


@router.delete("/reset")
async def reset_analytics(
    _admin: UserPublic = Depends(require_admin),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """Reset all analytics data.

    Clears query history, training data, and feedback records.
    Preserves chat sessions and messages.
    """
    return await analytics_service.reset_analytics()  # type: ignore[no-any-return]


@router.get("/debug")
async def debug_analytics(
    days: int = Query(default=30, ge=1, le=365),
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    """Debug endpoint to check analytics database status. Non-production only."""
    from nl_to_sql.config.settings import get_settings as _gs
    if _gs().app_env == "production":
        from fastapi import HTTPException as _H
        raise _H(status_code=404)
    from datetime import datetime, timedelta

    from sqlalchemy import func, select

    from nl_to_sql.infrastructure.database.models import ChatMessage, QueryHistoryRecord

    db_path = analytics_service._engine.url.render_as_string(hide_password=True)

    try:
        cutoff = datetime.utcnow() - timedelta(days=days)

        async with analytics_service._session_factory() as session:
            # Count records in ChatMessage table
            stmt1 = select(func.count()).select_from(ChatMessage).where(ChatMessage.sql != "")
            chat_count = (await session.execute(stmt1)).scalar() or 0

            # Count records in QueryHistoryRecord table
            stmt2 = select(func.count()).select_from(QueryHistoryRecord)
            history_count = (await session.execute(stmt2)).scalar() or 0

            # Check date range from chat_messages
            stmt3 = select(func.min(ChatMessage.timestamp), func.max(ChatMessage.timestamp)).where(ChatMessage.sql != "")
            min_date, max_date = (await session.execute(stmt3)).first() or (None, None)

            # Recent count
            stmt4 = select(func.count()).select_from(ChatMessage).where(ChatMessage.timestamp >= cutoff, ChatMessage.sql != "")
            recent_count = (await session.execute(stmt4)).scalar() or 0

            # Sample data from chat_messages
            stmt5 = (
                select(
                    ChatMessage.id,
                    ChatMessage.question,
                    ChatMessage.is_valid,
                    ChatMessage.timestamp,
                    ChatMessage.tokens_used,
                    ChatMessage.response_time_ms,
                )
                .where(ChatMessage.sql != "")
                .order_by(ChatMessage.timestamp.desc())
                .limit(3)
            )
            samples = (await session.execute(stmt5)).all()

        return {
            "database_path": db_path,
            "chat_messages_count": chat_count,
            "query_history_count": history_count,
            "date_range": {
                "earliest": min_date.isoformat() if min_date else None,
                "latest": max_date.isoformat() if max_date else None,
            },
            f"messages_in_last_{days}_days": recent_count,
            "sample_records": [
                {
                    "id": s.id,
                    "question": s.question[:50] if s.question else "",
                    "is_valid": s.is_valid,
                    "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                    "tokens_used": s.tokens_used,
                    "response_time_ms": s.response_time_ms,
                }
                for s in samples
            ],
            "status": "ok" if chat_count > 0 else "empty",
            "note": "Analytics now uses chat_messages table (not query_history)",
        }
    except Exception as e:
        return {
            "error": str(e),
            "database_path": db_path,
            "status": "error",
        }
