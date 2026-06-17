"""Analytics routes — GET /api/v1/analytics/*."""
from fastapi import APIRouter, Depends, Response

from nl_to_sql.api.dependencies import get_container, get_current_user
from nl_to_sql.config.container import ApplicationContainer
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


async def get_analytics_service(
    container: ApplicationContainer = Depends(get_container),
) -> AnalyticsService:
    """Get analytics service instance."""
    return container.analytics_service()


@router.get("/summary")
async def get_analytics_summary(
    days: int = 30,
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    response: Response = None,
) -> dict:
    """Get overall analytics summary."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_summary(days=days)


@router.get("/popular-queries")
async def get_popular_queries(
    limit: int = 10,
    days: int = 30,
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    response: Response = None,
) -> list[dict]:
    """Get most frequently asked queries."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_popular_queries(limit=limit, days=days)


@router.get("/failure-patterns")
async def get_failure_patterns(
    days: int = 30,
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    response: Response = None,
) -> list[dict]:
    """Get common failure patterns."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_failure_patterns(days=days)


@router.get("/table-usage")
async def get_table_usage(
    limit: int = 20,
    days: int = 30,
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    response: Response = None,
) -> list[dict]:
    """Get most frequently retrieved tables."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_table_usage(limit=limit, days=days)


@router.get("/intent-distribution")
async def get_intent_distribution(
    days: int = 30,
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    response: Response = None,
) -> list[dict]:
    """Get distribution of query intent types."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_intent_distribution(days=days)


@router.get("/prompt-versions")
async def get_prompt_version_performance(
    days: int = 30,
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    response: Response = None,
) -> list[dict]:
    """Get performance metrics for each prompt version."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=120"
    return await analytics_service.get_prompt_version_performance(days=days)


@router.delete("/reset")
async def reset_analytics(
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> dict:
    """Reset all analytics data.

    Clears query history, training data, and feedback records.
    Preserves chat sessions and messages.
    """
    return await analytics_service.reset_analytics()


@router.get("/debug")
async def debug_analytics(
    days: int = 30,
    current_user: UserPublic = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> dict:
    """Debug endpoint to check analytics database status."""
    from datetime import datetime, timedelta
    from sqlalchemy import select, func
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
