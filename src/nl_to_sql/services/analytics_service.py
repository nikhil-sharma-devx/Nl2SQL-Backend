"""Analytics Service — Aggregates query data for analytics dashboard."""
from datetime import datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import Integer, String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import Base, QueryHistoryRecord, ChatMessage
from nl_to_sql.infrastructure.database.schema_sync import ensure_schema
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url

logger = structlog.get_logger(__name__)


class AnalyticsService:
    """Aggregates query history data for analytics and insights.

    Features:
    - Overall statistics (total queries, success rate, etc.)
    - Popular queries tracking
    - Failure pattern analysis
    - Table usage statistics
    - Latency distribution
    - Prompt version performance

    SOLID:
      S — Only handles analytics aggregation
      D — Depends on SQLAlchemy for data access
    """

    def __init__(self, database_url: str) -> None:
        database_url = to_async_database_url(database_url)
        self._engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=False,
            pool_size=2,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=300,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self._logger = logger.bind(component="AnalyticsService")

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        # Schema is initialized once globally via query_history.initialize() on startup
        self._logger.info("Analytics database initialized (schema checked globally)")

    async def get_summary(self, days: int = 30) -> dict[str, Any]:
        """Get overall analytics summary.

        Args:
            days: Number of days to look back.

        Returns:
            Dictionary with summary statistics.
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            async with self._session_factory() as session:
                # Single-pass conditional aggregation replaces 5 sequential queries
                agg_result = await session.execute(
                    select(
                        func.count().filter(ChatMessage.sql != "").label("total"),
                        func.count().filter(ChatMessage.sql != "", ChatMessage.is_valid == True).label("successful"),
                        func.count().filter(ChatMessage.sql != "", ChatMessage.cached == True).label("cached"),
                        func.avg(ChatMessage.tokens_used).filter(ChatMessage.sql != "").label("avg_tokens"),
                        func.avg(ChatMessage.response_time_ms).filter(
                            ChatMessage.sql != "", ChatMessage.response_time_ms.isnot(None)
                        ).label("avg_latency"),
                    ).where(ChatMessage.timestamp >= cutoff)
                )
                row = agg_result.one()
                total_queries = row.total or 0
                successful_queries = row.successful or 0
                cached_queries = row.cached or 0
                avg_tokens = float(row.avg_tokens or 0)
                avg_latency = float(row.avg_latency or 0)

            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            cache_hit_rate = (cached_queries / total_queries * 100) if total_queries > 0 else 0

            return {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "failed_queries": total_queries - successful_queries,
                "success_rate": round(success_rate, 2),
                "cached_queries": cached_queries,
                "cache_hit_rate": round(cache_hit_rate, 2),
                "avg_tokens_used": round(avg_tokens, 2),
                "avg_response_time_ms": round(avg_latency, 2),
                "period_days": days,
            }
        except Exception as e:
            self._logger.error("Failed to get analytics summary", error=str(e))
            raise

    async def get_popular_queries(self, limit: int = 10, days: int = 30) -> list[dict[str, Any]]:
        """Get most frequently asked queries.

        Args:
            limit: Number of queries to return.
            days: Number of days to look back.

        Returns:
            List of popular queries with counts.
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            async with self._session_factory() as session:
                result = await session.execute(
                    select(
                        ChatMessage.question,
                        func.count().label("count"),
                    )
                    .where(
                        ChatMessage.timestamp >= cutoff,
                        ChatMessage.sql != "",  # Only messages with SQL
                    )
                    .group_by(ChatMessage.question)
                    .order_by(func.count().desc())
                    .limit(limit)
                )
                rows = result.fetchall()

            return [
                {"question": row[0], "count": row[1]}
                for row in rows
            ]
        except Exception as e:
            self._logger.error("Failed to get popular queries", error=str(e))
            raise

    async def get_failure_patterns(self, days: int = 30) -> list[dict[str, Any]]:
        """Get common failure patterns.

        Args:
            days: Number of days to look back.

        Returns:
            List of failure reasons with counts.
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            async with self._session_factory() as session:
                # PostgreSQL cannot GROUP BY JSON directly, so we cast to text
                # and use the same expression in both SELECT and GROUP BY
                validation_errors_text = ChatMessage.validation_errors.cast(String).label("validation_errors_text")

                result = await session.execute(
                    select(
                        validation_errors_text,
                        func.count().label("count"),
                    )
                    .where(
                        ChatMessage.timestamp >= cutoff,
                        ChatMessage.sql != "",
                        ChatMessage.is_valid == False,
                        ChatMessage.validation_errors.isnot(None),
                    )
                    .group_by(validation_errors_text)
                    .order_by(func.count().desc())
                    .limit(20)
                )
                rows = result.fetchall()

            return [
                {"errors": row[0], "count": row[1]}
                for row in rows
            ]
        except Exception as e:
            self._logger.error("Failed to get failure patterns", error=str(e))
            raise

    async def get_table_usage(self, days: int = 30, limit: int = 20) -> list[dict[str, Any]]:
        """Get most frequently retrieved tables.

        Args:
            days: Number of days to look back.
            limit: Number of tables to return.

        Returns:
            List of tables with usage counts.
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            async with self._session_factory() as session:
                from sqlalchemy import text
                rows = (await session.execute(
                    text("""
                        SELECT tbl, count(*) AS cnt
                        FROM chat_messages,
                             jsonb_array_elements_text(retrieved_tables::jsonb) AS tbl
                        WHERE timestamp >= :cutoff
                          AND sql != ''
                          AND retrieved_tables IS NOT NULL
                        GROUP BY tbl
                        ORDER BY cnt DESC
                        LIMIT :lim
                    """),
                    {"cutoff": cutoff, "lim": limit},
                )).all()

            return [{"table_name": row.tbl, "usage_count": row.cnt} for row in rows]
        except Exception as e:
            self._logger.error("Failed to get table usage", error=str(e))
            raise

    async def get_intent_distribution(self, days: int = 30) -> list[dict[str, Any]]:
        """Get distribution of query intent types.

        Args:
            days: Number of days to look back.

        Returns:
            List of intent types with counts.
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            async with self._session_factory() as session:
                result = await session.execute(
                    select(
                        ChatMessage.intent_type,
                        func.count().label("count"),
                    )
                    .where(
                        ChatMessage.timestamp >= cutoff,
                        ChatMessage.sql != "",
                        ChatMessage.intent_type.isnot(None),
                    )
                    .group_by(ChatMessage.intent_type)
                    .order_by(func.count().desc())
                )
                rows = result.fetchall()

            return [
                {"intent_type": row[0], "count": row[1]}
                for row in rows
            ]
        except Exception as e:
            self._logger.error("Failed to get intent distribution", error=str(e))
            raise

    async def get_prompt_version_performance(self, days: int = 30) -> list[dict[str, Any]]:
        """Get performance metrics for each prompt version.

        Args:
            days: Number of days to look back.

        Returns:
            List of prompt versions with success rates.
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            async with self._session_factory() as session:
                result = await session.execute(
                    select(
                        ChatMessage.prompt_version,
                        func.count().label("total"),
                        func.sum(
                            cast(ChatMessage.is_valid, Integer)
                        ).label("successes"),
                    )
                    .where(
                        ChatMessage.timestamp >= cutoff,
                        ChatMessage.sql != "",
                        ChatMessage.prompt_version.isnot(None),
                    )
                    .group_by(ChatMessage.prompt_version)
                )
                rows = result.fetchall()

            return [
                {
                    "prompt_version": row[0],
                    "total_uses": row[1],
                    "successful_queries": row[2] or 0,
                    "success_rate": round((row[2] or 0) / row[1] * 100, 2) if row[1] > 0 else 0,
                }
                for row in rows
            ]
        except Exception as e:
            self._logger.error("Failed to get prompt version performance", error=str(e))
            raise

    async def reset_analytics(self) -> dict[str, Any]:
        """Reset all analytics data.

        Clears query history, training data, and feedback records.
        Preserves chat sessions and messages.

        Returns:
            Dictionary with counts of deleted records.
        """
        from sqlalchemy import delete
        from nl_to_sql.infrastructure.database.models import FeedbackRecord, TrainingDataRecord

        try:
            async with self._session_factory() as session:
                # Use rowcount from DELETE to avoid 3 pre-flight COUNT queries
                r1 = await session.execute(delete(QueryHistoryRecord))
                r2 = await session.execute(delete(FeedbackRecord))
                r3 = await session.execute(delete(TrainingDataRecord))
                query_count = r1.rowcount
                feedback_count = r2.rowcount
                training_count = r3.rowcount
                await session.commit()

                self._logger.info(
                    "Analytics data reset successfully",
                    query_history_deleted=query_count,
                    feedback_deleted=feedback_count,
                    training_data_deleted=training_count,
                )

                return {
                    "message": "Analytics data reset successfully",
                    "deleted_records": {
                        "query_history": query_count,
                        "feedback": feedback_count,
                        "training_data": training_count,
                    },
                }

        except Exception as e:
            self._logger.error("Failed to reset analytics", error=str(e))
            raise

    async def dispose(self) -> None:
        """Close database connections."""
        await self._engine.dispose()
