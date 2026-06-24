"""Feedback Service — Collects and processes user feedback."""
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import FeedbackRecord
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url

logger = structlog.get_logger(__name__)


class FeedbackService:
    """Collects and processes user feedback to improve retrieval.

    Features:
    - Store user feedback (correct/incorrect tables, SQL corrections)
    - Update schema chunk metadata with usage patterns
    - Adjust retrieval weights based on feedback

    SOLID:
      S — Only handles feedback collection and processing
      D — Depends on SQLAlchemy for data access
    """

    def __init__(self, database_url: str) -> None:
        database_url = to_async_database_url(database_url)
        self._engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=False,
            pool_size=1,
            max_overflow=0,
            pool_recycle=300,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self._logger = logger.bind(component="FeedbackService")

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        # Schema is initialized once globally via query_history.initialize() on startup
        self._logger.info("Feedback database initialized (schema checked globally)")

    async def submit_feedback(
        self,
        feedback_type: str,
        feedback_data: dict[str, Any],
        query_id: int | None = None,
        session_id: str | None = None,
    ) -> int:
        """Submit user feedback.

        Args:
            feedback_type: Type of feedback (correct_tables, incorrect_tables, sql_correction).
            feedback_data: Feedback details.
            query_id: Optional query ID from history.
            session_id: Optional chat session ID.

        Returns:
            Feedback record ID.
        """
        async with self._session_factory() as session:
            record = FeedbackRecord(
                query_id=query_id,
                session_id=session_id,
                feedback_type=feedback_type,
                feedback_data=feedback_data,
            )
            session.add(record)
            await session.flush()
            await session.commit()

            self._logger.info(
                "Feedback submitted",
                feedback_id=record.id,
                feedback_type=feedback_type,
            )

            return int(record.id)

    async def get_feedback_summary(self, days: int = 30) -> dict[str, Any]:
        """Get summary of feedback received.

        Args:
            days: Number of days to look back.

        Returns:
            Dictionary with feedback statistics.
        """
        from datetime import datetime, timedelta

        from sqlalchemy import func

        cutoff = datetime.utcnow() - timedelta(days=days)

        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    FeedbackRecord.feedback_type,
                    func.count().label("count"),
                )
                .where(FeedbackRecord.timestamp >= cutoff)
                .group_by(FeedbackRecord.feedback_type)
            )
            rows = result.fetchall()

        return {
            "feedback_by_type": {row[0]: row[1] for row in rows},
            "total_feedback": sum(row[1] for row in rows),
            "period_days": days,
        }

    async def dispose(self) -> None:
        """Close database connections."""
        await self._engine.dispose()
