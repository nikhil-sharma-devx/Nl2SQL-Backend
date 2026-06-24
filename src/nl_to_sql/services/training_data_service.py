"""Training Data Service — Collects and manages training data from successful queries."""
import json
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import TrainingDataRecord
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url

logger = structlog.get_logger(__name__)


class TrainingDataService:
    """Collects and manages training data from successful queries.

    Features:
    - Automatically collects successful query-SQL pairs
    - Tracks usage for fine-tuning
    - Exports training data in various formats
    - Provides statistics on collected data

    SOLID:
      S — Only handles training data collection and management
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
        self._logger = logger.bind(component="TrainingDataService")

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        # Schema is initialized once globally via query_history.initialize() on startup
        self._logger.info("Training data database initialized (schema checked globally)")

    async def collect_training_data(
        self,
        question: str,
        sql: str,
        retrieved_tables: list[str],
        schema_context: str,
        intent_type: str | None = None,
        success_score: float = 1.0,
    ) -> int:
        """Store successful query as training data.

        Args:
            question: The natural language question.
            sql: The generated SQL query.
            retrieved_tables: Tables used in the query.
            schema_context: Schema context used for generation.
            intent_type: Query intent classification.
            success_score: Quality score (0-1).

        Returns:
            Training data record ID.
        """
        try:
            async with self._session_factory() as session:
                record = TrainingDataRecord(
                    question=question,
                    sql=sql,
                    retrieved_tables=retrieved_tables,
                    schema_context=schema_context,
                    intent_type=intent_type,
                    success_score=success_score,
                    used_for_training=False,
                )
                session.add(record)
                await session.commit()
                await session.refresh(record)

                self._logger.info(
                    "Training data collected",
                    record_id=record.id,
                    question=question[:50],
                    intent_type=intent_type,
                )

                return int(record.id)

        except Exception as exc:
            self._logger.warning(
                "Failed to collect training data",
                error=str(exc),
                question=question[:50],
            )
            return -1

    async def get_unused_training_data(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get training data not yet used for fine-tuning.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of training data dictionaries.
        """
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(TrainingDataRecord)
                    .where(TrainingDataRecord.used_for_training == False)  # noqa: E712
                    .order_by(TrainingDataRecord.created_at.desc())
                    .limit(limit)
                )
                records = result.scalars().all()

                return [
                    {
                        "id": record.id,
                        "question": record.question,
                        "sql": record.sql,
                        "retrieved_tables": record.retrieved_tables,
                        "schema_context": record.schema_context,
                        "intent_type": record.intent_type,
                        "success_score": record.success_score,
                        "created_at": record.created_at.isoformat(),
                    }
                    for record in records
                ]

        except Exception as exc:
            self._logger.error("Failed to get unused training data", error=str(exc))
            return []

    async def mark_as_used(self, ids: list[int]) -> int:
        """Mark training records as used for fine-tuning.

        Args:
            ids: List of training data record IDs.

        Returns:
            Number of records marked.
        """
        try:
            async with self._session_factory() as session:
                from sqlalchemy import update
                result = await session.execute(
                    update(TrainingDataRecord)
                    .where(TrainingDataRecord.id.in_(ids))
                    .values(used_for_training=True)
                )
                count: int = result.rowcount  # type: ignore[attr-defined]
                await session.commit()
                self._logger.info("Training data marked as used", count=count)
                return count

        except Exception as exc:
            self._logger.error("Failed to mark training data as used", error=str(exc))
            return 0

    async def export_training_data(
        self,
        format: str = "json",
        limit: int = 1000,
        include_used: bool = False,
    ) -> str:
        """Export training data in JSON format for fine-tuning.

        Args:
            format: Export format (json, jsonl).
            limit: Maximum number of records to export.
            include_used: Whether to include already-used records.

        Returns:
            JSON/JSONL string of training data.
        """
        try:
            async with self._session_factory() as session:
                query = select(TrainingDataRecord)
                if not include_used:
                    query = query.where(TrainingDataRecord.used_for_training == False)  # noqa: E712

                result = await session.execute(
                    query.order_by(TrainingDataRecord.created_at.desc()).limit(limit)
                )
                records = result.scalars().all()

                if format == "jsonl":
                    # JSONL format for OpenAI fine-tuning
                    lines = []
                    for record in records:
                        training_example = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a SQL expert. "
                                        "Convert natural language questions to SQL queries."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": record.question,
                                },
                                {
                                    "role": "assistant",
                                    "content": record.sql,
                                },
                            ]
                        }
                        lines.append(json.dumps(training_example))
                    return "\n".join(lines)
                else:
                    # Regular JSON format
                    data = [
                        {
                            "question": record.question,
                            "sql": record.sql,
                            "retrieved_tables": record.retrieved_tables,
                            "intent_type": record.intent_type,
                            "success_score": record.success_score,
                        }
                        for record in records
                    ]
                    return json.dumps(data, indent=2)

        except Exception as exc:
            self._logger.error("Failed to export training data", error=str(exc))
            return "[]"

    async def get_training_stats(self) -> dict[str, Any]:
        """Get training data statistics.

        Returns:
            Dictionary with training data statistics.
        """
        try:
            from sqlalchemy import func

            async with self._session_factory() as session:
                # Total records
                total_result = await session.execute(
                    select(func.count()).select_from(TrainingDataRecord)
                )
                total = total_result.scalar() or 0

                # Unused records
                unused_result = await session.execute(
                    select(func.count()).select_from(TrainingDataRecord).where(
                        TrainingDataRecord.used_for_training == False  # noqa: E712
                    )
                )
                unused = unused_result.scalar() or 0

                # Used records
                used = total - unused

                # Average success score
                avg_score_result = await session.execute(
                    select(func.avg(TrainingDataRecord.success_score))
                )
                avg_score = float(avg_score_result.scalar() or 0)

                # Intent distribution
                intent_result = await session.execute(
                    select(
                        TrainingDataRecord.intent_type,
                        func.count().label("count"),
                    )
                    .where(TrainingDataRecord.intent_type.isnot(None))
                    .group_by(TrainingDataRecord.intent_type)
                )
                intent_distribution = {
                    row[0]: row[1] for row in intent_result.fetchall()
                }

                return {
                    "total_records": total,
                    "unused_records": unused,
                    "used_records": used,
                    "avg_success_score": round(avg_score, 3),
                    "intent_distribution": intent_distribution,
                }

        except Exception as exc:
            self._logger.error("Failed to get training stats", error=str(exc))
            return {
                "total_records": 0,
                "unused_records": 0,
                "used_records": 0,
                "avg_success_score": 0.0,
                "intent_distribution": {},
            }

    async def get_recent_examples(self, limit: int = 2) -> list[dict[str, Any]]:
        """Fetch recent high-quality Q&A pairs for use as dynamic few-shot prompts."""
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(TrainingDataRecord)
                    .where(TrainingDataRecord.success_score >= 0.9)
                    .order_by(TrainingDataRecord.created_at.desc())
                    .limit(limit)
                )
                records = result.scalars().all()
                return [{"question": r.question, "sql": r.sql} for r in records]
        except Exception as exc:
            self._logger.warning("Failed to get few-shot examples", error=str(exc))
            return []

    async def dispose(self) -> None:
        """Close database connections."""
        await self._engine.dispose()
