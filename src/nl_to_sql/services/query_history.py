"""Query history service — persistent database-backed storage for query responses."""
import asyncio
from datetime import date, datetime
from decimal import Decimal
from typing import Any, TypedDict
from uuid import UUID

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.core.models.query import QueryResponse
from nl_to_sql.infrastructure.database.models import QueryHistoryRecord
from nl_to_sql.infrastructure.database.schema_sync import ensure_schema
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url

logger = structlog.get_logger(__name__)


class HistoryEntry(TypedDict):
    """Single history entry structure."""

    id: int
    timestamp: datetime
    query_response: QueryResponse


class QueryHistoryService:
    """Database-backed persistent query history store.

    Uses SQLAlchemy async operations for non-blocking database access.
    History survives server restarts.

    SOLID:
      S — Handles only history storage/retrieval.
      I — Minimal interface for history operations.
    """

    def __init__(self, database_url: str) -> None:
        """Initialize the history service with database connection.

        Args:
            database_url: SQLAlchemy async database URL for history storage.
        """
        database_url = to_async_database_url(database_url)
        self._engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=1,
            pool_timeout=30,
            pool_recycle=300,
            connect_args={"command_timeout": 30, "prepared_statement_cache_size": 0},
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self._logger = logger.bind(service="QueryHistory")

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively convert non-JSON-serializable objects for safe DB storage.

        PostgreSQL asyncpg can return Decimal, datetime, date, UUID, bytes, etc.
        All are converted to JSON-safe Python primitives.
        """
        if isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.hex()
        return obj

    async def initialize(self, retries: int = 3, retry_delay: float = 5.0) -> None:
        """Create the query_history table if it doesn't exist.

        Retries on failure to handle cold-start latency on cloud DB endpoints
        (e.g. Neon auto-suspend wakeup).

        Should be called during application startup.
        """
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                await ensure_schema(self._engine)
                self._logger.info("Query history database initialized")
                return
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    self._logger.warning(
                        "DB init failed, retrying",
                        attempt=attempt,
                        retries=retries,
                        delay=retry_delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(retry_delay)
        raise RuntimeError(
            f"Failed to initialize query history DB after {retries} attempts"
        ) from last_exc

    def _response_to_record(self, response: QueryResponse) -> dict[str, object]:
        """Convert QueryResponse to dict for database storage.

        Args:
            response: The QueryResponse to convert.

        Returns:
            Dict with fields matching QueryHistoryRecord columns.
        """
        # Sanitize execution_result — convert Decimal, datetime, UUID, etc. to JSON-safe types
        execution_result = self._sanitize_for_json(response.execution_result)

        return {
            "question": response.question,
            "sql": response.sql,
            "dialect": response.dialect,
            "is_valid": response.is_valid,
            "validation_errors": response.validation_errors,
            "retrieved_tables": response.retrieved_tables,
            "execution_result": execution_result,
            "execution_error": response.execution_error,
            "tokens_used": response.tokens_used,
            "cached": response.cached,
            "message": response.message,
        }

    def _record_to_response(self, record: QueryHistoryRecord) -> QueryResponse:
        """Convert QueryHistoryRecord to QueryResponse.

        Args:
            record: The database record to convert.

        Returns:
            QueryResponse Pydantic model.
        """
        return QueryResponse(
            question=record.question,
            sql=record.sql,
            dialect=record.dialect,
            is_valid=record.is_valid,
            validation_errors=record.validation_errors or [],
            retrieved_tables=record.retrieved_tables or [],
            execution_result=record.execution_result,
            execution_error=record.execution_error,
            tokens_used=record.tokens_used,
            cached=record.cached,
            message=record.message,
        )

    async def add(self, response: QueryResponse) -> None:
        """Add a QueryResponse to history.

        Args:
            response: The query response to store.
        """
        record_data = self._response_to_record(response)
        record_data["timestamp"] = datetime.utcnow()  # naive UTC — matches TIMESTAMP WITHOUT TIME ZONE column

        async with self._session_factory() as session:
            record = QueryHistoryRecord(**record_data)
            session.add(record)
            await session.commit()
            self._logger.debug(
                "History entry added",
                entry_id=record.id,
                question=response.question[:50],
            )

    async def get_all(self, limit: int = 50, offset: int = 0) -> list[HistoryEntry]:
        """Retrieve paginated history entries (newest first).

        Args:
            limit: Maximum number of entries to return.
            offset: Number of entries to skip.

        Returns:
            List of history entries ordered by newest first.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(QueryHistoryRecord)
                .order_by(QueryHistoryRecord.timestamp.desc())
                .limit(limit)
                .offset(offset)
            )
            records = result.scalars().all()

            entries: list[HistoryEntry] = []
            for record in records:
                entries.append({
                    "id": record.id,
                    "timestamp": record.timestamp,
                    "query_response": self._record_to_response(record),
                })
            return entries

    async def clear(self) -> None:
        """Clear all history entries."""
        async with self._session_factory() as session:
            result = await session.execute(delete(QueryHistoryRecord))
            await session.commit()
            self._logger.info("History cleared", deleted_count=result.rowcount)  # type: ignore[attr-defined]

    async def count(self) -> int:
        """Return total number of history entries.

        Returns:
            Total count of stored entries.
        """
        async with self._session_factory() as session:
            result = await session.execute(select(func.count()).select_from(QueryHistoryRecord))
            return result.scalar() or 0

    async def dispose(self) -> None:
        """Close all database connections."""
        await self._engine.dispose()
        self._logger.info("Query history database connections closed")
