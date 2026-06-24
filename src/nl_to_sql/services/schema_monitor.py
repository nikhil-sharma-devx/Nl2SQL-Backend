"""Schema Monitor — Automatically detects and ingests schema changes."""
import asyncio
from typing import Any

import structlog
from sqlalchemy import text

logger = structlog.get_logger(__name__)


class SchemaMonitor:
    """Monitors database for schema changes and auto-updates vector store.

    Features:
    - Periodically polls database for schema changes
    - Compares current schema with stored schema
    - Re-embeds and upserts changed tables
    - Runs as background task

    SOLID:
      S — Only handles schema monitoring
      D — Depends on database client and ingestion service
    """

    def __init__(
        self,
        db_client: Any,  # AsyncDatabaseClient
        ingestion_service: Any,  # SchemaIngestionService
        check_interval: int = 300,
        enabled: bool = True,
    ) -> None:
        self._db_client = db_client
        self._ingestion_service = ingestion_service
        self._check_interval = check_interval
        self._enabled = enabled
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_schema_hash: dict[str, str] = {}
        self._logger = logger.bind(component="SchemaMonitor")

    async def start(self) -> None:
        """Start the schema monitoring background task."""
        if not self._enabled:
            self._logger.info("Schema monitoring disabled")
            return

        if self._running:
            self._logger.warning("Schema monitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        self._logger.info(
            "Schema monitor started",
            check_interval=self._check_interval,
        )

    async def stop(self) -> None:
        """Stop the schema monitoring background task."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._logger.info("Schema monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_for_changes()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error(
                    "Schema monitoring error — will retry",
                    error=str(exc),
                )
                await asyncio.sleep(60)  # Wait before retry on error

    async def _check_for_changes(self) -> None:
        """Check for schema changes and update if needed.

        Uses a PostgreSQL advisory lock so only one worker process performs
        ingestion at a time — prevents duplicate simultaneous re-ingestions
        when multiple workers are running.
        """
        try:
            current_schema = await self._get_current_schema()
            current_hash = self._compute_schema_hash(current_schema)

            # Compare with last known schema
            changed_tables = []
            for table_name, table_hash in current_hash.items():
                if table_name not in self._last_schema_hash:
                    changed_tables.append(table_name)
                elif self._last_schema_hash[table_name] != table_hash:
                    changed_tables.append(table_name)

            if changed_tables:
                self._logger.info(
                    "Schema changes detected",
                    changed_tables=changed_tables,
                )
                acquired = await self._try_acquire_lock()
                if not acquired:
                    self._logger.info(
                        "Another worker is already ingesting — skipping this cycle"
                    )
                    return
                try:
                    await self._reingest_schema()
                    self._last_schema_hash = current_hash
                finally:
                    await self._release_lock()

        except Exception as exc:
            self._logger.warning("Failed to check schema changes", error=str(exc))

    async def _try_acquire_lock(self) -> bool:
        """Acquire PostgreSQL advisory lock to prevent concurrent ingestions."""
        try:
            async with self._db_client.session() as session:
                result = await session.execute(
                    text("SELECT pg_try_advisory_lock(hashtext('nl2sql_schema_monitor'))")
                )
                return bool(result.scalar())
        except Exception as exc:
            self._logger.warning("Could not acquire advisory lock", error=str(exc))
            return True  # Proceed anyway if locking itself fails (non-PostgreSQL)

    async def _release_lock(self) -> None:
        """Release PostgreSQL advisory lock."""
        try:
            async with self._db_client.session() as session:
                await session.execute(
                    text("SELECT pg_advisory_unlock(hashtext('nl2sql_schema_monitor'))")
                )
        except Exception as exc:
            self._logger.warning("Could not release advisory lock", error=str(exc))

    async def _get_current_schema(self) -> list[dict[str, Any]]:
        """Get current database schema.

        Returns:
            List of table metadata dictionaries.
        """
        # PostgreSQL-specific query to get schema
        query = """
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
        """

        async with self._db_client.session() as session:
            result = await session.execute(text(query))
            rows = result.fetchall()

        # Group by table
        tables: dict[str, dict[str, Any]] = {}
        for row in rows:
            table_name = row[0]
            if table_name not in tables:
                tables[table_name] = {
                    "name": table_name,
                    "columns": [],
                }
            tables[table_name]["columns"].append({
                "name": row[1],
                "data_type": row[2],
                "nullable": row[3] == "YES",
            })

        return list(tables.values())

    def _compute_schema_hash(
        self, schema: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Compute hash for each table's schema.

        Returns:
            Dictionary mapping table_name to schema hash.
        """
        import hashlib
        import json

        table_hashes = {}
        for table in schema:
            # Create a stable representation
            table_repr = json.dumps(table, sort_keys=True)
            table_hash = hashlib.sha256(table_repr.encode()).hexdigest()
            table_hashes[table["name"]] = table_hash

        return table_hashes

    async def _reingest_schema(self) -> None:
        """Re-ingest the full schema into the vector store using auto-discovery."""
        try:
            self._logger.info("Re-ingesting schema from live database")

            # Use auto-discovery to get current schema
            schema_dict = await self._db_client.reflect_schema(schema_name="public")

            # Build schema metadata and ingest
            from nl_to_sql.services.schema_ingestion import SchemaIngestionService
            schema_metadata = SchemaIngestionService.build_schema_from_dict(schema_dict)

            await self._ingestion_service.ingest(schema_metadata, reset=True)

            self._logger.info(
                "Schema re-ingestion complete",
                table_count=len(schema_metadata.tables),
                tables=schema_metadata.table_names,
            )
        except Exception as exc:
            self._logger.error("Schema re-ingestion failed", error=str(exc))
