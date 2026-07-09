"""Async SQLAlchemy client — manages the target database connection."""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.core.exceptions import DatabaseExecutionError
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url
from nl_to_sql.infrastructure.observability.tracing import set_span_attribute, trace_function

logger = structlog.get_logger(__name__)


class AsyncDatabaseClient:
    """Async SQLAlchemy session factory for the target query database.

    Responsibilities:
      - Create and manage the async engine.
      - Provide session context managers.
      - Execute raw SQL strings (for query execution feature).

    SOLID: S — Manages DB lifecycle only; not responsible for SQL generation.
    """

    MAX_ROWS = 5_000

    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 300,
        readonly: bool = True,
        statement_timeout_ms: int = 30_000,
    ) -> None:
        self._readonly = readonly
        self._statement_timeout_ms = max(1_000, int(statement_timeout_ms))
        # Ensure an async driver (e.g. plain postgresql:// -> postgresql+asyncpg://)
        database_url = to_async_database_url(database_url)

        # Sanitize asyncpg unsupported parameters
        import re
        if "asyncpg" in database_url:
            # asyncpg doesn't support channel_binding, which some providers like Supabase append
            database_url = re.sub(r"([?&])channel_binding=[^&]*", r"\1", database_url)
            # Remove trailing ? or & if we just stripped the last param
            database_url = database_url.rstrip("?&")
            # Replace sslmode= with ssl= for asyncpg
            if "sslmode=" in database_url:
                database_url = database_url.replace("sslmode=", "ssl=")

        self._engine = create_async_engine(
            database_url,
            echo=echo,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            connect_args={"prepared_statement_cache_size": 0},
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        logger.info(
            "Database engine created",
            url=self._sanitise_url(database_url),
            pool_size=pool_size,
            max_overflow=max_overflow,
        )

    async def _apply_session_guards(self, sess: AsyncSession) -> None:
        """Harden a session before running generated SQL (PostgreSQL only).

        - Statement timeout keeps a runaway query from pinning a pool
          connection indefinitely.
        - READ ONLY makes the transaction reject any write even if a
          non-SELECT slipped past the validator (belt-and-suspenders).
        """
        from sqlalchemy import text

        if self._engine.dialect.name != "postgresql":
            return
        await sess.execute(
            text(f"SET LOCAL statement_timeout = {self._statement_timeout_ms}")
        )
        if self._readonly:
            await sess.execute(text("SET TRANSACTION READ ONLY"))

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield an async SQLAlchemy session with automatic rollback on error."""
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    @trace_function("database.execute_sql")
    async def execute_sql(self, sql: str) -> list[dict[str, Any]]:
        """Execute a raw SQL string and return rows as a list of dicts.

        Args:
            sql: The SQL string to run against the target database.

        Returns:
            A list of row dictionaries.

        Raises:
            DatabaseExecutionError: On any DB error.
        """
        from sqlalchemy import text

        set_span_attribute("db.statement", sql)

        try:
            async with self.session() as sess:
                await self._apply_session_guards(sess)
                result = await sess.execute(text(sql))
                columns = list(result.keys())
                all_rows = result.fetchall()
                truncated = len(all_rows) > self.MAX_ROWS
                rows = [dict(zip(columns, row, strict=True)) for row in all_rows[: self.MAX_ROWS]]
                set_span_attribute("db.rows_returned", len(rows))
                set_span_attribute("db.truncated", truncated)
                return rows
        except Exception as exc:
            logger.error("SQL execution failed", error=str(exc))
            raise DatabaseExecutionError(
                f"Failed to execute SQL: {exc}", detail=str(exc)
            ) from exc

    async def execute_sql_stream(self, sql: str, batch_size: int = 100) -> AsyncGenerator[list[dict[str, Any]], None]:
        """Execute SQL and yield result rows in batches for large result sets.

        Streams rows in batches instead of buffering everything in memory,
        improving scalability when execute=true returns large datasets.

        Args:
            sql: The SQL string to run.
            batch_size: Number of rows per yielded batch.

        Yields:
            Lists of row dicts, one batch at a time.
        """
        from sqlalchemy import text

        set_span_attribute("db.statement", sql)

        try:
            async with self._session_factory() as sess:
                await self._apply_session_guards(sess)
                stream_result = await sess.stream(text(sql))
                columns = list(stream_result.keys())
                async for partition in stream_result.partitions(batch_size):
                    yield [dict(zip(columns, row, strict=True)) for row in partition]
        except Exception as exc:
            logger.error("SQL stream execution failed", error=str(exc))
            raise DatabaseExecutionError(
                f"Failed to stream SQL: {exc}", detail=str(exc)
            ) from exc

    async def health_check(self) -> bool:
        """Run a trivial query to verify the connection."""
        from sqlalchemy import text

        try:
            async with self.session() as sess:
                await sess.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def reflect_schema(self, schema_name: str = "public") -> dict[str, Any]:
        """Reflect database schema from live PostgreSQL database.

        Uses pg_catalog system tables directly (instead of the slow
        information_schema views) for speed on cloud-hosted pooled
        connections such as Neon.

        Args:
            schema_name: The PostgreSQL schema to reflect (default: 'public').

        Returns:
            Dictionary with database_name, dialect, and tables list.
        """
        from sqlalchemy import text

        logger.info("Reflecting schema from live database", schema=schema_name)

        # ── Fast pg_catalog query ──────────────────────────────────────────────
        # information_schema views are themselves complex multi-join views that
        # time out on Neon pooled endpoints.  Querying pg_catalog directly is
        # orders of magnitude faster because the tables are real system tables
        # with indexed OID look-ups.
        schema_query = """
        SELECT
            cls.relname                          AS table_name,
            att.attname                          AS column_name,
            pg_catalog.format_type(att.atttypid, att.atttypmod) AS data_type,
            NOT att.attnotnull                   AS is_nullable,
            pg_get_expr(ad.adbin, ad.adrelid)    AS column_default,
            att.attnum                           AS ordinal_position,
            /* primary key? */
            CASE WHEN pk_att.attnum IS NOT NULL THEN true ELSE false END AS is_pk,
            /* foreign key target table.column (NULL when not an FK) */
            fk_cls.relname                       AS fk_table,
            fk_att.attname                       AS fk_column,
            /* optional comments */
            obj_description(cls.oid, 'pg_class')           AS table_comment,
            col_description(cls.oid, att.attnum)           AS column_comment
        FROM pg_catalog.pg_class cls
        JOIN pg_catalog.pg_namespace ns
            ON ns.oid = cls.relnamespace
        JOIN pg_catalog.pg_attribute att
            ON att.attrelid = cls.oid
           AND att.attnum > 0
           AND NOT att.attisdropped
        /* column defaults */
        LEFT JOIN pg_catalog.pg_attrdef ad
            ON ad.adrelid = cls.oid AND ad.adnum = att.attnum
        /* primary-key membership */
        LEFT JOIN pg_catalog.pg_constraint pk_con
            ON pk_con.conrelid = cls.oid AND pk_con.contype = 'p'
        LEFT JOIN LATERAL unnest(pk_con.conkey) pk_col(attnum) ON true
        LEFT JOIN pg_catalog.pg_attribute pk_att
            ON pk_att.attrelid = cls.oid
           AND pk_att.attnum = pk_col.attnum
           AND pk_att.attnum = att.attnum
        /* foreign-key look-up */
        LEFT JOIN pg_catalog.pg_constraint fk_con
            ON fk_con.conrelid = cls.oid AND fk_con.contype = 'f'
        LEFT JOIN LATERAL (
            SELECT
                fk_con.confrelid,
                u.src_attnum,
                u.ref_attnum
            FROM unnest(fk_con.conkey, fk_con.confkey)
                 AS u(src_attnum, ref_attnum)
        ) fk_map ON fk_map.src_attnum = att.attnum
        LEFT JOIN pg_catalog.pg_class fk_cls
            ON fk_cls.oid = fk_map.confrelid
        LEFT JOIN pg_catalog.pg_attribute fk_att
            ON fk_att.attrelid = fk_map.confrelid
           AND fk_att.attnum = fk_map.ref_attnum
        WHERE ns.nspname = :schema
          AND cls.relkind = 'r'       -- ordinary tables only
        ORDER BY cls.relname, att.attnum
        """

        async with self.session() as sess:
            # Safety net: prevent indefinite hangs
            await sess.execute(text("SET LOCAL statement_timeout = '90s'"))

            result = await sess.execute(text(schema_query), {"schema": schema_name})
            rows = result.fetchall()

        # ── Build schema structure ─────────────────────────────────────────────
        tables_dict: dict[str, dict[str, Any]] = {}

        for row in rows:
            table_name    = row[0]
            column_name   = row[1]
            data_type     = row[2]
            is_nullable   = row[3]
            is_pk         = row[6]
            fk_ref_table  = row[7]
            fk_ref_column = row[8]
            table_comment = row[9]
            col_comment   = row[10]

            if table_name not in tables_dict:
                tables_dict[table_name] = {
                    "name": table_name,
                    "schema_name": schema_name,
                    "description": table_comment,
                    "columns": [],
                    "foreign_keys": {},
                }

            column_info: dict[str, Any] = {
                "name": column_name,
                "data_type": data_type.upper() if data_type else "UNKNOWN",
                "nullable": bool(is_nullable),
                "primary_key": bool(is_pk),
            }

            if col_comment:
                column_info["description"] = col_comment

            if fk_ref_table and fk_ref_column:
                fk_key = f"{fk_ref_table}.{fk_ref_column}"
                column_info["foreign_key"] = fk_key
                tables_dict[table_name]["foreign_keys"][column_name] = fk_key

            # Avoid duplicate columns (can happen when a column participates
            # in multiple FK constraints)
            existing_cols = {c["name"] for c in tables_dict[table_name]["columns"]}
            if column_name not in existing_cols:
                tables_dict[table_name]["columns"].append(column_info)

        # Convert to list format
        tables_list = []
        for table_data in tables_dict.values():
            table_data.pop("foreign_keys", None)
            tables_list.append(table_data)

        schema_def = {
            "database_name": "reflected_db",
            "dialect": "postgresql",
            "tables": tables_list,
        }

        logger.info(
            "Schema reflection complete",
            table_count=len(tables_list),
            tables=[t["name"] for t in tables_list],
        )

        return schema_def

    async def switch_engine(
        self,
        new_url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 300,
    ) -> None:
        """Switch the database connection string, actively disposing the old pool.

        Args:
            new_url: The new database connection string.
            echo: Set to True to print executed SQL statements to logs.
            pool_size: Engine connection pool size.
            max_overflow: Max number of connections allowed in overflow.
            pool_timeout: Timeout in seconds for obtaining a connection.
            pool_recycle: Connection recycle time in seconds.
        """
        # Ensure an async driver
        new_url = to_async_database_url(new_url)

        # Sanitize asyncpg parameters
        import re
        if "asyncpg" in new_url:
            new_url = re.sub(r"([?&])channel_binding=[^&]*", r"\1", new_url)
            new_url = new_url.rstrip("?&")
            if "sslmode=" in new_url:
                new_url = new_url.replace("sslmode=", "ssl=")

        logger.info("Actively closing and disposing old database engine connection pool")
        old_engine = self._engine

        # Create the new engine
        self._engine = create_async_engine(
            new_url,
            echo=echo,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            connect_args={"prepared_statement_cache_size": 0},
        )

        # Reconfigure sessionmaker to bind to the new engine
        self._session_factory.configure(bind=self._engine)

        # Dispose the old engine to actively close all connection pools
        await old_engine.dispose()

        logger.info(
            "Database engine switched successfully and old pool reclaimed",
            url=self._sanitise_url(new_url),
        )

    async def dispose(self) -> None:
        """Close all connections in the pool."""
        await self._engine.dispose()

    @staticmethod
    def _sanitise_url(url: str) -> str:
        """Remove credentials from the URL for safe logging."""
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            sanitised = parsed._replace(netloc=parsed.hostname or "")
            return urlunparse(sanitised)
        except Exception:
            return "<url>"
