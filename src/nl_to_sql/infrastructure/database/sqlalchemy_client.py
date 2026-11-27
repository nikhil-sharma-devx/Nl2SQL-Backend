"""Async SQLAlchemy client — manages the target database connection."""

import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.core.exceptions import DatabaseExecutionError
from nl_to_sql.infrastructure.database.url_utils import (
    asyncpg_connect_args,
    sanitize_url_for_logging,
    to_async_database_url,
)
from nl_to_sql.infrastructure.observability.tracing import set_span_attribute, trace_function

logger = structlog.get_logger(__name__)

# Medium: the db.statement tracing span recorded the full raw SQL text,
# including any literal values the LLM copied straight from the user's
# question (names, emails, etc.), with no redaction before export to the
# tracing backend. Replace string/numeric literals with placeholders —
# structure (which tables/columns/clauses were used) stays visible for
# debugging, values don't.
_STRING_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.)*'")
_NUMERIC_LITERAL_RE = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)")


def _redact_sql_for_tracing(sql: str) -> str:
    """Replace string/numeric literals in ``sql`` with placeholders for tracing spans."""
    redacted = _STRING_LITERAL_RE.sub("'?'", sql)
    return _NUMERIC_LITERAL_RE.sub("?", redacted)


# PostgreSQL's own catalogs: never user tables, never listed or reflected.
SYSTEM_SCHEMAS = frozenset({"pg_catalog", "information_schema", "pg_toast"})
_USER_SCHEMA_FILTER_SQL = (
    "ns.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast') "
    "AND ns.nspname NOT LIKE 'pg\\_temp\\_%' "
    "AND ns.nspname NOT LIKE 'pg\\_toast\\_temp\\_%'"
)
# MySQL's system databases and SQLite's temp schema: never user tables.
_GENERIC_SYSTEM_SCHEMAS = frozenset(
    {"information_schema", "mysql", "performance_schema", "sys", "temp"}
)
# r = table, p = partitioned table, v = view, m = materialized view, f = foreign table
_RELKINDS_SQL = "('r', 'p', 'v', 'm', 'f')"


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
            connect_args=asyncpg_connect_args(database_url),
        )
        self._install_readonly_hooks()
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

    def _install_readonly_hooks(self) -> None:
        """SQLite: make every pooled connection read-only (``PRAGMA query_only``).

        SQLite has no per-transaction READ ONLY, so the guard is set once per
        DBAPI connection as the pool opens it. PostgreSQL and MySQL are guarded
        per transaction in ``_apply_session_guards``.
        """
        if not self._readonly or self._engine.dialect.name != "sqlite":
            return
        from sqlalchemy import event

        @event.listens_for(self._engine.sync_engine, "connect")
        def _query_only(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA query_only = ON")
            cur.close()

    async def _apply_session_guards(self, sess: AsyncSession) -> None:
        """Harden a session before running generated SQL.

        - Statement timeout keeps a runaway query from pinning a pool
          connection indefinitely.
        - READ ONLY makes the transaction reject any write even if a
          non-SELECT slipped past the validator (belt-and-suspenders).

        PostgreSQL: ``SET LOCAL statement_timeout`` + ``SET TRANSACTION READ ONLY``.
        MySQL: ``max_execution_time`` (SELECTs) + ``SET TRANSACTION READ ONLY``
        (applies to the transaction the next statement opens).
        SQLite: read-only per connection (see ``_install_readonly_hooks``).
        """
        from sqlalchemy import text

        dialect = self._engine.dialect.name
        if dialect == "postgresql":
            await sess.execute(text(f"SET LOCAL statement_timeout = {self._statement_timeout_ms}"))
            if self._readonly:
                await sess.execute(text("SET TRANSACTION READ ONLY"))
        elif dialect == "mysql":
            await sess.execute(text(f"SET SESSION max_execution_time = {self._statement_timeout_ms}"))
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

    @asynccontextmanager
    async def advisory_xact_lock(self, name: str) -> AsyncGenerator[bool, None]:
        """Hold a PostgreSQL advisory lock for the duration of the ``async with`` block.

        Yields ``True`` when this caller owns the lock (or locking is unavailable,
        in which case the work proceeds unguarded, as before) and ``False`` when
        another worker already holds it.

        Uses a *transaction-scoped* lock inside a transaction kept open for the
        whole block. A session-level ``pg_advisory_lock`` / ``unlock`` pair is
        unsafe behind a transaction-mode pooler (Supabase :6543): the unlock can
        land on a different server connection, leaving the lock stuck on a
        shared backend. The transaction lock is released automatically when the
        transaction ends, including if this process crashes.
        """
        from sqlalchemy import text

        if self._engine.dialect.name != "postgresql":
            yield True
            return

        conn = None
        trans = None
        acquired = True
        try:
            conn = await self._engine.connect()
            trans = await conn.begin()
            result = await conn.execute(
                text("SELECT pg_try_advisory_xact_lock(hashtext(:name))"), {"name": name}
            )
            acquired = bool(result.scalar())
        except Exception as exc:
            logger.warning("Advisory lock unavailable, proceeding without it", lock=name, error=str(exc))
            await self._close_lock_connection(trans, conn)
            conn = trans = None
            acquired = True

        try:
            yield acquired
        finally:
            await self._close_lock_connection(trans, conn)

    @staticmethod
    async def _close_lock_connection(trans: Any, conn: Any) -> None:
        """End the lock transaction (releasing the lock) and return the connection."""
        try:
            if trans is not None:
                await trans.rollback()
        except Exception as exc:
            logger.warning("Failed to end advisory lock transaction", error=str(exc))
        try:
            if conn is not None:
                await conn.close()
        except Exception as exc:
            logger.warning("Failed to close advisory lock connection", error=str(exc))

    # Batch size for the server-side cursor in execute_sql — small enough to
    # bound memory, large enough to keep round trips reasonable.
    _EXECUTE_BATCH_SIZE = 1_000

    @trace_function("database.execute_sql")
    async def execute_sql(self, sql: str) -> list[dict[str, Any]]:
        """Execute a raw SQL string and return rows as a list of dicts.

        Args:
            sql: The SQL string to run against the target database.

        Returns:
            A list of row dictionaries, capped at ``MAX_ROWS``.

        Raises:
            DatabaseExecutionError: On any DB error.
        """
        from sqlalchemy import text

        set_span_attribute("db.statement", _redact_sql_for_tracing(sql))

        try:
            async with self.session() as sess:
                await self._apply_session_guards(sess)
                # H6: stream via a server-side cursor instead of
                # result.fetchall(), which pulled an unbounded LLM-generated
                # result set (no LIMIT) fully into memory before truncating.
                stream_result = await sess.stream(text(sql))
                columns = list(stream_result.keys())
                rows: list[dict[str, Any]] = []
                truncated = False
                async for partition in stream_result.partitions(self._EXECUTE_BATCH_SIZE):
                    for row in partition:
                        if len(rows) >= self.MAX_ROWS:
                            truncated = True
                            break
                        rows.append(dict(zip(columns, row, strict=True)))
                    if truncated:
                        break
                set_span_attribute("db.rows_returned", len(rows))
                set_span_attribute("db.truncated", truncated)
                return rows
        except Exception as exc:
            logger.error("SQL execution failed", error=str(exc))
            raise DatabaseExecutionError(f"Failed to execute SQL: {exc}", detail=str(exc)) from exc

    @trace_function("database.explain")
    async def explain(self, sql: str, *, analyze: bool = False) -> dict[str, Any]:
        """Run ``EXPLAIN`` for a query-cost / row-count preview (PostgreSQL only).

        Never executes the statement unless ``analyze=True`` on a read query.
        Write/DDL statements are detected and ANALYZE is refused for them, so a
        preview can never mutate data.

        Args:
            sql: The SQL statement to preview.
            analyze: When True (and the statement is a read), run
                ``EXPLAIN (ANALYZE, FORMAT JSON)`` — actually executing the read.

        Returns:
            A dict with ``supported`` (bool), ``estimated_rows`` (int | None),
            ``estimated_cost`` (float | None), ``plan`` (simplified tree | None),
            ``warnings`` (list of {type, message}) and ``message`` (str | None).
            Never raises to the caller — unsupported dialects / errors return a
            graceful ``supported=False`` result.
        """
        import json

        from sqlalchemy import text

        from nl_to_sql.infrastructure.database.explain_utils import (
            is_write_statement,
            parse_explain_plan,
        )

        dialect = self._engine.dialect.name
        if dialect != "postgresql":
            logger.info("explain: unsupported dialect", dialect=dialect)
            return {
                "supported": False,
                "estimated_rows": None,
                "estimated_cost": None,
                "plan": None,
                "warnings": [],
                "message": (
                    "Query preview is only supported on PostgreSQL databases "
                    f"(current dialect: {dialect})."
                ),
            }

        # Never execute (ANALYZE) a write statement — plan it only.
        if analyze and is_write_statement(sql):
            logger.warning("explain: refusing ANALYZE on a write statement — plain EXPLAIN only")
            analyze = False

        options = "ANALYZE, FORMAT JSON" if analyze else "FORMAT JSON"
        explain_sql = f"EXPLAIN ({options}) {sql}"
        set_span_attribute("db.statement", _redact_sql_for_tracing(explain_sql))

        try:
            async with self.session() as sess:
                await self._apply_session_guards(sess)
                result = await sess.execute(text(explain_sql))
                row = result.fetchone()
        except Exception as exc:
            logger.warning("explain failed", error=str(exc))
            return {
                "supported": False,
                "estimated_rows": None,
                "estimated_cost": None,
                "plan": None,
                "warnings": [],
                "message": f"Could not generate a preview for this query: {exc}",
            }

        if row is None:
            return {
                "supported": False,
                "estimated_rows": None,
                "estimated_cost": None,
                "plan": None,
                "warnings": [],
                "message": "The database returned no query plan.",
            }

        raw = row[0]
        try:
            plan_json = json.loads(raw) if isinstance(raw, str) else raw
            parsed: dict[str, Any] = parse_explain_plan(plan_json)
        except Exception as exc:
            logger.warning("explain: failed to parse plan", error=str(exc))
            return {
                "supported": False,
                "estimated_rows": None,
                "estimated_cost": None,
                "plan": None,
                "warnings": [],
                "message": f"Could not parse the query plan: {exc}",
            }

        parsed["supported"] = True
        parsed["message"] = None
        logger.info(
            "explain complete",
            estimated_rows=parsed["estimated_rows"],
            estimated_cost=parsed["estimated_cost"],
            warnings=len(parsed["warnings"]),
        )
        return parsed

    async def execute_sql_stream(
        self, sql: str, batch_size: int = 100
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
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

        set_span_attribute("db.statement", _redact_sql_for_tracing(sql))

        try:
            async with self._session_factory() as sess:
                await self._apply_session_guards(sess)
                stream_result = await sess.stream(text(sql))
                columns = list(stream_result.keys())
                async for partition in stream_result.partitions(batch_size):
                    yield [dict(zip(columns, row, strict=True)) for row in partition]
        except Exception as exc:
            logger.error("SQL stream execution failed", error=str(exc))
            raise DatabaseExecutionError(f"Failed to stream SQL: {exc}", detail=str(exc)) from exc

    async def health_check(self) -> bool:
        """Run a trivial query to verify the connection."""
        from sqlalchemy import text

        try:
            async with self.session() as sess:
                await sess.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def list_schemas(self) -> list[dict[str, Any]]:
        """List the database's schemas with how many queryable relations each has.

        Every schema except PostgreSQL's own catalogs (``SYSTEM_SCHEMAS`` and
        ``pg_temp_*`` / ``pg_toast_temp_*``). Counts tables, partitioned
        tables, views, materialized views and foreign tables; partitions of a
        partitioned table are not counted separately. Read-only transaction.
        """
        from sqlalchemy import text

        if self._engine.dialect.name != "postgresql":
            return await self._list_schemas_generic()

        query = f"""
        SELECT ns.nspname AS schema_name, count(cls.oid) AS table_count
        FROM pg_catalog.pg_namespace ns
        LEFT JOIN pg_catalog.pg_class cls
            ON cls.relnamespace = ns.oid
           AND cls.relkind IN {_RELKINDS_SQL}
           AND NOT cls.relispartition
        WHERE {_USER_SCHEMA_FILTER_SQL}
        GROUP BY ns.nspname
        ORDER BY ns.nspname
        """  # noqa: S608 - only module constants interpolated; no user input
        async with self.session() as sess:
            await sess.execute(text("SET LOCAL statement_timeout = '30s'"))
            if self._readonly:
                await sess.execute(text("SET TRANSACTION READ ONLY"))
            rows = (await sess.execute(text(query))).fetchall()
        return [{"name": r[0], "table_count": int(r[1])} for r in rows]

    async def reflect_schema(
        self, schema_name: str = "public", *, schemas: list[str] | None = None
    ) -> dict[str, Any]:
        """Reflect database schema from live PostgreSQL database.

        Uses pg_catalog system tables directly (instead of the slow
        information_schema views) for speed on cloud-hosted pooled
        connections such as Neon.

        Args:
            schema_name: The PostgreSQL schema to reflect (default: 'public').
            schemas: Reflect several schemas in one pass instead (overrides
                ``schema_name``). Each table carries its own ``schema_name``.

        Reflects ordinary and partitioned tables, views, materialized views and
        foreign tables (not individual partitions). A foreign key to a table
        outside ``public`` is written ``schema.table.column``.

        Returns:
            Dictionary with database_name, dialect, and tables list.
        """
        from sqlalchemy import text

        targets = list(dict.fromkeys(schemas)) if schemas else [schema_name]
        logger.info("Reflecting schema from live database", schemas=targets)
        if self._engine.dialect.name != "postgresql":
            return await self._reflect_generic(targets if schemas else None)

        # ── Fast pg_catalog query ──────────────────────────────────────────────
        # information_schema views are themselves complex multi-join views that
        # time out on Neon pooled endpoints.  Querying pg_catalog directly is
        # orders of magnitude faster because the tables are real system tables
        # with indexed OID look-ups.
        schema_query = f"""
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
            col_description(cls.oid, att.attnum)           AS column_comment,
            ns.nspname                           AS table_schema,
            fk_ns.nspname                        AS fk_schema
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
        LEFT JOIN pg_catalog.pg_namespace fk_ns
            ON fk_ns.oid = fk_cls.relnamespace
        LEFT JOIN pg_catalog.pg_attribute fk_att
            ON fk_att.attrelid = fk_map.confrelid
           AND fk_att.attnum = fk_map.ref_attnum
        WHERE ns.nspname = ANY(:schemas)
          AND cls.relkind IN {_RELKINDS_SQL}
          AND NOT cls.relispartition  -- a partitioned table, not each partition
        ORDER BY ns.nspname, cls.relname, att.attnum
        """  # noqa: S608 - only module constants interpolated; schemas are a bound param

        async with self.session() as sess:
            # Safety net: prevent indefinite hangs
            await sess.execute(text("SET LOCAL statement_timeout = '90s'"))
            # Reflection only reads pg_catalog, but enforce it at the database
            # too: a READ ONLY transaction rejects any write outright. Must come
            # before the first query of the transaction.
            if self._readonly:
                await sess.execute(text("SET TRANSACTION READ ONLY"))

            result = await sess.execute(text(schema_query), {"schemas": targets})
            rows = result.fetchall()

        # ── Build schema structure ─────────────────────────────────────────────
        tables_dict: dict[tuple[str, str], dict[str, Any]] = {}

        for row in rows:
            table_name = row[0]
            column_name = row[1]
            data_type = row[2]
            is_nullable = row[3]
            is_pk = row[6]
            fk_ref_table = row[7]
            fk_ref_column = row[8]
            table_comment = row[9]
            col_comment = row[10]
            table_schema = row[11]
            fk_schema = row[12]
            key = (table_schema, table_name)

            if key not in tables_dict:
                tables_dict[key] = {
                    "name": table_name,
                    "schema_name": table_schema,
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
                ref = (
                    fk_ref_table
                    if fk_schema in (None, "", "public")
                    else f"{fk_schema}.{fk_ref_table}"
                )
                fk_key = f"{ref}.{fk_ref_column}"
                column_info["foreign_key"] = fk_key
                tables_dict[key]["foreign_keys"][column_name] = fk_key

            # Avoid duplicate columns (can happen when a column participates
            # in multiple FK constraints)
            existing_cols = {c["name"] for c in tables_dict[key]["columns"]}
            if column_name not in existing_cols:
                tables_dict[key]["columns"].append(column_info)

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

    # ── MySQL / SQLite: SQLAlchemy inspector ─────────────────────────────────

    async def _run_inspector(self, fn: Any) -> Any:
        """Run ``fn(inspector)`` on one guarded (read-only) connection."""
        from sqlalchemy import inspect

        async with self._engine.connect() as conn:
            if self._engine.dialect.name == "mysql":
                from sqlalchemy import text

                await conn.execute(text(f"SET SESSION max_execution_time = {self._statement_timeout_ms}"))
                if self._readonly:
                    await conn.execute(text("SET TRANSACTION READ ONLY"))
            return await conn.run_sync(lambda sync_conn: fn(inspect(sync_conn)))

    async def _list_schemas_generic(self) -> list[dict[str, Any]]:
        def collect(insp: Any) -> list[dict[str, Any]]:
            out = []
            for name in insp.get_schema_names():
                if name.lower() in _GENERIC_SYSTEM_SCHEMAS:
                    continue
                count = len(insp.get_table_names(schema=name)) + len(insp.get_view_names(schema=name))
                out.append({"name": name, "table_count": count})
            return sorted(out, key=lambda x: x["name"])

        result: list[dict[str, Any]] = await self._run_inspector(collect)
        return result

    async def _reflect_generic(self, schemas: list[str] | None) -> dict[str, Any]:
        """Tables + views with columns, primary and foreign keys (MySQL, SQLite).

        Tables keep their real schema (MySQL database / SQLite ``main``) and are
        always written ``schema.table`` downstream, which is valid SQL in both.
        ``schemas=None`` means the connection's default schema.
        """
        dialect = self._engine.dialect.name

        def collect(insp: Any) -> list[dict[str, Any]]:
            targets = schemas or [insp.default_schema_name]
            tables: list[dict[str, Any]] = []
            for schema in targets:
                names = list(insp.get_table_names(schema=schema)) + list(insp.get_view_names(schema=schema))
                for name in names:
                    try:
                        pk_cols = set(insp.get_pk_constraint(name, schema=schema).get("constrained_columns") or [])
                    except Exception:  # views have none
                        pk_cols = set()
                    fks: dict[str, str] = {}
                    try:
                        for fk in insp.get_foreign_keys(name, schema=schema):
                            ref_schema = fk.get("referred_schema") or schema
                            for local, remote in zip(
                                fk.get("constrained_columns") or [], fk.get("referred_columns") or [], strict=False
                            ):
                                fks[local] = f"{ref_schema}.{fk['referred_table']}.{remote}"
                    except Exception:
                        pass
                    try:
                        comment = (insp.get_table_comment(name, schema=schema) or {}).get("text")
                    except Exception:  # not supported by every dialect
                        comment = None
                    columns = []
                    for col in insp.get_columns(name, schema=schema):
                        info: dict[str, Any] = {
                            "name": col["name"],
                            "data_type": str(col["type"]).upper(),
                            "nullable": bool(col.get("nullable", True)),
                            "primary_key": col["name"] in pk_cols,
                        }
                        if col.get("comment"):
                            info["description"] = col["comment"]
                        if col["name"] in fks:
                            info["foreign_key"] = fks[col["name"]]
                        columns.append(info)
                    tables.append(
                        {"name": name, "schema_name": schema, "description": comment, "columns": columns}
                    )
            return tables

        tables = await self._run_inspector(collect)
        logger.info("Schema reflection complete", dialect=dialect, tables=len(tables))
        return {"database_name": "reflected_db", "dialect": dialect, "tables": tables}

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
            connect_args=asyncpg_connect_args(new_url),
        )

        self._install_readonly_hooks()
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
        return sanitize_url_for_logging(url)
