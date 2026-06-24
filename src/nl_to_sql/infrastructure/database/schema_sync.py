"""Idempotent schema reconciliation for the application metadata database.

SQLAlchemy's ``Base.metadata.create_all`` creates *absent* tables, but it never
alters tables that already exist. So any column added to an ORM model after its
table was first created never reaches the database — producing runtime errors
such as ``column chat_sessions.user_id does not exist``.

``ensure_schema`` closes that gap: it creates missing tables and then inspects
the live database, issuing ``ALTER TABLE ... ADD COLUMN`` for every column that
exists on the ORM model but is missing in the database. It only ever **adds**
columns (never drops or retypes), so it is safe to run on every startup.
"""
from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import Column, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from nl_to_sql.infrastructure.database.models import Base

logger = structlog.get_logger(__name__)


def _literal_default(column: Column[Any]) -> str | None:
    """Return a SQL literal for a NOT NULL column's default, or None if not derivable."""
    server_default = column.server_default
    if server_default is not None:
        arg = getattr(server_default, "arg", None)
        text_attr = getattr(arg, "text", None)
        if text_attr is not None:
            return str(text_attr)

    default = column.default
    if default is not None and getattr(default, "is_scalar", False):
        arg = getattr(default, "arg", None)
        if arg is None or callable(arg):
            return None
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        if isinstance(arg, str):
            escaped = arg.replace("'", "''")
            return f"'{escaped}'"
    return None


def _reconcile_columns(connection: Connection) -> None:
    """Add any model columns that are missing from existing DB tables."""
    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    preparer = connection.dialect.identifier_preparer

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            # create_all (run just before this) already created it in full.
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue

            column_type = column.type.compile(dialect=connection.dialect)
            qualified_table = preparer.format_table(table)
            quoted_column = preparer.quote(column.name)
            clause = f"ALTER TABLE {qualified_table} ADD COLUMN {quoted_column} {column_type}"

            # Only enforce NOT NULL when we can supply a safe default for existing
            # rows; otherwise add the column as nullable to avoid failing on
            # already-populated tables.
            if not column.nullable:
                default_literal = _literal_default(column)
                if default_literal is not None:
                    clause += f" NOT NULL DEFAULT {default_literal}"

            connection.execute(text(clause))
            logger.info(
                "schema_sync: added missing column",
                table=table.name,
                column=column.name,
                type=str(column_type),
            )


async def ensure_schema(engine: AsyncEngine) -> None:
    """Create missing tables and add any missing columns. Idempotent and safe."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_reconcile_columns)
