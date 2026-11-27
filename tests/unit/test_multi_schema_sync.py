"""Syncing a connection whose tables live in several PostgreSQL schemas.

Regression: sync only reflected ``public`` (and only plain tables), so a
database whose tables live in other schemas synced "0 tables". Now a sync can
cover many schemas, tables keep their schema, the LLM sees ``schema.table``
outside ``public``, and the column validator understands qualified names.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.core.models.schema import ColumnInfo, TableInfo
from nl_to_sql.infrastructure.database.models import Base, UserSchemaTable
from nl_to_sql.rag.ingestion.doc_builder import DocBuilder
from nl_to_sql.services.schema_catalog_service import SchemaCatalogService
from nl_to_sql.services.sql_column_validator import SQLColumnValidator


def _table(schema: str, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "schema_name": schema,
        "description": None,
        "columns": [{"name": "id", "data_type": "INTEGER", "nullable": False, "primary_key": True}],
    }


class FakeDB:
    """Stands in for AsyncDatabaseClient: reflects whatever ``live`` holds."""

    def __init__(self, live: list[dict[str, Any]]) -> None:
        self.live = live
        self.calls: list[Any] = []

    async def reflect_schema(self, schema_name: str = "public", *, schemas: list[str] | None = None):
        wanted = set(schemas or [schema_name])
        self.calls.append(sorted(wanted))
        return {
            "database_name": "db",
            "dialect": "postgresql",
            "tables": [t for t in self.live if t["schema_name"] in wanted],
        }


@pytest.fixture
async def catalog() -> AsyncIterator[tuple[SchemaCatalogService, async_sessionmaker[AsyncSession]]]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    ingestion = MagicMock()
    ingestion.ingest = AsyncMock()
    try:
        yield SchemaCatalogService(factory, ingestion), factory
    finally:
        await engine.dispose()
        try:
            os.remove(path)
        except OSError:
            pass


async def _rows(factory: async_sessionmaker[AsyncSession]) -> set[tuple[str, str]]:
    async with factory() as db:
        rows = (await db.execute(select(UserSchemaTable))).scalars().all()
    return {(r.schema_name, r.table_name) for r in rows}


async def test_syncs_tables_from_every_listed_schema(catalog) -> None:
    svc, factory = catalog
    db = FakeDB([_table("public", "users"), _table("sales", "orders"), _table("hr", "orders")])
    result = await svc.sync_from_live_db("u1", "c1", db, schemas=["public", "sales", "hr"])  # type: ignore[arg-type]

    assert result.total_tables == 3
    assert sorted(result.new_tables) == ["hr.orders", "sales.orders", "users"]
    # Same table name in two schemas: both kept.
    assert await _rows(factory) == {("public", "users"), ("sales", "orders"), ("hr", "orders")}
    assert await svc.synced_schemas("c1") == ["hr", "public", "sales"]


async def test_unticked_schema_is_dropped_and_resync_is_stable(catalog) -> None:
    svc, factory = catalog
    db = FakeDB([_table("public", "users"), _table("sales", "orders")])
    await svc.sync_from_live_db("u1", "c1", db, schemas=["public", "sales"])  # type: ignore[arg-type]

    again = await svc.sync_from_live_db("u1", "c1", db, schemas=["public", "sales"])  # type: ignore[arg-type]
    assert again.new_tables == []

    await svc.sync_from_live_db("u1", "c1", db, schemas=["sales"])  # type: ignore[arg-type]
    assert await _rows(factory) == {("sales", "orders")}


async def test_legacy_single_schema_sync_leaves_other_schemas_alone(catalog) -> None:
    svc, factory = catalog
    db = FakeDB([_table("public", "users"), _table("sales", "orders")])
    await svc.sync_from_live_db("u1", "c1", db, schemas=["public", "sales"])  # type: ignore[arg-type]
    await svc.sync_from_live_db("u1", "c1", db, "public")  # type: ignore[arg-type]
    assert await _rows(factory) == {("public", "users"), ("sales", "orders")}


def test_prompt_qualifies_tables_outside_public() -> None:
    col = ColumnInfo(name="id", data_type="INTEGER")
    chunks = DocBuilder.build_chunks(
        [TableInfo(name="users", schema_name="public", columns=[col]),
         TableInfo(name="orders", schema_name="sales", columns=[col])]
    )
    texts = {c.table_name: c.content for c in chunks}
    assert texts["users"].startswith("Table: users\n")
    assert texts["orders"].startswith("Table: sales.orders\n")


def test_column_validator_reads_qualified_table_lines() -> None:
    ctx = SQLColumnValidator().extract_schema_from_context(
        "Table: sales.orders\nColumns:\n  - id (INTEGER)\n  - total (NUMERIC)\n"
    )
    assert ctx == {"orders": ["id", "total"]}
