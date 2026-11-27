"""MySQL/SQLite path of AsyncDatabaseClient, exercised on a real SQLite file.

Schema listing, reflection (tables, views, PK, FK as ``schema.table.column``)
and the read-only guard (PRAGMA query_only) that blocks any write.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Iterator

import pytest

from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient


@pytest.fixture
def db_path() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total REAL
        );
        CREATE VIEW big_orders AS SELECT * FROM orders WHERE total > 100;
        INSERT INTO customers VALUES (1, 'Ana');
        """
    )
    con.commit()
    con.close()
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def test_lists_and_reflects_tables_views_and_keys(db_path: str) -> None:
    client = AsyncDatabaseClient(database_url=f"sqlite+aiosqlite:///{db_path}")
    try:
        schemas = await client.list_schemas()
        assert schemas == [{"name": "main", "table_count": 3}]

        result = await client.reflect_schema(schemas=["main"])
        assert result["dialect"] == "sqlite"
        by_name = {t["name"]: t for t in result["tables"]}
        assert set(by_name) == {"customers", "orders", "big_orders"}
        assert all(t["schema_name"] == "main" for t in result["tables"])
        orders = {c["name"]: c for c in by_name["orders"]["columns"]}
        assert orders["id"]["primary_key"] is True
        assert orders["customer_id"]["foreign_key"] == "main.customers.id"
    finally:
        await client.dispose()


async def test_read_only_blocks_writes(db_path: str) -> None:
    client = AsyncDatabaseClient(database_url=f"sqlite+aiosqlite:///{db_path}")
    try:
        rows = await client.execute_sql("SELECT name FROM customers")
        assert rows == [{"name": "Ana"}]
        with pytest.raises(Exception, match=r"readonly|read-only|query_only|attempt to write"):
            await client.execute_sql("INSERT INTO customers VALUES (2, 'Ben')")
    finally:
        await client.dispose()
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT count(*) FROM customers").fetchone()[0] == 1
    con.close()
