"""reflect_schema runs inside a READ ONLY transaction (set before its first query)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient


async def test_reflection_is_read_only_before_querying() -> None:
    client = AsyncDatabaseClient.__new__(AsyncDatabaseClient)
    client._readonly = True
    statements: list[str] = []

    class FakeSession:
        async def execute(self, stmt, params=None):
            statements.append(str(stmt).strip().split("\n")[0])
            result = MagicMock()
            result.fetchall.return_value = []
            return result

    @asynccontextmanager
    async def session():
        yield FakeSession()

    client.session = session  # type: ignore[method-assign]
    client._engine = MagicMock()
    client._engine.dialect.name = "postgresql"
    try:
        await client.reflect_schema("public")
    except Exception:
        pass  # building the (empty) result may need more engine state; order is what matters

    assert statements[:2] == ["SET LOCAL statement_timeout = '90s'", "SET TRANSACTION READ ONLY"]
    assert len(statements) >= 3  # the catalog query came after the guard
