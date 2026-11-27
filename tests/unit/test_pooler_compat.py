"""Transaction-pooler compatibility: asyncpg connect args and advisory locks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.infrastructure.database.url_utils import asyncpg_connect_args

_PG_URL = "postgresql+asyncpg://u:p@pooler.example.com:6543/postgres"
_ENGINE_TARGET = "nl_to_sql.infrastructure.database.sqlalchemy_client.create_async_engine"


def test_asyncpg_args_disable_caches_and_name_statements_uniquely() -> None:
    args = asyncpg_connect_args(_PG_URL, command_timeout=30)

    assert args["statement_cache_size"] == 0
    assert args["prepared_statement_cache_size"] == 0
    assert args["command_timeout"] == 30
    name_func = args["prepared_statement_name_func"]
    assert name_func() != name_func()


def test_non_asyncpg_url_gets_only_extra_args() -> None:
    assert asyncpg_connect_args("mysql+aiomysql://u:p@h/db", connect_timeout=5) == {
        "connect_timeout": 5
    }


def test_db_client_engine_uses_pooler_safe_args() -> None:
    with patch(_ENGINE_TARGET) as create_engine:
        AsyncDatabaseClient(database_url=_PG_URL)

    connect_args = create_engine.call_args.kwargs["connect_args"]
    assert connect_args["statement_cache_size"] == 0
    assert "prepared_statement_name_func" in connect_args


def _client_with_lock_result(*, acquired: bool | None, fail: bool = False):
    """Build a client whose engine returns ``acquired`` from the lock query."""
    conn = MagicMock()
    trans = MagicMock()
    trans.rollback = AsyncMock()
    conn.begin = AsyncMock(return_value=trans)
    conn.close = AsyncMock()
    if fail:
        conn.execute = AsyncMock(side_effect=OSError("network down"))
    else:
        result = MagicMock()
        result.scalar.return_value = acquired
        conn.execute = AsyncMock(return_value=result)

    engine = MagicMock()
    engine.dialect.name = "postgresql"
    engine.connect = AsyncMock(return_value=conn)
    with patch(_ENGINE_TARGET, return_value=engine):
        client = AsyncDatabaseClient(database_url=_PG_URL)
    return client, conn, trans


@pytest.mark.asyncio
@pytest.mark.parametrize("acquired", [True, False])
async def test_advisory_lock_holds_transaction_until_block_exits(acquired: bool) -> None:
    client, conn, trans = _client_with_lock_result(acquired=acquired)

    async with client.advisory_xact_lock("job") as got:
        assert got is acquired
        sql = str(conn.execute.await_args.args[0])
        assert "pg_try_advisory_xact_lock" in sql
        trans.rollback.assert_not_awaited()  # lock still held inside the block

    trans.rollback.assert_awaited_once()  # ending the transaction releases it
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_advisory_lock_releases_when_body_raises() -> None:
    client, conn, trans = _client_with_lock_result(acquired=True)

    with pytest.raises(ValueError):
        async with client.advisory_xact_lock("job"):
            raise ValueError("ingest failed")

    trans.rollback.assert_awaited_once()
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_advisory_lock_failure_proceeds_unguarded() -> None:
    client, conn, _trans = _client_with_lock_result(acquired=None, fail=True)

    async with client.advisory_xact_lock("job") as got:
        assert got is True

    conn.close.assert_awaited_once()
