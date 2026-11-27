"""Unit tests for AsyncDatabaseClient.execute_sql streaming (H6).

Regression test: ``execute_sql`` used to call ``result.fetchall()`` *before*
truncating to ``MAX_ROWS`` — an unbounded LLM-generated query (no LIMIT)
could pull an arbitrarily large result set fully into memory. The fix
streams via a server-side cursor (``sess.stream().partitions()``) and stops
as soon as ``MAX_ROWS`` rows have been collected. The fake session/result
below deliberately has no ``execute``/``fetchall`` methods at all — if the
implementation ever regresses back to that pattern, these tests fail with
an AttributeError instead of silently passing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient


class _FakeStreamResult:
    """Emulates SQLAlchemy's async streaming Result — partitions only, no fetchall."""

    def __init__(self, columns: list[str], rows: list[tuple], partition_size: int) -> None:
        self._columns = columns
        self._rows = rows
        self._partition_size = partition_size

    def keys(self) -> list[str]:
        return self._columns

    async def partitions(self, size: int):
        step = self._partition_size or size
        for i in range(0, len(self._rows), step):
            yield self._rows[i : i + step]


class _FakeSession:
    def __init__(self, stream_result: _FakeStreamResult) -> None:
        self._stream_result = stream_result
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    async def stream(self, _stmt: object) -> _FakeStreamResult:
        return self._stream_result

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _client_with_rows(rows: list[tuple], partition_size: int = 1000) -> AsyncDatabaseClient:
    # The engine is never actually used (session_factory is overridden below) —
    # patch it out so construction doesn't need a real, connectable database.
    with patch(
        "nl_to_sql.infrastructure.database.sqlalchemy_client.create_async_engine"
    ):
        client = AsyncDatabaseClient(database_url="postgresql+asyncpg://u:p@h/db")
    stream_result = _FakeStreamResult(["id"], rows, partition_size)
    session = _FakeSession(stream_result)
    client._session_factory = lambda: session
    return client


class TestExecuteSqlStreaming:
    async def test_result_under_max_rows_returned_in_full(self) -> None:
        client = _client_with_rows([(i,) for i in range(10)])
        rows = await client.execute_sql("SELECT id FROM t")
        assert len(rows) == 10
        assert rows[0] == {"id": 0}

    async def test_result_over_max_rows_is_truncated(self) -> None:
        client = _client_with_rows([(i,) for i in range(AsyncDatabaseClient.MAX_ROWS + 500)])
        rows = await client.execute_sql("SELECT id FROM t")
        assert len(rows) == AsyncDatabaseClient.MAX_ROWS

    async def test_never_calls_fetchall_or_plain_execute(self) -> None:
        """If execute_sql regresses to session.execute()+fetchall(), this
        fake would raise AttributeError instead of silently buffering
        everything — asserting no exception here IS the regression guard."""
        client = _client_with_rows([(i,) for i in range(50)])
        rows = await client.execute_sql("SELECT id FROM t")
        assert len(rows) == 50

    async def test_partitioned_across_many_small_batches_still_correct(self) -> None:
        """Simulates a server-side cursor yielding many small partitions —
        the truncation must stop mid-partition, not just at partition boundaries."""
        client = _client_with_rows(
            [(i,) for i in range(AsyncDatabaseClient.MAX_ROWS + 7)], partition_size=17
        )
        rows = await client.execute_sql("SELECT id FROM t")
        assert len(rows) == AsyncDatabaseClient.MAX_ROWS


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
