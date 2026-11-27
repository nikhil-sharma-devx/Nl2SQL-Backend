"""Unit tests for the query cost / row-count preview.

Covers the pure plan-parsing helpers (``explain_utils``) and the DB client's
``explain`` method (write-statement / non-PostgreSQL / error handling).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nl_to_sql.infrastructure.database.explain_utils import (
    is_write_statement,
    parse_explain_plan,
)
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient

# ── Sample EXPLAIN (FORMAT JSON) outputs ────────────────────────────────────────

SEQ_SCAN_PLAN = [
    {
        "Plan": {
            "Node Type": "Seq Scan",
            "Relation Name": "customers",
            "Total Cost": 431.0,
            "Plan Rows": 10000,
        }
    }
]

INDEX_PLAN = [
    {
        "Plan": {
            "Node Type": "Index Scan",
            "Relation Name": "orders",
            "Total Cost": 8.3,
            "Plan Rows": 1,
        }
    }
]

EXPENSIVE_JOIN_PLAN = [
    {
        "Plan": {
            "Node Type": "Nested Loop",
            "Total Cost": 55000.0,
            "Plan Rows": 250000,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "orders",
                    "Total Cost": 100.0,
                    "Plan Rows": 5000,
                },
                {
                    "Node Type": "Index Scan",
                    "Relation Name": "customers",
                    "Total Cost": 5.0,
                    "Plan Rows": 1,
                },
            ],
        }
    }
]


# ── parse_explain_plan ──────────────────────────────────────────────────────────


def test_parse_extracts_rows_and_cost() -> None:
    parsed = parse_explain_plan(SEQ_SCAN_PLAN)
    assert parsed["estimated_rows"] == 10000
    assert parsed["estimated_cost"] == 431.0
    assert parsed["plan"]["node_type"] == "Seq Scan"


def test_seq_scan_produces_warning() -> None:
    parsed = parse_explain_plan(SEQ_SCAN_PLAN)
    types = {w["type"] for w in parsed["warnings"]}
    assert "seq_scan" in types
    assert "customers" in parsed["warnings"][0]["message"]


def test_index_scan_has_no_warnings() -> None:
    parsed = parse_explain_plan(INDEX_PLAN)
    assert parsed["warnings"] == []


def test_expensive_join_and_nested_seq_scan_warn() -> None:
    parsed = parse_explain_plan(EXPENSIVE_JOIN_PLAN)
    types = {w["type"] for w in parsed["warnings"]}
    assert "expensive_join" in types
    assert "seq_scan" in types  # nested child seq scan is also flagged


def test_parse_handles_empty_and_malformed() -> None:
    assert parse_explain_plan([])["plan"] is None
    assert parse_explain_plan({})["estimated_rows"] == 0
    assert parse_explain_plan(None)["warnings"] == []


# ── is_write_statement ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "  update t set x = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "-- a comment\nTRUNCATE t",
        "/* block */ CREATE TABLE t (id int)",
    ],
)
def test_write_statements_detected(sql: str) -> None:
    assert is_write_statement(sql) is True


@pytest.mark.parametrize(
    "sql",
    ["SELECT * FROM t", "  select 1", "WITH x AS (SELECT 1) SELECT * FROM x", ""],
)
def test_read_statements_not_write(sql: str) -> None:
    assert is_write_statement(sql) is False


# ── AsyncDatabaseClient.explain ─────────────────────────────────────────────────


def _make_client(dialect_name: str) -> AsyncDatabaseClient:
    with patch(
        "nl_to_sql.infrastructure.database.sqlalchemy_client.create_async_engine"
    ) as mk:
        engine = MagicMock()
        engine.dialect.name = dialect_name
        mk.return_value = engine
        return AsyncDatabaseClient(database_url="postgresql://u:p@h:5432/db")


def _stub_session(client: AsyncDatabaseClient, fake_sess: AsyncMock) -> None:
    @asynccontextmanager
    async def _session():  # type: ignore[no-untyped-def]
        yield fake_sess

    client.session = _session  # type: ignore[method-assign]
    client._apply_session_guards = AsyncMock()  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_explain_non_postgres_is_graceful() -> None:
    client = _make_client("sqlite")
    out = await client.explain("SELECT 1")
    assert out["supported"] is False
    assert "PostgreSQL" in out["message"]
    assert out["estimated_rows"] is None


@pytest.mark.asyncio
async def test_explain_parses_postgres_plan() -> None:
    client = _make_client("postgresql")
    result_mock = MagicMock()
    result_mock.fetchone = MagicMock(return_value=[SEQ_SCAN_PLAN])
    fake_sess = AsyncMock()
    fake_sess.execute = AsyncMock(return_value=result_mock)
    _stub_session(client, fake_sess)

    out = await client.explain("SELECT * FROM customers")
    assert out["supported"] is True
    assert out["estimated_rows"] == 10000
    assert any(w["type"] == "seq_scan" for w in out["warnings"])


@pytest.mark.asyncio
async def test_explain_refuses_analyze_on_write() -> None:
    client = _make_client("postgresql")
    captured: dict[str, str] = {}
    result_mock = MagicMock()
    result_mock.fetchone = MagicMock(return_value=[INDEX_PLAN])

    async def _capture(stmt, *a, **k):  # type: ignore[no-untyped-def]
        captured["sql"] = str(stmt)
        return result_mock

    fake_sess = AsyncMock()
    fake_sess.execute = _capture
    _stub_session(client, fake_sess)

    await client.explain("DELETE FROM customers", analyze=True)
    assert "ANALYZE" not in captured["sql"].upper()


@pytest.mark.asyncio
async def test_explain_error_is_graceful() -> None:
    client = _make_client("postgresql")

    @asynccontextmanager
    async def _boom():  # type: ignore[no-untyped-def]
        raise RuntimeError("connection refused")
        yield  # pragma: no cover

    client.session = _boom  # type: ignore[method-assign]
    client._apply_session_guards = AsyncMock()  # type: ignore[method-assign]

    out = await client.explain("SELECT 1")
    assert out["supported"] is False
    assert "connection refused" in out["message"]
