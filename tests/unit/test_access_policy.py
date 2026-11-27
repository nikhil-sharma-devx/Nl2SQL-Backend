"""AccessPolicy + PolicyEnforcingClient — table allow-list and restricted columns.

Security-critical: these tests include the bypass shapes (aliases, CTEs,
subqueries, filter-clause probing, star expansion, unparseable SQL) rather
than only the happy path.
"""

from __future__ import annotations

from typing import Any

import pytest

from nl_to_sql.core.access_policy import REDACTED, AccessPolicy
from nl_to_sql.core.exceptions import AccessPolicyViolationError, DatabaseExecutionError
from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.infrastructure.database.policy_client import PolicyEnforcingClient
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient

POLICY = AccessPolicy.from_config(
    allowed_tables=["customers", "orders", "public.regions"],
    restricted_columns=["customers.email", "Customers.SSN"],
)


class TestFromConfig:
    def test_normalises_names(self) -> None:
        assert POLICY.allowed_tables == frozenset({"customers", "orders", "regions"})
        assert POLICY.restricted == {"customers": frozenset({"email", "ssn"})}

    def test_none_allowed_means_every_table(self) -> None:
        p = AccessPolicy.from_config(None, [])
        assert p.is_empty
        assert p.table_allowed("anything")

    def test_ignores_malformed_restricted_entries(self) -> None:
        p = AccessPolicy.from_config(None, ["nodot", "", ".x", "t."])
        assert p.restricted == {}


class TestViolations:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT name FROM customers",
            "SELECT c.name, o.total FROM customers c JOIN orders o ON o.customer_id = c.id",
            "WITH recent AS (SELECT * FROM orders) SELECT count(*) FROM recent",
            "SELECT r.name FROM regions r",
            "EXPLAIN SELECT name FROM customers",
            "EXPLAIN (FORMAT JSON) SELECT name FROM customers",
        ],
    )
    def test_allowed_queries_pass(self, sql: str) -> None:
        assert POLICY.violations(sql) == []

    def test_disallowed_table(self) -> None:
        problems = POLICY.violations("SELECT * FROM payroll")
        assert problems == ["Table `payroll` is not available on this connection."]

    def test_disallowed_table_in_subquery(self) -> None:
        sql = "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM payroll)"
        assert any("payroll" in p for p in POLICY.violations(sql))

    def test_schema_qualified_disallowed_table(self) -> None:
        assert POLICY.violations("SELECT * FROM hr.payroll")

    def test_restricted_column_direct(self) -> None:
        assert POLICY.violations("SELECT email FROM customers") == [
            "Column `customers.email` is restricted on this connection."
        ]

    def test_restricted_column_through_alias(self) -> None:
        assert POLICY.violations("SELECT c.email AS e FROM customers c")

    def test_restricted_column_case_insensitive(self) -> None:
        assert POLICY.violations('SELECT "SSN" FROM customers')

    def test_restricted_column_in_where_clause_is_blocked(self) -> None:
        # Probing a value through a filter leaks it one bit at a time.
        assert POLICY.violations("SELECT count(*) FROM customers WHERE email LIKE 'a%'")

    def test_restricted_column_inside_function(self) -> None:
        assert POLICY.violations("SELECT lower(c.email) FROM customers c")

    def test_restricted_column_via_cte(self) -> None:
        sql = "WITH x AS (SELECT email FROM customers) SELECT * FROM x"
        assert POLICY.violations(sql)

    def test_same_column_name_on_other_table_is_fine(self) -> None:
        assert POLICY.violations("SELECT o.email FROM orders o") == []

    def test_unparseable_sql_fails_closed(self) -> None:
        problems = POLICY.violations("SELEC name FROM FROM customers ((")
        assert problems and "blocked" in problems[0]

    def test_empty_policy_never_blocks(self) -> None:
        assert AccessPolicy().violations("SELEC garbage ((") == []


class TestRedaction:
    def test_star_expansion_is_redacted(self) -> None:
        rows = [{"id": 1, "name": "Ann", "email": "a@x.com", "ssn": "123"}]
        out = POLICY.redact_rows(rows, {"customers"})
        assert out == [{"id": 1, "name": "Ann", "email": REDACTED, "ssn": REDACTED}]

    def test_only_referenced_tables_are_considered(self) -> None:
        rows = [{"email": "ops@x.com"}]
        assert POLICY.redact_rows(rows, {"orders"}) == rows


class TestFilterChunks:
    def test_drops_tables_and_restricted_lines(self) -> None:
        chunks = [
            SchemaChunk(
                chunk_id="public.customers",
                table_name="customers",
                content="Table: customers\nColumns:\n  - id (int)\n  - email (text)\n  - name (text)",
            ),
            SchemaChunk(chunk_id="public.payroll", table_name="payroll", content="Table: payroll"),
            SchemaChunk(
                chunk_id="public.customers#col=email",
                table_name="customers",
                content="Table customers column email (text)",
                metadata={"column": "email", "is_child": True},
            ),
        ]
        out = POLICY.filter_chunks(chunks)
        assert [c.chunk_id for c in out] == ["public.customers"]
        assert "email" not in out[0].content
        assert "name (text)" in out[0].content


class _FakeClient(AsyncDatabaseClient):
    """Records executed SQL; never touches a real database."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:  # no super().__init__
        self.rows = rows
        self.executed: list[str] = []

    async def execute_sql(self, sql: str) -> list[dict[str, Any]]:
        self.executed.append(sql)
        return self.rows

    async def explain(self, sql: str, *, analyze: bool = False) -> dict[str, Any]:
        self.executed.append(sql)
        return {"plan": "ok"}


class TestPolicyEnforcingClient:
    async def test_blocked_query_never_reaches_database(self) -> None:
        inner = _FakeClient([])
        client = PolicyEnforcingClient(inner, POLICY, "postgresql")
        with pytest.raises(AccessPolicyViolationError) as err:
            await client.execute_sql("SELECT email FROM customers")
        assert inner.executed == []
        # Subclass of DatabaseExecutionError so the self-correction loop catches it.
        assert isinstance(err.value, DatabaseExecutionError)

    async def test_allowed_query_runs_and_star_is_redacted(self) -> None:
        inner = _FakeClient([{"id": 1, "email": "a@x.com"}])
        client = PolicyEnforcingClient(inner, POLICY, "postgresql")
        rows = await client.execute_sql("SELECT * FROM customers")
        assert rows == [{"id": 1, "email": REDACTED}]

    async def test_explain_is_enforced(self) -> None:
        inner = _FakeClient([])
        client = PolicyEnforcingClient(inner, POLICY, "postgresql")
        with pytest.raises(AccessPolicyViolationError):
            await client.explain("SELECT * FROM payroll")
        assert inner.executed == []
