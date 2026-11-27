"""Unit tests for _redact_sql_for_tracing (Medium finding).

The db.statement tracing span recorded the full raw SQL text — including
any literal values the LLM copied straight from the user's question — with
no redaction before export to the tracing backend.
"""
from __future__ import annotations

from nl_to_sql.infrastructure.database.sqlalchemy_client import _redact_sql_for_tracing


class TestRedactSqlForTracing:
    def test_string_literal_redacted(self) -> None:
        redacted = _redact_sql_for_tracing("SELECT * FROM users WHERE email = 'jane@example.com'")
        assert "jane@example.com" not in redacted
        assert "SELECT * FROM users WHERE email = '?'" == redacted

    def test_numeric_literal_redacted(self) -> None:
        redacted = _redact_sql_for_tracing("SELECT * FROM orders WHERE amount > 15000")
        assert "15000" not in redacted
        assert "SELECT * FROM orders WHERE amount > ?" == redacted

    def test_table_and_column_names_untouched(self) -> None:
        """Structure must stay visible — only literal values are redacted."""
        redacted = _redact_sql_for_tracing("SELECT id, table1 FROM orders2 WHERE id = 5")
        assert redacted == "SELECT id, table1 FROM orders2 WHERE id = ?"

    def test_multiple_literals_all_redacted(self) -> None:
        redacted = _redact_sql_for_tracing(
            "SELECT * FROM t WHERE name = 'Jane Doe' AND age = 30 AND city = 'Paris'"
        )
        assert "Jane Doe" not in redacted
        assert "Paris" not in redacted
        assert "30" not in redacted
