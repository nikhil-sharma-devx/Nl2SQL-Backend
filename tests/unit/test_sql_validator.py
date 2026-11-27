"""Unit tests for SQLValidatorService."""
import pytest

from nl_to_sql.services.sql_validator import SQLValidatorService


@pytest.fixture
def validator() -> SQLValidatorService:
    return SQLValidatorService(dialect="postgresql")


class TestSQLValidatorService:
    def test_valid_select_passes(self, validator: SQLValidatorService) -> None:
        result = validator.validate("SELECT customer_id, email FROM customers LIMIT 10;")
        assert result.is_valid is True
        assert result.errors == []
        assert result.normalised_sql is not None

    def test_empty_sql_fails(self, validator: SQLValidatorService) -> None:
        result = validator.validate("")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_cannot_answer_marker_fails(self, validator: SQLValidatorService) -> None:
        result = validator.validate("-- CANNOT_ANSWER")
        assert result.is_valid is False
        assert any("cannot be answered" in e for e in result.errors)

    def test_syntax_error_fails(self, validator: SQLValidatorService) -> None:
        # Use a fragment with genuinely invalid SQL syntax that sqlglot rejects
        result = validator.validate("SELECT FROM WHERE;")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_valid_aggregation(self, validator: SQLValidatorService) -> None:
        sql = """
        SELECT country, COUNT(*) AS total
        FROM customers
        GROUP BY country
        ORDER BY total DESC;
        """
        result = validator.validate(sql)
        assert result.is_valid is True

    def test_valid_join(self, validator: SQLValidatorService) -> None:
        sql = """
        SELECT c.email, SUM(o.total_amount) AS revenue
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.email;
        """
        result = validator.validate(sql)
        assert result.is_valid is True
