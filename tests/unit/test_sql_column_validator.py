"""Unit tests for SQLColumnValidator — catches column hallucinations."""
import pytest

from nl_to_sql.services.sql_column_validator import SQLColumnValidator


@pytest.fixture
def validator() -> SQLColumnValidator:
    return SQLColumnValidator(dialect="postgresql")


SCHEMA: dict[str, list[str]] = {
    "customers": ["customer_id", "email", "country", "created_at"],
    "orders": ["order_id", "customer_id", "total_amount", "status"],
    "products": ["product_id", "name", "category_id", "price"],
}


class TestValidate:

    def test_valid_simple_query_returns_no_errors(self, validator: SQLColumnValidator) -> None:
        sql = "SELECT c.customer_id, c.email FROM customers c"
        errors = validator.validate(sql, SCHEMA)
        assert errors == []

    def test_valid_join_returns_no_errors(self, validator: SQLColumnValidator) -> None:
        sql = (
            "SELECT c.email, o.total_amount "
            "FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id"
        )
        errors = validator.validate(sql, SCHEMA)
        assert errors == []

    def test_hallucinated_column_returns_error(self, validator: SQLColumnValidator) -> None:
        sql = "SELECT c.phone_number FROM customers c"
        errors = validator.validate(sql, SCHEMA)
        assert len(errors) >= 1
        assert any("phone_number" in e for e in errors)

    def test_suggests_similar_column(self, validator: SQLColumnValidator) -> None:
        # "customer_code" is close to "customer_id"
        sql = "SELECT c.customer_code FROM customers c"
        errors = validator.validate(sql, SCHEMA)
        assert len(errors) >= 1
        # Should suggest customer_id since they share prefix
        assert any("customer_id" in e for e in errors)

    def test_select_star_is_allowed(self, validator: SQLColumnValidator) -> None:
        sql = "SELECT * FROM customers"
        errors = validator.validate(sql, SCHEMA)
        assert errors == []

    def test_unknown_table_returns_error(self, validator: SQLColumnValidator) -> None:
        sql = "SELECT x.foo FROM nonexistent_table x"
        errors = validator.validate(sql, SCHEMA)
        assert len(errors) >= 1
        assert any("nonexistent_table" in e for e in errors)

    def test_valid_aggregation_returns_no_errors(self, validator: SQLColumnValidator) -> None:
        sql = (
            "SELECT o.status, COUNT(o.order_id) AS cnt "
            "FROM orders o GROUP BY o.status"
        )
        errors = validator.validate(sql, SCHEMA)
        assert errors == []

    def test_mysql_dialect_alias(self) -> None:
        mysql_validator = SQLColumnValidator(dialect="mysql")
        sql = "SELECT p.product_id, p.name FROM products p"
        errors = mysql_validator.validate(sql, SCHEMA)
        assert errors == []

    def test_postgresql_dialect_alias_normalized(self) -> None:
        # "postgresql" should be internally mapped to "postgres"
        validator = SQLColumnValidator(dialect="postgresql")
        assert validator._dialect == "postgres"

    def test_empty_sql_does_not_crash(self, validator: SQLColumnValidator) -> None:
        # An empty/parse-error SQL should produce an error entry, not raise
        errors = validator.validate("", SCHEMA)
        assert isinstance(errors, list)

    def test_multiple_hallucinated_columns(self, validator: SQLColumnValidator) -> None:
        sql = "SELECT c.fake_col1, c.fake_col2 FROM customers c"
        errors = validator.validate(sql, SCHEMA)
        assert len(errors) >= 2


class TestFindSimilarColumns:

    def test_prefix_match(self, validator: SQLColumnValidator) -> None:
        suggestions = validator._find_similar_columns("customer_code", ["customer_id", "email"])
        assert "customer_id" in suggestions

    def test_substring_match(self, validator: SQLColumnValidator) -> None:
        suggestions = validator._find_similar_columns("amount", ["total_amount", "name"])
        assert "total_amount" in suggestions

    def test_no_match_returns_empty(self, validator: SQLColumnValidator) -> None:
        suggestions = validator._find_similar_columns("xyz_abc", ["email", "country"])
        assert suggestions == []

    def test_max_suggestions_respected(self, validator: SQLColumnValidator) -> None:
        cols = [f"cust_{i}" for i in range(10)]
        suggestions = validator._find_similar_columns("cust_x", cols, max_suggestions=2)
        assert len(suggestions) <= 2


class TestExtractSchemaFromContext:

    def test_parses_table_and_columns(self, validator: SQLColumnValidator) -> None:
        text = (
            "Table: products\n"
            "Columns:\n"
            "  - product_id (integer, PK)\n"
            "  - name (text)\n"
            "  - category_id (integer, FK -> categories.category_id)\n"
        )
        schema = validator.extract_schema_from_context(text)
        assert "products" in schema
        assert "product_id" in schema["products"]
        assert "name" in schema["products"]
        assert "category_id" in schema["products"]

    def test_multiple_tables_parsed(self, validator: SQLColumnValidator) -> None:
        text = (
            "Table: orders\nColumns:\n  - order_id (integer)\n  - status (text)\n\n"
            "Table: customers\nColumns:\n  - customer_id (integer)\n"
        )
        schema = validator.extract_schema_from_context(text)
        assert "orders" in schema
        assert "customers" in schema
        assert "order_id" in schema["orders"]

    def test_empty_text_returns_empty_dict(self, validator: SQLColumnValidator) -> None:
        schema = validator.extract_schema_from_context("")
        assert schema == {}
