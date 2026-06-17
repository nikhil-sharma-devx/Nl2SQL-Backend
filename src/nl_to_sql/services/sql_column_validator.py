"""SQL Column Validator — Validates that all referenced columns exist in schema.

Uses sqlglot to parse SQL and extract all table.column references, then validates
them against the provided schema context. This catches LLM hallucinations where
the model invents column names that don't exist.

Example errors caught:
- "p.category doesn't exist (did you mean p.category_id?)"
- "c.city doesn't exist in customers table"
- "Table 'orders' referenced but not in schema context"

SOLID:
  S — Only validates column existence
  D — Depends on schema context (no database connection needed)
"""
import re
from typing import Any

import sqlglot
import sqlglot.expressions as exp
import structlog

logger = structlog.get_logger(__name__)


class SQLColumnValidator:
    """Validates SQL queries against schema to catch column hallucinations."""

    def __init__(self, dialect: str = "postgres") -> None:
        """Initialize with SQL dialect.

        Args:
            dialect: SQL dialect for parsing (postgres, mysql, etc.)
        """
        if dialect.lower() == "postgresql":
            self._dialect = "postgres"
        else:
            self._dialect = dialect

    def validate(
        self,
        sql: str,
        schema_context: dict[str, list[str]],
    ) -> list[str]:
        """Validate that all columns in SQL exist in schema.

        Args:
            sql: The SQL query to validate.
            schema_context: Dict mapping table_name -> list of column names.
                          Example: {"products": ["product_id", "name", "category_id"]}

        Returns:
            List of validation errors. Empty list means valid.
        """
        errors: list[str] = []

        try:
            # Parse SQL
            parsed = sqlglot.parse_one(sql, read=self._dialect)

            # Extract all column references
            column_refs = self._extract_column_references(parsed)

            # Validate each column reference
            for table_alias, column_name in column_refs:
                # Resolve alias to actual table name
                table_name = self._resolve_alias(table_alias, parsed)

                if table_name and table_name in schema_context:
                    # Check if column exists
                    valid_columns = schema_context[table_name]
                    if column_name not in valid_columns:
                        # Suggest similar column names
                        suggestions = self._find_similar_columns(
                            column_name, valid_columns
                        )
                        suggestion_msg = ""
                        if suggestions:
                            suggestion_msg = f" Did you mean: {', '.join(suggestions)}?"

                        errors.append(
                            f"Column '{table_alias}.{column_name}' does not exist "
                            f"in table '{table_name}'.{suggestion_msg} "
                            f"Available columns: {', '.join(valid_columns)}"
                        )
                elif table_name and table_name not in schema_context:
                    errors.append(
                        f"Table '{table_name}' is not in the schema context. "
                        f"Available tables: {', '.join(schema_context.keys())}"
                    )

        except Exception as exc:
            logger.warning(
                "Failed to parse SQL for column validation",
                error=str(exc),
                sql=sql[:100],
            )
            # Don't fail validation on parse errors — let other validators handle it
            errors.append(f"SQL parse error: {str(exc)}")

        return errors

    def _extract_column_references(
        self, parsed: exp.Expression
    ) -> list[tuple[str | None, str]]:
        """Extract all table.column references from parsed SQL.

        Returns:
            List of (table_alias_or_name, column_name) tuples.
        """
        column_refs: list[tuple[str | None, str]] = []

        for column in parsed.find_all(exp.Column):
            table = column.table  # alias or table name
            col_name = column.name

            # Skip function calls and special columns
            if col_name == "*":
                continue

            column_refs.append((table if table else None, col_name))

        return column_refs

    def _resolve_alias(
        self,
        alias: str | None,
        parsed: exp.Expression,
    ) -> str | None:
        """Resolve table alias to actual table name.

        Example: "p" -> "products" (if "FROM products p" or "FROM products AS p")
        """
        if not alias:
            return None

        # Build alias -> table_name map from FROM and JOIN clauses
        alias_map: dict[str, str] = {}

        # Extract FROM tables
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            table_alias = table.alias if table.alias else table_name
            alias_map[table_alias] = table_name

        # If alias is in map, return actual table name
        return alias_map.get(alias, alias)

    def _find_similar_columns(
        self,
        target: str,
        valid_columns: list[str],
        max_suggestions: int = 3,
    ) -> list[str]:
        """Find column names similar to the target (for helpful error messages).

        Uses simple string similarity (common prefixes/suffixes).
        """
        suggestions: list[str] = []

        for col in valid_columns:
            # Check if they share a common prefix
            if col.startswith(target[:3]) or target.startswith(col[:3]):
                suggestions.append(col)
            # Check if they contain the same substring
            elif target.lower() in col.lower() or col.lower() in target.lower():
                suggestions.append(col)

        return suggestions[:max_suggestions]

    def extract_schema_from_context(
        self,
        schema_context_text: str,
    ) -> dict[str, list[str]]:
        """Parse schema context text to extract table -> columns mapping.

        This is a fallback when structured schema isn't available.
        Parses text like:
            Table: products
            Columns:
              - product_id (integer, PK)
              - name (text)
              - category_id (integer, FK -> categories.category_id)

        Args:
            schema_context_text: Formatted schema context string.

        Returns:
            Dict mapping table_name -> list of column names.
        """
        schema: dict[str, list[str]] = {}
        current_table: str | None = None

        # Pattern to match table declarations
        table_pattern = re.compile(r'^Table[:\s]+(\w+)', re.MULTILINE)
        # Pattern to match column declarations
        column_pattern = re.compile(r'^\s*[-•*]\s+(\w+)\s+', re.MULTILINE)

        for line in schema_context_text.split('\n'):
            # Check for table declaration
            table_match = table_pattern.match(line)
            if table_match:
                current_table = table_match.group(1)
                schema[current_table] = []
                continue

            # Check for column declaration
            if current_table:
                col_match = column_pattern.match(line)
                if col_match:
                    schema[current_table].append(col_match.group(1))

        logger.debug(
            "Extracted schema from context",
            tables=list(schema.keys()),
        )

        return schema
