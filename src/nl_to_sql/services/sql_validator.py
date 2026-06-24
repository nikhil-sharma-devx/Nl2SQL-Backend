"""SQL validator — verifies generated SQL using sqlglot."""
import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp
import structlog

from nl_to_sql.core.interfaces.i_sql_validator import ISQLValidator
from nl_to_sql.core.models.sql_result import ValidationResult

logger = structlog.get_logger(__name__)

# Map our config dialect names to sqlglot dialect names
_DIALECT_MAP: dict[str, str] = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "bigquery": "bigquery",
    "snowflake": "snowflake",
}

_CANNOT_ANSWER_MARKER = "-- CANNOT_ANSWER"

# Functions that can exfiltrate data or execute OS commands even inside a SELECT
_BLOCKED_FUNCTIONS: frozenset[str] = frozenset({
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_ls_logdir",
    "pg_ls_waldir", "pg_ls_archive_statusdir", "lo_import", "lo_export",
    "lo_create", "lo_unlink", "dblink", "dblink_exec", "dblink_open",
    "dblink_fetch", "dblink_connect", "copy", "pg_sleep",
    "pg_cancel_backend", "pg_terminate_backend", "pg_reload_conf",
    "pg_rotate_logfile", "pg_file_write", "pg_file_rename",
    "pg_file_unlink", "pg_logdir_ls", "inet_client_addr",
})


class SQLValidatorService(ISQLValidator):  # type: ignore[misc]
    """Validates SQL using sqlglot's parser.

    Checks:
      1. Syntax is valid for the target dialect.
      2. The LLM did not signal it cannot answer (CANNOT_ANSWER marker).
      3. Returns normalised SQL on success.

    SOLID:
      S — Only validates; does not generate or execute.
      O — New validation rules can be added via subclassing or decorators.
    """

    def __init__(self, dialect: str = "postgresql") -> None:
        self._dialect = _DIALECT_MAP.get(dialect.lower(), dialect.lower())

    def validate(self, sql: str) -> ValidationResult:
        """Validate the SQL string and return a ValidationResult.

        Args:
            sql: The cleaned SQL string from the generator.

        Returns:
            ValidationResult with is_valid, errors, and normalised SQL.
        """
        if not sql or not sql.strip():
            return ValidationResult(is_valid=False, errors=["Empty SQL was generated."])

        if _CANNOT_ANSWER_MARKER in sql:
            return ValidationResult(
                is_valid=False,
                errors=["The LLM determined that the question cannot be answered "
                        "from the available schema."],
            )

        try:
            expressions = sqlglot.parse(sql, dialect=self._dialect, error_level=sqlglot.ErrorLevel.RAISE)
        except sqlglot.errors.ParseError as exc:
            errors = [str(e) for e in exc.errors]
            logger.debug("SQL validation failed", errors=errors)
            return ValidationResult(is_valid=False, errors=errors)
        except Exception as exc:
            return ValidationResult(is_valid=False, errors=[f"Unexpected validation error: {exc}"])

        if not expressions:
            return ValidationResult(is_valid=False, errors=["Could not parse any SQL statements."])

        # Reject any statement that is not a bare SELECT
        for statement in expressions:
            if not isinstance(statement, exp.Select):
                stmt_type = type(statement).__name__
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Only SELECT statements are permitted. Received: {stmt_type}."],
                )

        # Reject queries that reference dangerous functions
        for statement in expressions:
            if statement is None:
                continue
            for node in statement.walk():
                func_name = None
                if isinstance(node, exp.Anonymous):
                    func_name = (node.name or "").lower()
                elif isinstance(node, exp.Func):
                    func_name = type(node).__name__.lower()
                if func_name and func_name in _BLOCKED_FUNCTIONS:
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Use of '{func_name}' is not permitted."],
                    )

        # Normalise: re-format to canonical style
        try:
            normalised = sqlglot.transpile(
                sql, read=self._dialect, write=self._dialect, pretty=True
            )[0]
        except Exception:
            normalised = sql  # fall back to original if transpile fails

        logger.debug("SQL validation passed")
        return ValidationResult(is_valid=True, normalised_sql=normalised)
