"""ISQLValidator — Abstract interface for SQL validation strategies."""
from abc import ABC, abstractmethod

from nl_to_sql.core.models.sql_result import ValidationResult


class ISQLValidator(ABC):
    """Contract for validating generated SQL.

    SOLID: Single Responsibility — only validates; does not generate or execute.
           Interface Segregation  — callers that only need validation depend
                                    only on this slim interface.
    """

    @abstractmethod
    def validate(self, sql: str) -> ValidationResult:
        """Validate the given SQL string.

        Args:
            sql: The raw SQL string produced by the LLM.

        Returns:
            ValidationResult with is_valid flag and a list of error messages.
        """
        ...
