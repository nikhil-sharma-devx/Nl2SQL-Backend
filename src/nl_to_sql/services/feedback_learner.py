"""Feedback Learner — Learns from user feedback to prevent recurring mistakes.

Analyzes feedback patterns and injects them into LLM prompts to avoid repeating
the same errors. This creates a self-improving system that gets better over time.

SOLID:
  S — Only handles feedback learning and pattern injection
  D — Depends on feedback service and database
"""
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class FeedbackPattern(BaseModel):
    """Represents a learned pattern from user feedback."""

    pattern_id: str
    error_type: str  # e.g., "wrong_column", "missing_join", "wrong_aggregation"
    description: str  # Human-readable description
    example_sql: str  # The incorrect SQL that was generated
    correction: str  # The corrected SQL or guidance
    table_name: str | None = None  # Table involved (if applicable)
    column_name: str | None = None  # Column involved (if applicable)
    occurrence_count: int = 1
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackLearner:
    """Learns from feedback to improve SQL generation quality."""

    def __init__(
        self,
        feedback_service: Any | None = None,
        training_data_service: Any | None = None,
    ) -> None:
        self._feedback_service = feedback_service
        self._training_data_service = training_data_service
        self._patterns: dict[str, FeedbackPattern] = {}
        self._logger = logger.bind(component="FeedbackLearner")

    async def record_feedback(
        self,
        question: str,
        generated_sql: str,
        feedback_type: str,  # "positive" or "negative"
        error_type: str | None = None,
        user_correction: str | None = None,
        user_notes: str | None = None,
    ) -> None:
        """Record user feedback and learn from it.

        Args:
            question: Original user question.
            generated_sql: SQL that was generated.
            feedback_type: "positive" or "negative".
            error_type: Type of error (for negative feedback).
            user_correction: User's corrected SQL (if provided).
            user_notes: Additional notes from user.
        """
        if feedback_type == "positive":
            self._logger.info("Positive feedback received", question=question[:50])
            # Reinforce: save to training_data so fine-tuning picks it up
            if self._training_data_service:
                try:
                    await self._training_data_service.collect_training_data(
                        question=question,
                        sql=generated_sql,
                        retrieved_tables=[],
                        schema_context="",
                        intent_type="user_approved",
                        success_score=1.0,
                    )
                    self._logger.info("Positive feedback saved to training data")
                except Exception as exc:
                    self._logger.warning("Failed to save positive feedback to training data", error=str(exc))
            return

        # Negative feedback — learn from mistake
        if error_type:
            pattern = self._create_pattern(
                question=question,
                generated_sql=generated_sql,
                error_type=error_type,
                user_correction=user_correction,
                user_notes=user_notes,
            )

            self._patterns[pattern.pattern_id] = pattern

            if self._feedback_service:
                try:
                    await self._feedback_service.submit_feedback(
                        feedback_type="pattern",
                        feedback_data={
                            "pattern_id": pattern.pattern_id,
                            "error_type": pattern.error_type,
                            "description": pattern.description,
                            "example_sql": pattern.example_sql,
                            "correction": pattern.correction,
                        },
                    )
                except Exception as exc:
                    self._logger.warning("Failed to persist pattern", error=str(exc))

            # If user supplied a corrected SQL, save it as high-quality training data
            if user_correction and user_correction.strip() and self._training_data_service:
                try:
                    await self._training_data_service.collect_training_data(
                        question=question,
                        sql=user_correction.strip(),
                        retrieved_tables=[],
                        schema_context="",
                        intent_type="user_corrected",
                        success_score=1.0,
                    )
                    self._logger.info("User correction saved to training data")
                except Exception as exc:
                    self._logger.warning("Failed to save correction to training data", error=str(exc))

            self._logger.info("Learned from negative feedback", error_type=error_type, pattern_id=pattern.pattern_id)

    def _create_pattern(
        self,
        question: str,
        generated_sql: str,
        error_type: str,
        user_correction: str | None = None,
        user_notes: str | None = None,
    ) -> FeedbackPattern:
        """Create a feedback pattern from error."""
        import hashlib

        # Create unique pattern ID
        pattern_hash = hashlib.md5(  # noqa: S324 — used for dedup, not security
            f"{error_type}:{generated_sql[:100]}".encode()
        ).hexdigest()[:12]

        # Extract table/column info if possible
        table_name = self._extract_table_name(generated_sql)
        column_name = self._extract_column_name(error_type)

        # Build description
        description = self._build_description(
            error_type=error_type,
            table_name=table_name,
            column_name=column_name,
            user_notes=user_notes,
        )

        return FeedbackPattern(
            pattern_id=f"pattern_{pattern_hash}",
            error_type=error_type,
            description=description,
            example_sql=generated_sql,
            correction=user_correction or "See description for guidance",
            table_name=table_name,
            column_name=column_name,
        )

    def _extract_table_name(self, sql: str) -> str | None:
        """Extract table name from SQL query."""
        import re

        # Simple extraction - could be enhanced with sqlglot
        match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_column_name(self, error_type: str) -> str | None:
        """Extract column name from error type."""
        # Error types like "column 'p.category' does not exist"
        import re

        match = re.search(r"column ['\"]?\w+\.(\w+)['\"]?", error_type)
        if match:
            return match.group(1)
        return None

    def _build_description(
        self,
        error_type: str,
        table_name: str | None,
        column_name: str | None,
        user_notes: str | None,
    ) -> str:
        """Build human-readable description of the pattern."""
        parts = []

        if table_name and column_name:
            parts.append(
                f"Table '{table_name}' does not have column '{column_name}'"
            )
        elif table_name:
            parts.append(f"Issue with table '{table_name}': {error_type}")
        else:
            parts.append(error_type)

        if user_notes:
            parts.append(f"User note: {user_notes}")

        return ". ".join(parts)

    async def get_relevant_patterns(self, question: str) -> list[FeedbackPattern]:
        """Get relevant feedback patterns for a question."""
        return list(self._patterns.values())

    def build_pattern_context(self, patterns: list[FeedbackPattern]) -> str:
        """Format patterns into a prompt context string."""
        if not patterns:
            return ""
        lines = [
            "\n⚠️ COMMON MISTAKES TO AVOID (learned from user feedback):",
            ""
        ]
        for pattern in patterns:
            lines.append(f"• {pattern.description}")
            if pattern.correction:
                lines.append(f"  Correction: {pattern.correction}")
            lines.append("")
        return "\n".join(lines)

    def get_learning_prompt(self, tables: list[str] | None = None) -> str:
        """Generate prompt section with learned patterns to avoid.

        Args:
            tables: Optional filter to only include patterns for these tables.

        Returns:
            Formatted string with common mistakes to avoid.
        """
        if not self._patterns:
            return ""

        # Filter patterns by tables if specified
        relevant_patterns = self._patterns
        if tables:
            relevant_patterns = {
                pid: pattern
                for pid, pattern in self._patterns.items()
                if pattern.table_name in tables
            }

        return self.build_pattern_context(list(relevant_patterns.values()))

    async def analyze_and_optimize(self) -> dict[str, Any]:
        """Analyze all feedback patterns and generate optimization suggestions.

        Returns:
            Dictionary with insights and recommendations.
        """
        if not self._patterns:
            return {"message": "No feedback patterns to analyze"}

        # Group by error type
        error_counts: dict[str, int] = {}
        for pattern in self._patterns.values():
            error_counts[pattern.error_type] = (
                error_counts.get(pattern.error_type, 0) + 1
            )

        # Find most common errors
        top_errors = sorted(
            error_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        insights: dict[str, Any] = {
            "total_patterns": len(self._patterns),
            "top_error_types": top_errors,
            "recommendations": [],
        }

        # Generate recommendations
        for error_type, count in top_errors:
            if "column" in error_type.lower():
                insights["recommendations"].append(
                    f"Improve schema retrieval for column accuracy ({count} occurrences)"
                )
            elif "join" in error_type.lower():
                insights["recommendations"].append(
                    f"Enhance FK relationship detection ({count} occurrences)"
                )

        return insights

    def get_pattern_count(self) -> int:
        """Return total number of learned patterns."""
        return len(self._patterns)
