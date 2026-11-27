"""Answer confidence — a deterministic, evidence-based trust signal.

The pipeline already observes everything needed to tell a non-technical user
how much to trust an answer: whether the SQL validated, whether it needed
self-correction, whether execution failed or returned nothing, and whether the
model flagged the question as unanswerable. This module turns those
observations into a level plus plain-English reasons.

It deliberately does NOT ask the LLM to rate itself — a model's opinion of its
own output is not evidence, and a fluent wrong answer is exactly what this
signal exists to catch (PRODUCT.md principle 1: correctness over confidence).

Pure function, no I/O — unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ConfidenceLevel = Literal["high", "medium", "low"]

_CANNOT_ANSWER_MARKER = "-- CANNOT_ANSWER"
_CANNOT_ANSWER_TEXT = "cannot be answered"
# Interpretation choices beyond this count mean the question was open enough
# that the reader should review the assumptions before trusting the number.
_MANY_ASSUMPTIONS = 3


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Outcome of :func:`assess_confidence`."""

    level: ConfidenceLevel
    reasons: list[str] = field(default_factory=list)


def is_cannot_answer(sql: str, validation_errors: list[str]) -> bool:
    """True when the model signalled the schema can't answer the question."""
    if _CANNOT_ANSWER_MARKER in (sql or ""):
        return True
    return any(_CANNOT_ANSWER_TEXT in e.lower() for e in validation_errors)


def assess_confidence(
    *,
    sql: str,
    is_valid: bool,
    validation_errors: list[str],
    attempts: int,
    retrieved_tables: list[str],
    used_tables: list[str],
    execute_requested: bool,
    execution_result: list[dict[str, object]] | None,
    execution_error: str | None,
    empty_result_warning: str | None,
    assumptions: list[str],
) -> ConfidenceAssessment:
    """Classify how far a user should trust this answer, with reasons.

    Low    — the answer must not be relied on (no valid SQL, failed execution,
             or nothing in the schema matched).
    Medium — usable, but something deserves a second look.
    High   — validated first time, executed cleanly (if asked), few assumptions.
    """
    if is_cannot_answer(sql, validation_errors):
        return ConfidenceAssessment(
            "low",
            ["The connected tables don't appear to contain the data this question asks about."],
        )
    if not retrieved_tables:
        return ConfidenceAssessment("low", ["No tables in your schema matched this question."])
    if not is_valid:
        return ConfidenceAssessment(
            "low",
            [
                f"Vectrix could not produce a query that passed validation after "
                f"{attempts} attempt{'s' if attempts != 1 else ''}."
            ],
        )
    if execution_error:
        return ConfidenceAssessment(
            "low", ["The query was valid but failed when it ran against your database."]
        )

    reasons: list[str] = []
    if attempts > 1:
        reasons.append(
            f"The first draft failed validation; this query was corrected on attempt {attempts}."
        )
    unexpected = sorted({t.lower() for t in used_tables} - {t.lower() for t in retrieved_tables})
    if unexpected:
        reasons.append(
            "The query uses tables that weren't in the retrieved schema: "
            + ", ".join(unexpected)
            + "."
        )
    if execute_requested and execution_result is not None and len(execution_result) == 0:
        reasons.append(empty_result_warning or "The query ran but returned no rows.")
    if len(assumptions) >= _MANY_ASSUMPTIONS:
        reasons.append(
            f"The question was open to interpretation ({len(assumptions)} assumptions were made)."
        )

    if reasons:
        return ConfidenceAssessment("medium", reasons)
    return ConfidenceAssessment("high", [])
