"""Pydantic models for SQL generation results and validation."""
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Raw response from the LLM provider."""

    content: str = Field(..., description="Raw text returned by the LLM.")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ValidationResult(BaseModel):
    """Result of SQL validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    normalised_sql: str | None = Field(
        default=None,
        description="SQL re-formatted by the validator (on success).",
    )


class GeneratedSQL(BaseModel):
    """The artifact produced by the SQL generator."""

    raw_sql: str = Field(..., description="SQL as returned by the LLM.")
    cleaned_sql: str = Field(
        ...,
        description="SQL after stripping markdown fences and whitespace.",
    )
    dialect: str
    validation: ValidationResult
    tokens_used: int = 0
    used_tables: list[str] = Field(
        default_factory=list,
        description="Tables actually used in the generated SQL query.",
    )
    attempt: int = Field(
        default=1,
        description="Which retry attempt produced this result (1-indexed).",
    )
    suggested_chart: dict[str, Any] | None = Field(default=None)
    follow_up_questions: list[str] = Field(default_factory=list)
