"""Pydantic models for API request/response contracts."""
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming NL-to-SQL query request."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question to convert to SQL.",
        examples=["Show me the top 5 customers by total order value"],
    )
    dialect: str | None = Field(
        default=None,
        description="SQL dialect override (postgresql, mysql). "
        "Falls back to the server-configured default.",
        examples=["postgresql"],
    )
    execute: bool = Field(
        default=False,
        description="If true, execute the generated SQL against the target DB "
        "and return results alongside the query.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional chat session ID to associate this query with.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    streaming: bool = Field(
        default=False,
        description="If true, stream the SQL generation response for faster perceived latency.",
    )


class QueryResponse(BaseModel):
    """Payload returned after NL-to-SQL conversion."""

    question: str = Field(..., description="The original natural language question.")
    sql: str = Field(..., description="The generated SQL query.")
    dialect: str = Field(..., description="SQL dialect used for generation.")
    is_valid: bool = Field(..., description="Whether the SQL passed validation.")
    validation_errors: list[str] = Field(
        default_factory=list,
        description="List of validation error messages (empty when is_valid=True).",
    )
    retrieved_tables: list[str] = Field(
        default_factory=list,
        description="Tables whose schema context was retrieved for this query.",
    )
    used_tables: list[str] = Field(
        default_factory=list,
        description="Tables actually used in the generated SQL query.",
    )
    execution_result: list[dict[str, Any]] | None = Field(
        default=None,
        description="Query results if execute=true was requested.",
    )
    execution_error: str | None = Field(
        default=None,
        description="Error message if SQL execution failed.",
    )
    tokens_used: int = Field(default=0, description="Total tokens consumed by LLM.")
    cached: bool = Field(default=False, description="True if result came from cache.")
    message: str | None = Field(
        default=None,
        description="Optional message for greetings or off-topic responses.",
    )
    suggested_chart: dict[str, Any] | None = Field(
        default=None,
        description="Optimal chart configuration (type, x_axis, y_axis) if graphable.",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="List of 3 suggested follow-up questions.",
    )

    # Analytics and intelligence fields
    intent_type: str | None = Field(
        default=None,
        description="Type of query intent (aggregation, filtering, join, etc.).",
    )
    query_complexity: int | None = Field(
        default=None,
        description="Query complexity on a 1-10 scale.",
    )
    prompt_version: str | None = Field(
        default=None,
        description="Prompt version identifier for A/B testing.",
    )
    retrieval_method: str | None = Field(
        default=None,
        description="Method used for schema retrieval (vector, hybrid, semantic_cache).",
    )
    response_time_ms: int | None = Field(
        default=None,
        description="Total response time in milliseconds.",
    )
