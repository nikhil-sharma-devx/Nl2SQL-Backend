"""Pydantic models for API request/response contracts."""
from typing import Any, Literal

from pydantic import BaseModel, Field

# Pipeline stages streamed over SSE — the single source of truth for stage
# names. The frontend mirrors this union in `frontend/src/api/client.ts`
# (PipelineStage); keep both in sync.
PipelineStage = Literal[
    "initializing",
    "retrieving_schema",
    "schema_retrieved",
    "generating_sql",
    "sql_generated",
    "validating_sql",
    "executing_sql",
]


class PipelineStageEvent(BaseModel):
    """A single progress event emitted by the streaming pipeline."""

    status: Literal["started", "progress", "complete", "error"] = Field(
        ..., description="Lifecycle status of the stream."
    )
    stage: PipelineStage | None = Field(
        default=None, description="Pipeline stage this event refers to."
    )
    tables: list[str] | None = Field(
        default=None, description="Tables grounded on (schema_retrieved stage)."
    )
    sql: str | None = Field(
        default=None, description="Draft SQL (sql_generated stage)."
    )
    cached: bool | None = Field(
        default=None, description="True when the final payload came from cache."
    )
    data: dict[str, Any] | None = Field(
        default=None, description="Full QueryResponse payload (complete status)."
    )
    response_time_ms: int | None = Field(default=None)
    error: str | None = Field(default=None, description="Error message (error status).")
    type: str | None = Field(default=None, description="Error type name (error status).")

    def to_sse(self) -> dict[str, Any]:
        """Compact dict for the SSE frame (drops None fields)."""
        return self.model_dump(exclude_none=True)


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
    stage_timings: dict[str, int] | None = Field(
        default=None,
        description="Per-stage latency in milliseconds "
        "(retrieval, table_selection, generation, validation, execution).",
    )
