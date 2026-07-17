"""Pydantic models for configuration API request/response contracts."""
from pydantic import BaseModel, Field


class LLMConfigResponse(BaseModel):
    """Current LLM configuration response."""

    provider: str = Field(..., description="Active LLM provider (groq).")
    model: str = Field(..., description="Active LLM model identifier.")
    available_providers: list[str] = Field(
        default_factory=list,
        description="List of supported providers.",
    )


class LLMConfigUpdate(BaseModel):
    """Request body for updating LLM configuration."""

    provider: str = Field(
        ...,
        description="LLM provider to switch to (groq).",
        examples=["groq"],
    )
    model: str = Field(
        ...,
        description="Model name to use with the provider.",
        examples=["llama3-70b-8192"],
    )


class LLMConfigUpdateResponse(BaseModel):
    """Response after updating LLM configuration."""

    provider: str = Field(..., description="Updated LLM provider.")
    model: str = Field(..., description="Updated LLM model name.")
    message: str = Field(..., description="Status message.")


class AvailableModelsResponse(BaseModel):
    """Map of provider names to their supported models."""

    groq: list[str] = Field(default_factory=list, description="Available Groq models.")
    openai: list[str] = Field(default_factory=list, description="Available OpenAI models.")
    anthropic: list[str] = Field(default_factory=list, description="Available Anthropic Claude models.")
    gemini: list[str] = Field(default_factory=list, description="Available Google Gemini models.")


class DatabaseConfigResponse(BaseModel):
    """Current database configuration response."""

    database_url: str = Field(..., description="Active database connection string.")
    available_databases: dict[str, str] = Field(default_factory=dict, description="Pre-defined databases from config")


class DatabaseConfigUpdate(BaseModel):
    """Request body for updating database configuration."""

    database_url: str = Field(
        ...,
        description="New database connection string to switch to.",
        examples=["postgresql+asyncpg://postgres:password@localhost:5432/postgres"],
    )


class DatabaseConfigUpdateResponse(BaseModel):
    """Response after updating database configuration."""

    database_url: str = Field(..., description="Updated database connection string.")
    message: str = Field(..., description="Status message.")


class RagConfigResponse(BaseModel):
    """Current Phase-3 RAG quality configuration (runtime-adjustable)."""

    schema_descriptions_enabled: bool = Field(
        ..., description="P1 — embed LLM-generated NL table descriptions at ingest (needs re-ingest)."
    )
    multi_query_enabled: bool = Field(..., description="P3 — multi-query retrieval.")
    multi_query_max: int = Field(..., description="Max extra query variants for multi-query.")
    few_shot_retrieval_enabled: bool = Field(
        ..., description="P2 — inject semantically-similar past NL→SQL examples."
    )
    few_shot_top_k: int = Field(..., description="Number of few-shot examples to inject.")
    parent_child_chunking_enabled: bool = Field(
        ..., description="P4 — column-level child chunks at ingest (needs re-ingest)."
    )
    hyde_enabled: bool = Field(..., description="P5 — HyDE hypothetical-document embedding.")
    adaptive_top_k_enabled: bool = Field(..., description="P7 — adaptive retrieval top_k.")
    adaptive_top_k_min: int = Field(..., description="Lower bound for adaptive top_k.")
    adaptive_top_k_max: int = Field(..., description="Upper bound for adaptive top_k.")


class RagConfigUpdate(BaseModel):
    """Partial update of the RAG configuration. Only supplied fields change."""

    schema_descriptions_enabled: bool | None = None
    multi_query_enabled: bool | None = None
    multi_query_max: int | None = Field(default=None, ge=0, le=10)
    few_shot_retrieval_enabled: bool | None = None
    few_shot_top_k: int | None = Field(default=None, ge=0, le=10)
    parent_child_chunking_enabled: bool | None = None
    hyde_enabled: bool | None = None
    adaptive_top_k_enabled: bool | None = None
    adaptive_top_k_min: int | None = Field(default=None, ge=1, le=50)
    adaptive_top_k_max: int | None = Field(default=None, ge=1, le=50)
