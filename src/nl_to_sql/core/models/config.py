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

    groq: list[str] = Field(
        default_factory=list,
        description="Available Groq models.",
    )
    openai: list[str] = Field(
        default_factory=list,
        description="Available OpenAI models.",
    )


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
