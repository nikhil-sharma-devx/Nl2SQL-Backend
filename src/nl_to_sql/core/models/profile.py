"""Pydantic models for Profile / API key management."""
from pydantic import BaseModel, Field


class SaveAPIKeyRequest(BaseModel):
    """Request body for saving a personal LLM API key."""

    api_key: str = Field(..., min_length=8, description="The API key to store encrypted")


class APIKeyStatusItem(BaseModel):
    """Status of a single LLM provider's API key availability."""

    provider: str = Field(..., description="Provider identifier, e.g. 'groq' or 'openai'")
    label: str = Field(..., description="Human-readable provider name")
    has_user_key: bool = Field(..., description="Whether the user has stored their own key")
    has_server_key: bool = Field(..., description="Whether the server has a key configured")
    key_preview: str | None = Field(None, description="Masked key preview, e.g. 'sk-abc...xyz'")
    available_models: list[str] = Field(default_factory=list, description="Models available for this provider")


class APIKeyStatusResponse(BaseModel):
    """Response listing the status of all supported LLM providers."""

    keys: list[APIKeyStatusItem]
    active_provider: str = Field(..., description="Currently active LLM provider")
    active_model: str = Field(..., description="Currently active LLM model")


class SaveAPIKeyResponse(BaseModel):
    """Response after saving an API key."""

    provider: str = Field(..., description="Provider the key was saved for")
    key_preview: str = Field(..., description="Masked key preview")
    message: str = Field(..., description="Success message")


class DeleteAPIKeyResponse(BaseModel):
    """Response after deleting an API key."""

    provider: str = Field(..., description="Provider the key was removed from")
    message: str = Field(..., description="Success message")


class AvailableProviderItem(BaseModel):
    """A single provider's availability for the requesting user."""

    provider: str
    label: str
    source: str = Field(..., description="'user', 'server', or 'none'")
    available_models: list[str]
    is_active: bool = Field(..., description="Whether this is the currently active provider")


class AvailableProvidersResponse(BaseModel):
    """Which providers are usable (have at least one valid key source)."""

    providers: list[AvailableProviderItem]
