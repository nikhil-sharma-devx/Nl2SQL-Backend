"""Profile routes — manage per-user LLM API keys (BYOK) and database connections (BYOD)."""
from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from nl_to_sql.api.dependencies import get_current_user
from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.core.models.profile import (
    APIKeyStatusItem,
    APIKeyStatusResponse,
    AvailableProviderItem,
    AvailableProvidersResponse,
    DeleteAPIKeyResponse,
    SaveAPIKeyRequest,
    SaveAPIKeyResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/profile", tags=["Profile"])

SUPPORTED_PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {
        "label": "Groq (Llama)",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
    "openai": {
        "label": "OpenAI (GPT)",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "models": ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001", "claude-3-5-sonnet-20241022"],
    },
    "gemini": {
        "label": "Google Gemini",
        "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-lite"],
    },
    "together": {
        "label": "Together AI (Fine-tuning)",
        "models": ["Llama 3.1 8B", "Llama 3.1 70B", "Llama 3.2 3B", "Mistral 7B"],
    },
}


def _mask_key(key: str) -> str:
    """Return a masked key preview safe to expose in API responses."""
    if len(key) > 10:
        return key[:6] + "..." + key[-4:]
    return "****"


def _get_api_key_service() -> Any:
    """Access the API key service from the DI container, returning None if unavailable."""
    try:
        from nl_to_sql.api.dependencies import _get_container
        container = _get_container()
        return container.api_key_service()
    except Exception:
        return None


@router.get(
    "/api-keys",
    response_model=APIKeyStatusResponse,
    summary="Get API key status for all providers",
    description=(
        "Returns whether the authenticated user has stored their own key "
        "and whether the server has a key configured, for each supported LLM provider."
    ),
)
async def get_api_key_status(
    current_user: UserPublic = Depends(get_current_user),
) -> APIKeyStatusResponse:
    """List API key availability for all supported providers."""
    settings = get_settings()
    api_key_svc = _get_api_key_service()

    server_keys = {
        "groq": bool(settings.groq_api_key),
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "gemini": bool(settings.gemini_api_key),
        "together": bool(settings.together_api_key),
    }

    items: list[APIKeyStatusItem] = []
    for provider, meta in SUPPORTED_PROVIDERS.items():
        has_user_key = False
        key_preview = None

        if api_key_svc is not None:
            try:
                user_key = await api_key_svc.get_key(current_user.id, provider)
                if user_key:
                    has_user_key = True
                    key_preview = _mask_key(user_key)
            except Exception as exc:
                logger.warning("Failed to check user API key", provider=provider, error=str(exc))

        items.append(
            APIKeyStatusItem(
                provider=provider,
                label=meta["label"],
                has_user_key=has_user_key,
                has_server_key=server_keys.get(provider, False),
                key_preview=key_preview,
                available_models=meta["models"],
            )
        )

    return APIKeyStatusResponse(
        keys=items,
        active_provider=settings.llm_provider,
        active_model=settings.llm_model,
    )


@router.put(
    "/api-keys/{provider}",
    response_model=SaveAPIKeyResponse,
    summary="Save or update a personal API key for a provider",
    description=(
        "Stores the API key encrypted at rest. "
        "The key is used automatically when you make queries — "
        "your personal key takes priority over the server's key."
    ),
)
async def save_api_key(
    provider: str,
    body: SaveAPIKeyRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> SaveAPIKeyResponse:
    """Save or update a user's API key for a specific provider."""
    provider = provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider '{provider}'. Supported: {list(SUPPORTED_PROVIDERS)}",
        )

    api_key_svc = _get_api_key_service()
    if api_key_svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key service is not available. Please try again later.",
        )

    try:
        await api_key_svc.save_key(
            user_id=current_user.id,
            provider=provider,
            api_key=body.api_key.strip(),
        )
    except Exception as exc:
        logger.error("Failed to save API key", provider=provider, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save API key. Please try again.",
        ) from exc

    preview = _mask_key(body.api_key.strip())
    logger.info("User saved API key", user_id=current_user.id, provider=provider)
    return SaveAPIKeyResponse(
        provider=provider,
        key_preview=preview,
        message=f"{SUPPORTED_PROVIDERS[provider]['label']} API key saved successfully.",
    )


@router.delete(
    "/api-keys/{provider}",
    response_model=DeleteAPIKeyResponse,
    summary="Remove a stored personal API key",
    description="Permanently removes the stored key for the given provider. Falls back to the server key after deletion.",
)
async def delete_api_key(
    provider: str,
    current_user: UserPublic = Depends(get_current_user),
) -> DeleteAPIKeyResponse:
    """Delete the user's stored API key for a specific provider."""
    provider = provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider '{provider}'. Supported: {list(SUPPORTED_PROVIDERS)}",
        )

    api_key_svc = _get_api_key_service()
    if api_key_svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key service is not available. Please try again later.",
        )

    try:
        await api_key_svc.delete_key(user_id=current_user.id, provider=provider)
    except Exception as exc:
        logger.error("Failed to delete API key", provider=provider, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key. Please try again.",
        ) from exc

    logger.info("User deleted API key", user_id=current_user.id, provider=provider)
    return DeleteAPIKeyResponse(
        provider=provider,
        message=f"{SUPPORTED_PROVIDERS[provider]['label']} API key removed. Will fall back to server key if available.",
    )


@router.get(
    "/available-providers",
    response_model=AvailableProvidersResponse,
    summary="List providers available for the current user",
    description=(
        "Returns which providers are usable — either via the user's own stored key "
        "or the server's configured key — and marks which is currently active."
    ),
)
async def get_available_providers(
    current_user: UserPublic = Depends(get_current_user),
) -> AvailableProvidersResponse:
    """Return providers the user can actually use (have at least one key source)."""
    settings = get_settings()
    api_key_svc = _get_api_key_service()

    server_keys = {
        "groq": bool(settings.groq_api_key),
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "gemini": bool(settings.gemini_api_key),
        "together": bool(settings.together_api_key),
    }

    result: list[AvailableProviderItem] = []
    for provider, meta in SUPPORTED_PROVIDERS.items():
        has_user_key = False
        if api_key_svc is not None:
            try:
                has_user_key = await api_key_svc.has_key(current_user.id, provider)
            except Exception:
                pass

        has_server_key = server_keys.get(provider, False)

        if has_user_key:
            source = "user"
        elif has_server_key:
            source = "server"
        else:
            source = "none"

        result.append(
            AvailableProviderItem(
                provider=provider,
                label=meta["label"],
                source=source,
                available_models=meta["models"],
                is_active=(provider == settings.llm_provider),
            )
        )

    return AvailableProvidersResponse(providers=result)


# ── Per-user database connection (BYOD) ───────────────────────────────────────

class SaveDatabaseRequest(BaseModel):
    database_url: str


class DatabaseConnectionResponse(BaseModel):
    has_connection: bool
    url_preview: str | None = None  # masked, e.g. "postgresql+asyncpg://user:***@host/db"


class DatabaseConnectionSaveResponse(BaseModel):
    url_preview: str
    message: str


class DatabaseConnectionDeleteResponse(BaseModel):
    message: str


def _mask_url(url: str) -> str:
    """Replace the password in a connection URL with ***."""
    import re
    return re.sub(r"(:)[^:@]+(@)", r"\1***\2", url)


def _get_user_db_service() -> Any:
    try:
        from nl_to_sql.api.dependencies import _get_container
        return _get_container().user_db_service()
    except Exception:
        return None


@router.get(
    "/database",
    response_model=DatabaseConnectionResponse,
    summary="Get current personal database connection status",
)
async def get_database_connection(
    current_user: UserPublic = Depends(get_current_user),
) -> DatabaseConnectionResponse:
    svc = _get_user_db_service()
    if svc is None:
        return DatabaseConnectionResponse(has_connection=False)
    url = await svc.get_raw(current_user.id)
    if url is None:
        return DatabaseConnectionResponse(has_connection=False)
    return DatabaseConnectionResponse(has_connection=True, url_preview=_mask_url(url))


@router.put(
    "/database",
    response_model=DatabaseConnectionSaveResponse,
    summary="Save or update personal database connection",
    description=(
        "Accepts any PostgreSQL connection string (Supabase, Neon, Railway, RDS, etc.). "
        "Validates connectivity before saving. Stored encrypted at rest. "
        "Once saved, your queries run against your own database instead of the server default."
    ),
)
async def save_database_connection(
    body: SaveDatabaseRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> DatabaseConnectionSaveResponse:
    from sqlalchemy import text

    from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
    from nl_to_sql.infrastructure.database.url_utils import to_async_database_url

    raw = body.database_url.strip()
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Database URL cannot be empty.")

    normalised = to_async_database_url(raw)

    # Validate the connection before saving
    try:
        tmp = AsyncDatabaseClient(database_url=normalised)
        async with tmp.session() as sess:
            await sess.execute(text("SELECT 1"))
        await tmp.dispose()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid connection string format. Check host, port, username and password.",
        ) from exc
    except Exception as exc:
        logger.warning("Database connection test failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not connect to the database. Check your credentials and host.",
        ) from exc

    svc = _get_user_db_service()
    if svc is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service unavailable.")

    await svc.save(current_user.id, normalised)
    return DatabaseConnectionSaveResponse(
        url_preview=_mask_url(normalised),
        message="Database connection saved. Your queries will now use this database.",
    )


@router.delete(
    "/database",
    response_model=DatabaseConnectionDeleteResponse,
    summary="Remove personal database connection",
    description="Removes your stored connection. Queries will fall back to the server default database.",
)
async def delete_database_connection(
    current_user: UserPublic = Depends(get_current_user),
) -> DatabaseConnectionDeleteResponse:
    svc = _get_user_db_service()
    if svc is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service unavailable.")
    await svc.delete(current_user.id)
    return DatabaseConnectionDeleteResponse(message="Database connection removed. Falling back to server default.")
