"""Configuration routes — GET/PUT /api/v1/config/llm, GET /api/v1/config/models, GET/PUT /api/v1/config/database."""
import ipaddress
import re
import socket

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from nl_to_sql.api.dependencies import get_container, get_current_user, require_admin
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.config.container import ApplicationContainer
from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.core.models.config import (
    AvailableModelsResponse,
    DatabaseConfigResponse,
    DatabaseConfigUpdate,
    DatabaseConfigUpdateResponse,
    LLMConfigResponse,
    LLMConfigUpdate,
    LLMConfigUpdateResponse,
)
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["Configuration"])
_settings = get_settings()
_rate = f"{_settings.rate_limit_requests}/minute"

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _reject_ssrf_host(url: str) -> None:
    """Resolve URL hostname and raise HTTP 400 if it targets a private/internal IP."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        if not host:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid database URL: missing host.")
        addrs = socket.getaddrinfo(host, None)
        for addr_info in addrs:
            ip = ipaddress.ip_address(addr_info[4][0])
            if any(ip in net for net in _PRIVATE_NETS):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Database URL targets a reserved or internal network address.",
                )
    except HTTPException:
        raise
    except OSError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database host could not be resolved.",
        ) from None


def _mask_url(url: str) -> str:
    return re.sub(r":([^:@]+)@", ":***@", url)


# Available models per provider
AVAILABLE_MODELS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
}

_AVAILABLE_PROVIDERS = ["groq", "openai"]


@router.get(
    "/llm",
    response_model=LLMConfigResponse,
    summary="Get current LLM configuration",
    description="Returns the currently active LLM provider and model.",
)
async def get_llm_config(
    container: ApplicationContainer = Depends(get_container),
) -> LLMConfigResponse:
    """Get the current LLM provider and model configuration."""
    config = ApplicationContainer.get_current_llm_config(container)
    return LLMConfigResponse(
        provider=config["provider"],
        model=config["model"],
        available_providers=_AVAILABLE_PROVIDERS,
    )


@router.put(
    "/llm",
    response_model=LLMConfigUpdateResponse,
    summary="Update LLM configuration",
    description="Switch to a different LLM provider and/or model at runtime.",
)
@limiter.limit(_rate)
async def update_llm_config(
    request: Request,  # required by SlowAPI for IP extraction
    body: LLMConfigUpdate,
    container: ApplicationContainer = Depends(get_container),
    _admin: UserPublic = Depends(require_admin),
) -> LLMConfigUpdateResponse:
    """Update the LLM provider and model at runtime.

    Validates that the API key exists for the requested provider.
    """
    provider = body.provider.lower()
    model = body.model

    # Validate provider
    if provider not in _AVAILABLE_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider '{provider}'. Available: {_AVAILABLE_PROVIDERS}",
        )

    # Validate model exists for provider
    available_models = AVAILABLE_MODELS.get(provider, [])
    if model not in available_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{model}' for provider '{provider}'. "
            f"Available: {available_models}",
        )

    # Validate API key exists for the provider
    settings = container.config()
    if provider == "groq" and not settings.groq_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Groq API key is not configured. Set GROQ_API_KEY or add your key in Profile → API Keys.",
        )
    if provider == "openai" and not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OpenAI API key is not configured. Set OPENAI_API_KEY or add your key in Profile → API Keys.",
        )

    # Switch the provider
    ApplicationContainer.switch_llm_provider(container, provider, model)

    return LLMConfigUpdateResponse(
        provider=provider,
        model=model,
        message="LLM provider updated",
    )


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    summary="Get available models",
    description="Returns all available models grouped by provider.",
)
async def get_available_models() -> AvailableModelsResponse:
    """Get all available LLM models grouped by provider."""
    return AvailableModelsResponse(
        groq=AVAILABLE_MODELS["groq"],
        openai=AVAILABLE_MODELS["openai"],
    )


@router.get(
    "/database",
    response_model=DatabaseConfigResponse,
    summary="Get current database configuration",
    description="Returns the currently active target database connection string and available databases.",
)
async def get_database_config(
    container: ApplicationContainer = Depends(get_container),
    current_user: UserPublic = Depends(get_current_user),
) -> DatabaseConfigResponse:
    """Get the current database connection string and available DBs."""
    settings = container.config()
    return DatabaseConfigResponse(
        database_url=_mask_url(settings.database_url),
        available_databases={
            name: _mask_url(url)
            for name, url in settings.parsed_available_databases.items()
        },
    )


@router.put(
    "/database",
    response_model=DatabaseConfigUpdateResponse,
    summary="Update database connection",
    description="Switch to a different target database at runtime. Validates the connection before switching.",
)
@limiter.limit(_rate)
async def update_database_config(
    request: Request,
    body: DatabaseConfigUpdate,
    container: ApplicationContainer = Depends(get_container),
    _admin: UserPublic = Depends(require_admin),
) -> DatabaseConfigUpdateResponse:
    """Update the target database connection string at runtime.

    Creates a temporary client to verify the connection is valid before switching.
    """
    new_url = body.database_url.strip()

    if not new_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database URL cannot be empty.",
        )

    # Automatically add asyncpg driver if missing for postgresql
    if new_url.startswith("postgresql://"):
        new_url = new_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # asyncpg expects 'ssl=require' instead of 'sslmode=require'
    if "sslmode=" in new_url:
        new_url = new_url.replace("sslmode=", "ssl=")

    # Guard against SSRF before opening any network connection
    _reject_ssrf_host(new_url)

    # Verify the new connection works before switching
    try:
        temp_client = AsyncDatabaseClient(database_url=new_url)
        from sqlalchemy import text
        async with temp_client.session() as sess:
            await sess.execute(text("SELECT 1"))
        await temp_client.dispose()
    except HTTPException:
        raise
    except ValueError as exc:
        # Usually happens when SQLAlchemy fails to parse the URL (e.g. invalid port)
        logger.error("Database connection test failed (URL format)", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid database connection string format. Please ensure the port, username, password, and host are correct.",
        ) from exc
    except Exception as exc:
        logger.error("Database connection test failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database connection failed. Check the host, port, credentials, and SSL settings.",
        ) from exc

    # Switch target engine and dispose old pools actively
    await ApplicationContainer.switch_db_client(container, new_url)
    logger.info("Database connection updated at runtime")

    return DatabaseConfigUpdateResponse(
        database_url=_mask_url(new_url),
        message="Database connection updated successfully.",
    )
