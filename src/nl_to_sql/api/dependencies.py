"""FastAPI dependency providers — bridge between DI container and route handlers."""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from nl_to_sql.config.container import ApplicationContainer
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.rag.ingestion.pipeline import IngestionPipeline
from nl_to_sql.services.api_key_service import APIKeyService
from nl_to_sql.services.chat_session_service import ChatSessionService
from nl_to_sql.services.query_history import QueryHistoryService
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.schema_catalog_service import SchemaCatalogService
from nl_to_sql.services.schema_ingestion import SchemaIngestionService
from nl_to_sql.services.user_db_service import UserDbConnectionService


@lru_cache(maxsize=1)
def _get_container() -> ApplicationContainer:
    """Return the singleton DI container."""
    container = ApplicationContainer()
    return container


def get_container() -> ApplicationContainer:
    """Dependency: Return the DI container itself (for config routes)."""
    return _get_container()


def get_orchestrator() -> QueryOrchestrator:
    """Dependency: QueryOrchestrator (main pipeline entry point)."""
    return _get_container().query_orchestrator()


def get_schema_ingestion() -> SchemaIngestionService:
    """Dependency: SchemaIngestionService."""
    return _get_container().schema_ingestion()


def get_schema_catalog() -> SchemaCatalogService:
    """Dependency: SchemaCatalogService (per-user schema catalog)."""
    return _get_container().schema_catalog_service()


def get_vector_store() -> IVectorStore:
    """Dependency: IVectorStore (for health checks and status)."""
    return _get_container().vector_store()


def get_db_client() -> AsyncDatabaseClient:
    """Dependency: AsyncDatabaseClient (for health checks and execution)."""
    return _get_container().db_client()


def get_llm_provider() -> ILLMProvider:
    """Dependency: ILLMProvider (for runtime provider switching)."""
    return _get_container().llm_provider()


def get_query_history() -> QueryHistoryService:
    """Dependency: QueryHistoryService (for query history access)."""
    return _get_container().query_history()


def get_session_service() -> ChatSessionService:
    """Dependency: ChatSessionService (for chat session management)."""
    return _get_container().session_service()


def get_ingestion_pipeline() -> IngestionPipeline:
    """Dependency: IngestionPipeline (for schema refresh from live DB)."""
    return _get_container().ingestion_pipeline()


def get_api_key_service() -> APIKeyService:
    """Dependency: APIKeyService (for per-user API key management)."""
    return _get_container().api_key_service()


def get_user_db_service() -> UserDbConnectionService:
    """Dependency: UserDbConnectionService (per-user BYOD connections)."""
    return _get_container().user_db_service()


# Must be defined before get_request_orchestrator — used as a default arg (evaluated at definition time)
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_request_orchestrator(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer_scheme)] = None,
) -> QueryOrchestrator:
    """Build a QueryOrchestrator with the right LLM provider for this request.

    Priority order:
      1. Authenticated user's personal API key (if stored for the active provider)
      2. Server's configured key
    Falls back silently — never raises an exception from this dependency.
    """
    from nl_to_sql.config.settings import get_settings
    from nl_to_sql.services.sql_generator import SQLGeneratorService

    container = _get_container()
    settings = get_settings()

    llm_provider = container.llm_provider()  # server default (reflects any runtime switch)

    if credentials is not None:
        try:
            import types

            from nl_to_sql.config.container import ApplicationContainer, _create_llm_provider
            from nl_to_sql.services.auth_service import decode_access_token

            token_data = decode_access_token(credentials.credentials)
            api_key_svc = container.api_key_service()

            # Use the *currently active* provider/model (may differ from startup env values
            # if the user switched at runtime via PUT /api/v1/config/llm).
            active = ApplicationContainer.get_current_llm_config(container)
            active_provider = active["provider"]
            active_model = active["model"]

            user_key = await api_key_svc.get_key(token_data.user_id, active_provider)
            if user_key:
                patched = types.SimpleNamespace(
                    groq_api_key=user_key if active_provider == "groq" else settings.groq_api_key,
                    openai_api_key=user_key if active_provider == "openai" else settings.openai_api_key,
                    anthropic_api_key=user_key if active_provider == "anthropic" else settings.anthropic_api_key,
                    gemini_api_key=user_key if active_provider == "gemini" else settings.gemini_api_key,
                )
                llm_provider = _create_llm_provider(active_provider, active_model, patched)  # type: ignore[arg-type]
        except Exception:
            pass  # Silently fall through to server key

    # Resolve per-user database client (BYOD) — falls back to server default
    db_client = container.db_client()
    if credentials is not None:
        try:
            from nl_to_sql.services.auth_service import decode_access_token
            token_data = decode_access_token(credentials.credentials)
            user_db_svc = container.user_db_service()
            user_client = await user_db_svc.get_client(token_data.user_id)
            if user_client is not None:
                db_client = user_client
        except Exception:
            pass  # Silently fall through to server default

    sql_generator = SQLGeneratorService(
        llm_provider=llm_provider,
        dialect=settings.sql_dialect,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        feedback_learner=container.feedback_learner(),
    )

    # Resolve user_id for per-user cache isolation
    resolved_user_id: str | None = None
    if credentials is not None:
        try:
            from nl_to_sql.services.auth_service import decode_access_token
            _td = decode_access_token(credentials.credentials)
            resolved_user_id = _td.user_id
        except Exception:
            pass

    # Per-user schema retrieval isolation (flag-gated). When enabled, scope every
    # vector-store read to the authenticated user's chunks.
    schema_retriever = container.schema_retriever()
    if settings.schema_per_user_isolation and resolved_user_id is not None:
        schema_retriever._user_id = resolved_user_id

    # Apply the *live* Phase-3 RAG flags (runtime-adjustable via PUT /config/rag)
    # to this request's retriever. HyDE uses the per-request provider so it
    # honours a caller's personal API key.
    schema_retriever._multi_query_enabled = settings.rag_multi_query_enabled
    schema_retriever._multi_query_max = settings.rag_multi_query_max
    schema_retriever._hyde_enabled = settings.rag_hyde_enabled
    schema_retriever._llm_provider = llm_provider

    return QueryOrchestrator(
        retriever=schema_retriever,
        generator=sql_generator,
        validator=container.sql_validator(),
        cache=container.active_cache(),
        max_retries=settings.sql_max_retries,
        db_client=db_client,
        query_history=container.query_history(),
        query_classifier=container.query_classifier(),
        session_service=container.session_service(),
        training_data_service=container.training_data_service(),
        table_selector=container.table_selector(),
        fk_extractor=container.fk_extractor(),
        column_validator=container.column_validator(),
        user_id=resolved_user_id,
        example_store=container.example_store(),
        few_shot_enabled=settings.rag_few_shot_retrieval_enabled,
        few_shot_top_k=settings.rag_few_shot_top_k,
        adaptive_top_k_enabled=settings.rag_adaptive_top_k_enabled,
        top_k_min=settings.rag_adaptive_top_k_min,
        top_k_max=settings.rag_adaptive_top_k_max,
    )


# ── Auth Dependency ────────────────────────────────────────────────────────────

_AUTH_CACHE_TTL = 45  # seconds
_auth_cache: dict[str, tuple[float, UserPublic]] = {}


def _auth_cache_get(cache_key: str) -> UserPublic | None:
    entry = _auth_cache.get(cache_key)
    if entry and time.monotonic() - entry[0] < _AUTH_CACHE_TTL:
        return entry[1]
    _auth_cache.pop(cache_key, None)
    return None


def _auth_cache_set(cache_key: str, user: UserPublic) -> None:
    if len(_auth_cache) > 4096:
        # Evict oldest quarter to bound memory use
        oldest = sorted(_auth_cache, key=lambda k: _auth_cache[k][0])[: len(_auth_cache) // 4]
        for k in oldest:
            _auth_cache.pop(k, None)
    _auth_cache[cache_key] = (time.monotonic(), user)


def auth_cache_invalidate_session(user_id: str, session_id: str | None) -> None:
    """Remove a specific session from the auth cache (call on logout/revoke)."""
    _auth_cache.pop(f"{user_id}:{session_id}", None)


def auth_cache_invalidate_user(user_id: str) -> None:
    """Remove all cache entries for a user (call on revoke-all or deactivation)."""
    keys = [k for k in _auth_cache if k.startswith(f"{user_id}:")]
    for k in keys:
        _auth_cache.pop(k, None)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer_scheme)] = None,
) -> UserPublic:
    """FastAPI dependency: validate Bearer JWT and return the current user.

    Raises HTTP 401 if the token is missing, malformed, or expired.

    Usage in route:
        current_user: UserPublic = Depends(get_current_user)
    """
    from jose import JWTError
    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import User
    from nl_to_sql.services.auth_service import decode_access_token

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        token_data = decode_access_token(credentials.credentials)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    cache_key = f"{token_data.user_id}:{token_data.session_id}"
    cached = _auth_cache_get(cache_key)
    if cached is not None:
        return cached

    from nl_to_sql.infrastructure.database.models import UserLoginSession

    session_svc: ChatSessionService = _get_container().session_service()
    async with session_svc._session_factory() as db_sess:
        result = await db_sess.execute(
            select(User).where(User.id == token_data.user_id, User.is_active.is_(True))
        )
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or deactivated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if token_data.session_id:
            sess_result = await db_sess.execute(
                select(UserLoginSession).where(
                    UserLoginSession.id == token_data.session_id,
                    UserLoginSession.revoked_at.is_(None),
                )
            )
            if sess_result.scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )

    user_public = UserPublic.model_validate(user)
    _auth_cache_set(cache_key, user_public)
    return user_public


async def require_admin(
    current_user: UserPublic = Security(get_current_user),
) -> UserPublic:
    """FastAPI dependency: requires the current user to be an admin.

    Admins are defined via the ADMIN_EMAILS setting (comma-separated list).
    Returns the user on success; raises HTTP 403 otherwise.
    """
    from nl_to_sql.config.settings import get_settings
    settings = get_settings()
    if current_user.email.lower() not in settings.admin_email_list:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user
