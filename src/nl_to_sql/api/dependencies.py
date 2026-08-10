"""FastAPI dependency providers — bridge between DI container and route handlers."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated

import structlog
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
from nl_to_sql.services.connection_service import ConnectionService
from nl_to_sql.services.dashboard_service import DashboardService
from nl_to_sql.services.export_service import ExportService
from nl_to_sql.services.metrics_service import MetricsService
from nl_to_sql.services.query_history import QueryHistoryService
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.scheduled_query_service import ScheduledQueryService
from nl_to_sql.services.schema_catalog_service import SchemaCatalogService
from nl_to_sql.services.schema_doc_service import SchemaDocService
from nl_to_sql.services.schema_ingestion import SchemaIngestionService
from nl_to_sql.services.starter_content_service import StarterContentService

logger = structlog.get_logger(__name__)


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


def get_schema_doc_service() -> SchemaDocService:
    """Dependency: SchemaDocService (RAG-powered schema explanations)."""
    return _get_container().schema_doc_service()


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


def get_export_service() -> ExportService:
    """Dependency: ExportService (query export + share-link delivery)."""
    return _get_container().export_service()


def get_dashboard_service() -> DashboardService:
    """Dependency: DashboardService (per-user dashboards + chart recommendation)."""
    return _get_container().dashboard_service()


def get_scheduled_query_service() -> ScheduledQueryService:
    """Dependency: ScheduledQueryService (per-user recurring NL queries + alerts)."""
    return _get_container().scheduled_query_service()


def get_metrics_service() -> MetricsService:
    """Dependency: MetricsService (connection-scoped certified metrics catalog)."""
    return _get_container().metrics_service()


def get_starter_content_service() -> StarterContentService:
    """Dependency: StarterContentService (seeds built-in example content)."""
    return _get_container().starter_content_service()


# Connections whose vectors have already been checked/self-healed this process
# (bounds the lazy re-embed to one Qdrant round-trip per connection per worker).
_vectors_checked: set[str] = set()


async def resolve_active_connection(
    credentials: HTTPAuthorizationCredentials | None,
) -> tuple[str | None, str | None, AsyncDatabaseClient]:
    """Resolve (user_id, connection_id, db_client) for the current request.

    The active connection is the user's persisted default. A connection with no
    stored DSN (the built-in "Server Default") resolves the client to the
    platform database. Never raises — falls back to the server default client.

    ``(None, None, db_client)`` means "no active connection" and is returned
    for two different reasons: genuinely no/invalid credentials, or an
    authenticated user whose connection lookup itself failed (C7 — e.g. a
    transient DB error). Both cases fail CLOSED to the same shared-only scope;
    the latter is additionally logged, since it's a real failure worth seeing
    rather than a normal anonymous request. Every downstream per-tenant filter
    (vector store, example store, semantic cache) must treat a ``None``
    ``connection_id`` as "shared-only", never as "unrestricted" — that
    invariant, not this function, is what actually prevents a resolution
    failure from turning into a cross-tenant read.
    """
    container = _get_container()
    db_client = container.db_client()
    if credentials is None:
        return None, None, db_client

    from nl_to_sql.services.auth_service import decode_access_token

    try:
        token_data = decode_access_token(credentials.credentials)
    except Exception:
        # Invalid/expired token — equivalent to no credentials: anonymous, shared-only.
        return None, None, db_client

    try:
        conn_svc = container.connection_service()
        connection_id = await conn_svc.get_active_connection_id(token_data.user_id)
        user_client = await conn_svc.get_client(connection_id)
        if user_client is not None:
            db_client = user_client
        return token_data.user_id, connection_id, db_client
    except Exception as exc:
        logger.warning(
            "resolve_active_connection: connection lookup failed for an "
            "authenticated user — failing closed to shared-only scope",
            user_id=token_data.user_id,
            error=str(exc),
        )
        return None, None, db_client


async def _maybe_self_heal_vectors(user_id: str | None, connection_id: str | None) -> None:
    """One-time lazy re-embed of a connection whose vector store is empty."""
    if not user_id or not connection_id or connection_id in _vectors_checked:
        return
    _vectors_checked.add(connection_id)
    try:
        container = _get_container()
        count = await container.vector_store().count(connection_id=connection_id)
        await container.schema_catalog_service().ensure_embedded(user_id, connection_id, count)
    except Exception:
        pass  # non-fatal — retrieval will surface an empty-schema error if truly empty


async def get_request_db_client(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer_scheme)] = None,
) -> AsyncDatabaseClient:
    """Resolve the active-connection database client, falling back to the server default.

    Mirrors the DB-client resolution inside ``get_request_orchestrator`` so
    widget "refresh" re-runs each SQL against the authenticated user's currently
    selected database connection.
    """
    _user_id, _connection_id, db_client = await resolve_active_connection(credentials)
    return db_client


def get_ingestion_pipeline() -> IngestionPipeline:
    """Dependency: IngestionPipeline (for schema refresh from live DB)."""
    return _get_container().ingestion_pipeline()


def get_api_key_service() -> APIKeyService:
    """Dependency: APIKeyService (for per-user API key management)."""
    return _get_container().api_key_service()


def get_connection_service() -> ConnectionService:
    """Dependency: ConnectionService (per-user multiple BYOD connections)."""
    return _get_container().connection_service()


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

    The actual ~15-service assembly is shared with the no-HTTP-context worker
    path via :func:`nl_to_sql.services.orchestrator_factory.build_orchestrator`
    (moved out of this module so ``workers/`` can reuse it without importing
    ``api/`` — see that module's docstring).
    """
    from nl_to_sql.config.settings import get_settings
    from nl_to_sql.services.orchestrator_factory import build_orchestrator

    container = _get_container()
    settings = get_settings()

    llm_provider = container.llm_provider()  # server default (reflects any runtime switch)

    if credentials is not None:
        try:
            from nl_to_sql.config.container import ApplicationContainer, create_llm_provider
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
                # Build a real Settings with the active provider's key swapped for the
                # user's personal key, so create_llm_provider gets the expected type.
                patched = settings.model_copy(update={f"{active_provider}_api_key": user_key})
                llm_provider = create_llm_provider(active_provider, active_model, patched)
        except Exception:
            pass  # Silently fall through to server key

    # Resolve the active connection (user_id, connection_id, client). Falls back
    # to the server default client for the built-in "Server Default" connection.
    resolved_user_id, resolved_connection_id, db_client = await resolve_active_connection(
        credentials
    )
    # Self-heal legacy connections whose vectors predate connection_id scoping.
    await _maybe_self_heal_vectors(resolved_user_id, resolved_connection_id)

    return build_orchestrator(
        container, settings, llm_provider, resolved_user_id, resolved_connection_id, db_client
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


async def get_active_connection_id(
    current_user: UserPublic = Security(get_current_user),
) -> str:
    """Dependency: the current user's active (default) connection id.

    Ensures the user always has at least one connection (a Server Default is
    created on demand), so routes can rely on a non-null connection id.
    """
    return await _get_container().connection_service().get_active_connection_id(current_user.id)


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
