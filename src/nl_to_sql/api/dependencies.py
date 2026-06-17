"""FastAPI dependency providers — bridge between DI container and route handlers."""
from __future__ import annotations

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
from nl_to_sql.services.chat_session_service import ChatSessionService
from nl_to_sql.services.query_history import QueryHistoryService
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.schema_ingestion import SchemaIngestionService


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


def get_api_key_service():
    """Dependency: APIKeyService (for per-user API key management)."""
    return _get_container().api_key_service()


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
    from nl_to_sql.infrastructure.llm.groq_provider import GroqProvider
    from nl_to_sql.services.sql_generator import SQLGeneratorService

    container = _get_container()
    settings = get_settings()

    llm_provider = container.llm_provider()  # server default

    if credentials is not None:
        try:
            from nl_to_sql.services.auth_service import decode_access_token
            token_data = decode_access_token(credentials.credentials)
            api_key_svc = container.api_key_service()
            user_key = await api_key_svc.get_key(token_data.user_id, settings.llm_provider)
            if user_key:
                if settings.llm_provider == "openai":
                    from nl_to_sql.infrastructure.llm.openai_provider import OpenAIProvider
                    llm_provider = OpenAIProvider(api_key=user_key, model=settings.llm_model)
                else:
                    llm_provider = GroqProvider(api_key=user_key, model=settings.llm_model)
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

    return QueryOrchestrator(
        retriever=container.schema_retriever(),
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
    )


# ── Auth Dependency ────────────────────────────────────────────────────────────


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

    # Verify user still exists and is active, and that their login session has not been revoked
    from nl_to_sql.infrastructure.database.models import User, UserLoginSession

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

    return UserPublic.model_validate(user)
