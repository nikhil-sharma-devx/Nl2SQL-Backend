"""Shared QueryOrchestrator assembly.

Used by both the HTTP-request path (``api/dependencies.py``'s
``get_request_orchestrator``, which resolves the LLM provider / user /
connection / DB client from a bearer token) and the no-HTTP-context worker
path (``workers/scheduled_query_worker.py``, which resolves them directly
from a stored ``ScheduledQuery``) — so both construct the identical ~15-
service wiring and can never drift apart.

Lives in ``services/`` (not ``api/dependencies.py``) so workers can import it
without violating the layering rule that workers never import from ``api/``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.sql_generator import SQLGeneratorService

if TYPE_CHECKING:
    from nl_to_sql.config.container import ApplicationContainer
    from nl_to_sql.config.settings import Settings


def build_orchestrator(
    container: ApplicationContainer,
    settings: Settings,
    llm_provider: ILLMProvider,
    user_id: str | None,
    connection_id: str | None,
    db_client: AsyncDatabaseClient,
) -> QueryOrchestrator:
    """Assemble a QueryOrchestrator from an already-resolved provider/connection/client."""
    sql_generator = SQLGeneratorService(
        llm_provider=llm_provider,
        dialect=settings.sql_dialect,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        feedback_learner=container.feedback_learner(),
    )

    # Per-connection schema retrieval isolation (flag-gated). When enabled, scope
    # every vector-store read to the active connection's chunks so retrieval never
    # sees another connection's tables.
    schema_retriever = container.schema_retriever()
    if settings.schema_per_user_isolation and connection_id is not None:
        schema_retriever._connection_id = connection_id

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
        user_id=user_id,
        connection_id=connection_id,
        example_store=container.example_store(),
        few_shot_enabled=settings.rag_few_shot_retrieval_enabled,
        few_shot_top_k=settings.rag_few_shot_top_k,
        adaptive_top_k_enabled=settings.rag_adaptive_top_k_enabled,
        top_k_min=settings.rag_adaptive_top_k_min,
        top_k_max=settings.rag_adaptive_top_k_max,
        query_rewriter=container.query_rewriter(),
        correction_detector=container.correction_detector(),
        conversation_max_turns=settings.conversation_max_turns,
    )


async def build_orchestrator_for_connection(
    container: ApplicationContainer,
    settings: Settings,
    user_id: str,
    connection_id: str,
) -> QueryOrchestrator:
    """Build an orchestrator with no HTTP context (e.g. for a scheduled-query worker).

    Always uses the server's configured LLM provider (``container.llm_provider()``)
    — there is no bearer token at cron time to carry a personal API-key
    preference. The DB client is resolved via ``ConnectionService.get_client``
    with the same server-default fallback as the live request path, so a
    schedule against the "Server Default" connection behaves identically to an
    interactive query against it.
    """
    llm_provider = container.llm_provider()
    conn_svc = container.connection_service()
    db_client = await conn_svc.get_client(user_id, connection_id)
    if db_client is None:
        db_client = container.db_client()
    return build_orchestrator(
        container, settings, llm_provider, user_id, connection_id, db_client
    )
