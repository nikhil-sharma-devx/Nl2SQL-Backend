"""Dependency Injection container — wires all services via dependency-injector.

Object construction logic lives in `config/factories/`; this module is the
thin wiring layer that composes those factories into providers.
"""
from typing import Any

from dependency_injector import containers, providers

from nl_to_sql.config.factories import (
    build_cache,
    build_embedder,
    build_llm_provider,
    build_vector_store,
    create_llm_provider,
)
from nl_to_sql.config.settings import Settings
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.infrastructure.bm25_store import BM25Store
from nl_to_sql.infrastructure.cache.semantic_cache import SemanticCache
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.rag.ingestion.pipeline import IngestionPipeline
from nl_to_sql.rag.retrieval.fk_extractor import FKRelationshipExtractor
from nl_to_sql.rag.retrieval.retrieval_chain import RetrievalChain
from nl_to_sql.rag.retrieval.table_selector import TableSelectorService
from nl_to_sql.services.analytics_service import AnalyticsService
from nl_to_sql.services.api_key_service import APIKeyService
from nl_to_sql.services.chat_session_service import ChatSessionService
from nl_to_sql.services.feedback_learner import FeedbackLearner
from nl_to_sql.services.feedback_service import FeedbackService
from nl_to_sql.services.fine_tuning_service import FineTuningService
from nl_to_sql.services.prompt_manager import PromptManager
from nl_to_sql.services.query_classifier import QueryClassifier
from nl_to_sql.services.query_expander import QueryExpander
from nl_to_sql.services.query_history import QueryHistoryService
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.query_rewriter import QueryRewriter
from nl_to_sql.services.reranker import CrossEncoderReranker
from nl_to_sql.services.schema_catalog_service import SchemaCatalogService
from nl_to_sql.services.schema_ingestion import SchemaIngestionService
from nl_to_sql.services.schema_monitor import SchemaMonitor
from nl_to_sql.services.schema_retriever import SchemaRetriever
from nl_to_sql.services.sql_column_validator import SQLColumnValidator
from nl_to_sql.services.sql_generator import SQLGeneratorService
from nl_to_sql.services.sql_validator import SQLValidatorService
from nl_to_sql.services.training_data_service import TrainingDataService
from nl_to_sql.services.ttl_manager import TTLManager
from nl_to_sql.services.user_db_service import UserDbConnectionService


class ApplicationContainer(containers.DeclarativeContainer):
    """Top-level IoC container.

    All application services are defined here as providers. FastAPI route
    handlers receive dependencies via `api/dependencies.py` which unpacks
    this container.

    SOLID: D — High-level orchestrators depend on injected abstractions.
    """

    # ── Configuration ─────────────────────────────────────────────────────────
    config = providers.Singleton(Settings)

    # ── Runtime mutable state for LLM provider ────────────────────────────────
    _current_llm_provider: providers.Object[Any] = providers.Object(None)
    _current_llm_model: providers.Object[Any] = providers.Object(None)

    # ── Infrastructure ────────────────────────────────────────────────────────
    llm_provider = providers.Singleton(
        build_llm_provider,
        settings=config,
    )

    embedder = providers.Singleton(
        build_embedder,
        settings=config,
    )

    vector_store = providers.Singleton(
        build_vector_store,
        settings=config,
    )

    cache = providers.Singleton(
        build_cache,
        settings=config,
    )

    semantic_cache = providers.Singleton(
        SemanticCache,
        embedder=embedder,
        vector_store=vector_store,
        exact_cache=cache,
        threshold=config.provided.semantic_cache_threshold,
        collection_name=config.provided.semantic_cache_collection,
    )

    active_cache = providers.Selector(
        providers.Callable(lambda s: str(s.semantic_cache_enabled), config),
        **{
            "True": semantic_cache,
            "False": cache,
        }
    )

    db_client = providers.Singleton(
        AsyncDatabaseClient,
        database_url=config.provided.database_url,
        pool_size=config.provided.db_pool_size,
        max_overflow=config.provided.db_max_overflow,
        pool_timeout=config.provided.db_pool_timeout,
        pool_recycle=config.provided.db_pool_recycle,
        readonly=config.provided.target_db_readonly,
        statement_timeout_ms=config.provided.target_statement_timeout_ms,
    )

    # ── Services ──────────────────────────────────────────────────────────────

    # ── User DB Connection Service (BYOD — per-user encrypted database URLs) ─
    user_db_service = providers.Singleton(
        UserDbConnectionService,
        database_url=config.provided.history_database_url,
        secret_key=config.provided.secret_key,
    )

    _use_hybrid_search = providers.Callable(
        lambda s: s.vector_store_provider == "qdrant",
        config,
    )

    _bm25_effective = providers.Callable(
        lambda s: s.bm25_enabled and s.vector_store_provider != "qdrant",
        config,
    )

    schema_retriever = providers.Factory(
        SchemaRetriever,
        embedder=embedder,
        vector_store=vector_store,
        top_k=config.provided.vector_store_top_k,
        use_hybrid_search=_use_hybrid_search,
        hybrid_alpha=config.provided.hybrid_search_alpha,
    )

    # ── Training Data Service (defined early so feedback_learner can reference it) ──
    training_data_service = providers.Singleton(
        TrainingDataService,
        database_url=config.provided.history_database_url,
    )

    # ── Feedback Learner (MUST be before sql_generator) ──────────────────────
    feedback_learner = providers.Singleton(
        FeedbackLearner,
        feedback_service=None,
        training_data_service=training_data_service,
    )

    sql_generator = providers.Factory(
        SQLGeneratorService,
        llm_provider=llm_provider,
        dialect=config.provided.sql_dialect,
        temperature=config.provided.llm_temperature,
        max_tokens=config.provided.llm_max_tokens,
        feedback_learner=feedback_learner,
    )

    sql_validator = providers.Singleton(
        SQLValidatorService,
        dialect=config.provided.sql_dialect,
    )

    schema_ingestion = providers.Factory(
        SchemaIngestionService,
        embedder=embedder,
        vector_store=vector_store,
    )

    # ── BM25 Store ───────────────────────────────────────────────────────────
    bm25_store = providers.Singleton(
        BM25Store,
        index_path=config.provided.bm25_index_path,
    )

    # ── RAG: Ingestion Pipeline ──────────────────────────────────────────────
    ingestion_pipeline = providers.Factory(
        IngestionPipeline,
        db_client=db_client,
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
        chunk_strategy=config.provided.chunk_strategy,
        chunk_size=config.provided.chunk_size,
        chunk_overlap=config.provided.chunk_overlap,
        embedding_batch_size=config.provided.embedding_batch_size,
        bm25_enabled=_bm25_effective,
    )

    # ── RAG: Retrieval Chain ─────────────────────────────────────────────────
    retrieval_chain = providers.Factory(
        RetrievalChain,
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
        top_k=config.provided.vector_store_top_k,
        use_hybrid_search=_use_hybrid_search,
        hybrid_alpha=config.provided.hybrid_search_alpha,
        bm25_enabled=_bm25_effective,
        bm25_top_k=config.provided.bm25_top_k,
        reranker_enabled=config.provided.reranker_enabled,
        reranker_model=config.provided.reranker_model,
        reranker_top_k=config.provided.reranker_top_k,
    )

    # ── History ──────────────────────────────────────────────────────────────
    query_history = providers.Singleton(
        QueryHistoryService,
        database_url=config.provided.history_database_url,
    )

    # ── Chat Sessions ─────────────────────────────────────────────────────────
    session_service = providers.Singleton(
        ChatSessionService,
        database_url=config.provided.history_database_url,
    )

    # ── API Key Service (BYOK — per-user encrypted keys) ─────────────────────
    api_key_service = providers.Singleton(
        APIKeyService,
        session_factory=session_service.provided._session_factory,
        secret_key=config.provided.secret_key,
    )

    # ── Query Classification ─────────────────────────────────────────────────
    query_classifier = providers.Singleton(QueryClassifier)

    # ── TTL Manager ──────────────────────────────────────────────────────────
    ttl_manager = providers.Singleton(
        TTLManager,
        base_ttl=config.provided.cache_ttl_seconds,
    )

    # ── Query Expander ───────────────────────────────────────────────────────
    query_expander = providers.Singleton(
        QueryExpander,
        use_synonyms=config.provided.query_expansion_enabled,
    )

    # ── Query Rewriter ───────────────────────────────────────────────────────
    query_rewriter = providers.Singleton(
        QueryRewriter,
        llm_provider=llm_provider,
        enabled=config.provided.query_rewriting_enabled,
    )

    # ── Prompt Manager ───────────────────────────────────────────────────────
    prompt_manager = providers.Singleton(
        PromptManager,
        ab_testing_enabled=True,
    )

    # ── Re-ranker ────────────────────────────────────────────────────────────
    reranker = providers.Singleton(
        CrossEncoderReranker,
        model_name=config.provided.reranker_model,
        top_k=config.provided.reranker_top_k,
        enabled=config.provided.reranker_enabled,
    )

    # ── Fine-Tuning Service ──────────────────────────────────────────────────
    fine_tuning_service = providers.Singleton(
        FineTuningService,
        provider=config.provided.fine_tuning_provider,
        openai_api_key=config.provided.openai_api_key,
        together_api_key=config.provided.together_api_key,
        training_data_service=training_data_service,
    )

    # ── Table Selector (Phase B of two-phase schema grounding) ───────────────
    table_selector = providers.Singleton(
        TableSelectorService,
        llm_provider=llm_provider,
        temperature=0.0,
        max_tokens=256,
    )

    # ── FK Relationship Extractor (Layer 1: FK-Aware Retrieval) ──────────────
    fk_extractor = providers.Singleton(
        FKRelationshipExtractor,
        vector_store=vector_store,
        schema_metadata=None,  # Will be populated during ingestion
    )

    # ── SQL Column Validator (Layer 2: Column Validation) ────────────────────
    column_validator = providers.Singleton(
        SQLColumnValidator,
        dialect=config.provided.sql_dialect,
    )

    # ── Schema Catalog Service (per-user schema management source of truth) ──
    schema_catalog_service = providers.Singleton(
        SchemaCatalogService,
        session_factory=session_service.provided._session_factory,
        ingestion=schema_ingestion,
        per_user_isolation=config.provided.schema_per_user_isolation,
    )

    # ── Schema Monitor ───────────────────────────────────────────────────────
    schema_monitor = providers.Singleton(
        SchemaMonitor,
        db_client=db_client,
        ingestion_service=schema_ingestion,
        check_interval=config.provided.schema_monitor_interval_seconds,
        enabled=config.provided.schema_monitor_enabled,
    )

    # ── Analytics Service ────────────────────────────────────────────────────
    analytics_service = providers.Singleton(
        AnalyticsService,
        database_url=config.provided.history_database_url,
    )

    # ── Feedback Service ─────────────────────────────────────────────────────
    feedback_service = providers.Singleton(
        FeedbackService,
        database_url=config.provided.history_database_url,
    )

    query_orchestrator = providers.Factory(
        QueryOrchestrator,
        retriever=schema_retriever,
        generator=sql_generator,
        validator=sql_validator,
        cache=active_cache,
        max_retries=config.provided.sql_max_retries,
        db_client=db_client,
        query_history=query_history,
        query_classifier=query_classifier,
        session_service=session_service,
        training_data_service=training_data_service,
        table_selector=table_selector,
        fk_extractor=fk_extractor,
        column_validator=column_validator,
    )

    @staticmethod
    def switch_llm_provider(
        container: "ApplicationContainer",
        provider: str,
        model: str,
    ) -> ILLMProvider:
        """Switch the LLM provider at runtime.

        Creates a new provider instance and overrides the singleton.
        Also updates the runtime state tracking.

        Args:
            container: The DI container instance.
            provider: The provider name ("groq").
            model: The model name to use.

        Returns:
            The new ILLMProvider instance.
        """
        settings: Settings = container.config()
        new_provider = create_llm_provider(provider, model, settings)

        # Override the singleton with the new instance
        container.llm_provider.override(providers.Object(new_provider))

        # Update runtime state tracking
        container._current_llm_provider.override(providers.Object(provider))
        container._current_llm_model.override(providers.Object(model))

        return new_provider

    @staticmethod
    async def switch_db_client(
        container: "ApplicationContainer",
        database_url: str,
    ) -> AsyncDatabaseClient:
        """Switch the database client connection string at runtime.

        Args:
            container: The DI container instance.
            database_url: The new database connection string.

        Returns:
            The AsyncDatabaseClient instance.
        """
        client = container.db_client()
        await client.switch_engine(database_url)

        # Update settings to keep them in sync
        settings: Settings = container.config()
        settings.database_url = database_url

        return client

    @staticmethod
    def get_current_llm_config(container: "ApplicationContainer") -> dict[str, Any]:
        """Get the current LLM configuration.

        Returns runtime overrides if set, otherwise falls back to settings.

        Args:
            container: The DI container instance.

        Returns:
            Dictionary with "provider" and "model" keys.
        """
        settings: Settings = container.config()

        # Check for runtime overrides
        current_provider = container._current_llm_provider()
        current_model = container._current_llm_model()

        if current_provider is not None:
            return {
                "provider": current_provider,
                "model": current_model,
            }

        # Fall back to settings
        return {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }
