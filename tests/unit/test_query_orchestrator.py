"""Unit tests for QueryOrchestrator."""
from unittest.mock import MagicMock

import pytest

from nl_to_sql.core.models.query import QueryRequest, QueryResponse
from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.schema_retriever import SchemaRetriever
from nl_to_sql.services.sql_generator import SQLGeneratorService
from nl_to_sql.services.sql_validator import SQLValidatorService


@pytest.fixture
def orchestrator(
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
    mock_llm_provider: MagicMock,
    in_memory_cache: InMemoryCache,
) -> QueryOrchestrator:
    retriever = SchemaRetriever(embedder=mock_embedder, vector_store=mock_vector_store, top_k=2)
    generator = SQLGeneratorService(llm_provider=mock_llm_provider, dialect="postgresql")
    validator = SQLValidatorService(dialect="postgresql")
    return QueryOrchestrator(
        retriever=retriever,
        generator=generator,
        validator=validator,
        cache=in_memory_cache,
        max_retries=2,
        connection_id="conn-test",  # real requests always resolve a connection
    )


class TestQueryOrchestrator:

    async def test_run_returns_valid_response(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        request = QueryRequest(question="Show me all customers")
        response = await orchestrator.run(request)
        assert isinstance(response, QueryResponse)
        assert response.sql != ""
        assert response.cached is False

    async def test_second_run_hits_cache(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        request = QueryRequest(question="Show me all customers from the US")
        # First call — miss
        r1 = await orchestrator.run(request)
        # Second call — hit (only if first was valid)
        if r1.is_valid:
            r2 = await orchestrator.run(request)
            assert r2.cached is True

    async def test_cache_key_is_case_insensitive(self) -> None:
        key1 = QueryOrchestrator._make_cache_key("Show all customers", "postgresql", None)
        key2 = QueryOrchestrator._make_cache_key("show all customers", "postgresql", None)
        assert key1 == key2

    async def test_cache_key_differs_by_dialect(self) -> None:
        key1 = QueryOrchestrator._make_cache_key("Count customers", "postgresql", None)
        key2 = QueryOrchestrator._make_cache_key("Count customers", "mysql", None)
        assert key1 != key2


class TestRunStreamMatchesRun:
    """run_stream() shares its schema-retrieval + conversation-history logic
    with run() via _retrieve_schema/_load_conversation_history — this locks
    in that the streaming path still reaches a `complete` event with a valid
    payload after that extraction."""

    async def test_run_stream_reaches_complete_with_valid_sql(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        request = QueryRequest(question="Show me all customers")
        events = [chunk async for chunk in orchestrator.run_stream(request)]
        assert events, "run_stream produced no events"
        stages = [e.get("stage") for e in events]
        assert "retrieving_schema" in stages
        assert "schema_retrieved" in stages
        statuses = [e.get("status") for e in events]
        assert "complete" in statuses
        complete_event = next(e for e in events if e.get("status") == "complete")
        assert complete_event["data"]["sql"] != ""
