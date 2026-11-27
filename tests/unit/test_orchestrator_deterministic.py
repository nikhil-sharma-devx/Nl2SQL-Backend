"""Deterministic orchestrator tests using a scripted ILLMProvider.

Unlike ``test_query_orchestrator.py`` (which uses a single canned MagicMock),
these drive the self-correction/retry loop with a fake provider that returns a
*scripted sequence* of completions, so we can assert the orchestrator:

  1. retries generation when the first SQL fails validation, and
  2. surfaces the repaired, valid SQL on the second attempt.

Also covers cache miss -> hit determinism end to end. (IMPROVEMENTS.md item 19.)
"""
from unittest.mock import MagicMock

from nl_to_sql.core.models.query import QueryRequest, QueryResponse
from nl_to_sql.core.models.sql_result import LLMResponse
from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.schema_retriever import SchemaRetriever
from nl_to_sql.services.sql_generator import SQLGeneratorService
from nl_to_sql.services.sql_validator import SQLValidatorService


class ScriptedLLMProvider:
    """ILLMProvider stand-in that returns a fixed sequence of completions.

    Each call to ``complete`` pops the next scripted SQL string; the last entry
    repeats once the script is exhausted (so an over-eager retry loop doesn't
    IndexError). ``call_count`` lets tests assert how many attempts ran.
    """

    def __init__(self, scripted_sql: list[str]) -> None:
        assert scripted_sql, "script must contain at least one response"
        self._script = scripted_sql
        self.call_count = 0

    async def complete(self, *args: object, **kwargs: object) -> LLMResponse:
        idx = min(self.call_count, len(self._script) - 1)
        self.call_count += 1
        return LLMResponse(
            content=self._script[idx],
            prompt_tokens=50,
            completion_tokens=10,
        )

    async def health_check(self) -> bool:
        return True


def _make_orchestrator(
    provider: ScriptedLLMProvider,
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
    cache: InMemoryCache,
    max_retries: int = 3,
    connection_id: str | None = "conn-test",
    training_data_service: object | None = None,
) -> QueryOrchestrator:
    retriever = SchemaRetriever(embedder=mock_embedder, vector_store=mock_vector_store, top_k=2)
    generator = SQLGeneratorService(llm_provider=provider, dialect="postgresql")
    validator = SQLValidatorService(dialect="postgresql")
    return QueryOrchestrator(
        retriever=retriever,
        generator=generator,
        validator=validator,
        cache=cache,
        max_retries=max_retries,
        # Real requests always resolve a connection; None is the fail-closed path.
        connection_id=connection_id,
        training_data_service=training_data_service,  # type: ignore[arg-type]
    )


class TestDeterministicOrchestrator:

    async def test_retries_then_succeeds(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        # Attempt 1: a non-SELECT statement — parses but the validator rejects it.
        # Attempt 2: a valid SELECT — should pass and break the loop.
        provider = ScriptedLLMProvider(
            scripted_sql=[
                "UPDATE customers SET country = 'US';",
                "SELECT customer_id, email FROM customers LIMIT 10;",
            ]
        )
        orch = _make_orchestrator(
            provider, mock_embedder, mock_vector_store, in_memory_cache
        )

        response = await orch.run(QueryRequest(question="list customer emails"))

        assert isinstance(response, QueryResponse)
        assert response.is_valid is True
        assert response.sql.strip().upper().startswith("SELECT")
        # The first (invalid) attempt forced exactly one retry.
        assert provider.call_count == 2

    async def test_invalid_never_repairs_stays_invalid(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        # Every attempt returns a non-SELECT — orchestrator exhausts max_retries
        # and returns an invalid response rather than raising.
        provider = ScriptedLLMProvider(
            scripted_sql=["DELETE FROM customers WHERE 1 = 1;"]
        )
        orch = _make_orchestrator(
            provider, mock_embedder, mock_vector_store, in_memory_cache, max_retries=2
        )

        response = await orch.run(QueryRequest(question="delete everything"))

        assert response.is_valid is False
        assert provider.call_count == 2  # tried max_retries times

    async def test_cache_miss_then_hit(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        provider = ScriptedLLMProvider(
            scripted_sql=["SELECT customer_id FROM customers LIMIT 5;"]
        )
        orch = _make_orchestrator(
            provider, mock_embedder, mock_vector_store, in_memory_cache
        )
        request = QueryRequest(question="how many customer ids are there")

        first = await orch.run(request)
        assert first.cached is False
        assert first.is_valid is True
        calls_after_first = provider.call_count

        second = await orch.run(request)
        assert second.cached is True
        # A cache hit must NOT re-invoke the LLM.
        assert provider.call_count == calls_after_first
        assert second.sql == first.sql

    async def test_unscoped_request_never_uses_the_cache(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        """No resolved connection → an unscoped cache would be shared by everyone
        in that state, so it is neither read nor written."""
        provider = ScriptedLLMProvider(
            scripted_sql=["SELECT customer_id FROM customers LIMIT 5;"] * 2
        )
        orch = _make_orchestrator(
            provider, mock_embedder, mock_vector_store, in_memory_cache, connection_id=None
        )
        request = QueryRequest(question="how many customer ids are there")
        first = await orch.run(request)
        second = await orch.run(request)
        assert first.is_valid and not first.cached
        assert second.cached is False
        assert provider.call_count >= 2  # the LLM ran both times

    async def test_unscoped_request_does_not_create_shared_training_rows(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        """A training row with no connection_id is offered to EVERY user as a
        few-shot example — an unresolved request must not write one."""
        from unittest.mock import AsyncMock

        training = MagicMock(collect_training_data=AsyncMock(), get_recent_examples=AsyncMock(return_value=[]))
        provider = ScriptedLLMProvider(scripted_sql=["SELECT customer_id FROM customers LIMIT 5;"] * 2)

        unscoped = _make_orchestrator(
            provider, mock_embedder, mock_vector_store, in_memory_cache,
            connection_id=None, training_data_service=training,
        )
        await unscoped.run(QueryRequest(question="how many customer ids are there"))
        training.collect_training_data.assert_not_awaited()

        scoped = _make_orchestrator(
            provider, mock_embedder, mock_vector_store, InMemoryCache(),
            connection_id="conn-a", training_data_service=training,
        )
        await scoped.run(QueryRequest(question="how many customer ids are there"))
        training.collect_training_data.assert_awaited_once()
        assert training.collect_training_data.await_args.kwargs["connection_id"] == "conn-a"
