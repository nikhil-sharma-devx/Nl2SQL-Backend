"""Unit/integration tests for multi-turn conversational querying.

Covers conversation pruning/compression flowing into generation, the 5-turn
refinement chain, ambiguous follow-up clarification, and single-turn
backwards-compatibility.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from nl_to_sql.core.models.query import QueryRequest, QueryResponse
from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.schema_retriever import SchemaRetriever
from nl_to_sql.services.sql_generator import SQLGeneratorService
from nl_to_sql.services.sql_validator import SQLValidatorService


def _turn(question: str, sql: str, minute: int) -> MagicMock:
    msg = MagicMock()
    msg.question = question
    msg.sql = sql
    msg.timestamp = datetime(2024, 1, 1, 12, 0, 0) + timedelta(minutes=minute)
    return msg


def _session_with(turns: list[MagicMock]) -> MagicMock:
    sess = MagicMock()
    sess.messages = turns
    svc = MagicMock()
    svc.get_session = AsyncMock(return_value=sess)
    svc.add_message = AsyncMock()
    return svc


def _build_orchestrator(
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
    mock_llm_provider: MagicMock,
    in_memory_cache: InMemoryCache,
    session_service: MagicMock | None = None,
    conversation_max_turns: int = 6,
) -> tuple[QueryOrchestrator, AsyncMock]:
    generator = SQLGeneratorService(llm_provider=mock_llm_provider, dialect="postgresql")
    spy = AsyncMock(wraps=generator.generate)
    generator.generate = spy  # type: ignore[method-assign]
    orch = QueryOrchestrator(
        retriever=SchemaRetriever(embedder=mock_embedder, vector_store=mock_vector_store, top_k=2),
        generator=generator,
        validator=SQLValidatorService(dialect="postgresql"),
        cache=in_memory_cache,
        max_retries=2,
        session_service=session_service,
        conversation_max_turns=conversation_max_turns,
    )
    return orch, spy


class TestConversationPruning:
    async def test_five_turn_refinement_chain_passes_bounded_history(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_llm_provider: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        # A long refinement session: sales by month → 2024 → Europe → sort → top5
        # ... plus extra older turns so pruning has something to compress.
        prior = [
            _turn("show revenue overall", "SELECT SUM(total) FROM orders;", 0),
            _turn("break it down", "SELECT c FROM t;", 1),
            _turn("show sales by month", "SELECT month, SUM(total) FROM orders GROUP BY month;", 2),
            _turn("only for 2024", "SELECT month FROM orders WHERE year=2024;", 3),
            _turn("now only Europe", "SELECT month FROM orders WHERE region='EU';", 4),
            _turn("sort descending", "SELECT month FROM orders ORDER BY 2 DESC;", 5),
            _turn("top 5", "SELECT month FROM orders ORDER BY 2 DESC LIMIT 5;", 6),
            _turn("show as percentages", "SELECT month, 100.0 FROM orders;", 7),
        ]
        svc = _session_with(prior)
        orch, spy = _build_orchestrator(
            mock_embedder, mock_vector_store, mock_llm_provider, in_memory_cache,
            session_service=svc, conversation_max_turns=6,
        )

        req = QueryRequest(question="only the profitable ones", session_id="s1")
        await orch.run(req)

        assert spy.await_count >= 1
        history = spy.call_args.kwargs["conversation_history"]
        # History is capped (6 recent + 1 summary), not the full 8 prior turns.
        assert len(history) <= 6 + 1
        assert len(history) < len(prior) + 1
        # Recency preserved: the most recent prior turn is present verbatim.
        assert history[-1]["sql"] == prior[-1].sql
        assert history[-1]["question"] == prior[-1].question
        # The oldest turns were compressed into a leading summary.
        assert history[0].get("compressed") is True

    async def test_single_turn_unchanged_no_history(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_llm_provider: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        orch, spy = _build_orchestrator(
            mock_embedder, mock_vector_store, mock_llm_provider, in_memory_cache,
        )
        response = await orch.run(QueryRequest(question="Show me all customers"))
        assert isinstance(response, QueryResponse)
        assert response.needs_clarification is False
        assert response.sql != ""
        # No session → no conversation history passed.
        assert spy.call_args.kwargs["conversation_history"] is None


class TestAmbiguousClarification:
    async def test_ambiguous_followup_requests_clarification(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_llm_provider: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        prior = [_turn("show sales by month", "SELECT month FROM orders;", 0)]
        svc = _session_with(prior)
        orch, spy = _build_orchestrator(
            mock_embedder, mock_vector_store, mock_llm_provider, in_memory_cache,
            session_service=svc,
        )

        req = QueryRequest(question="now do that", session_id="s1")
        response = await orch.run(req)

        assert response.needs_clarification is True
        assert response.clarification_prompt
        assert response.sql == ""
        # We asked instead of guessing — no SQL generation happened.
        assert spy.await_count == 0
        # Clarification is persisted to the session for chat continuity.
        svc.add_message.assert_awaited()

    async def test_resolvable_followup_does_not_clarify(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_llm_provider: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        prior = [_turn("show sales by month", "SELECT month FROM orders;", 0)]
        svc = _session_with(prior)
        orch, spy = _build_orchestrator(
            mock_embedder, mock_vector_store, mock_llm_provider, in_memory_cache,
            session_service=svc,
        )

        req = QueryRequest(question="now sort descending", session_id="s1")
        response = await orch.run(req)

        assert response.needs_clarification is False
        assert response.sql != ""
        assert spy.await_count >= 1

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("do that", True),
            ("change it", True),
            ("the other one", True),
            ("now only 2024", False),
            ("now europe", False),
            ("sort descending", False),
            ("top 5", False),
            ("group by region", False),
        ],
    )
    def test_ambiguous_followup_heuristic(self, question: str, expected: bool) -> None:
        assert QueryOrchestrator._is_ambiguous_followup(question) is expected
