"""Integration tests for Natural Language Data Corrections.

Two layers:
  1. Orchestrator flow — a correction message rewrites the previous turn (via
     QueryRewriter) and regenerates SQL using the prior conversation history,
     with real detector/retriever/generator/validator wired together.
  2. HTTP surface — the /api/v1/query endpoint accepts the optional
     ``is_correction`` field and returns the canonical error envelope.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.core.models.query import QueryRequest, QueryResponse
from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache
from nl_to_sql.services.correction_detector import CorrectionDetector
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.schema_retriever import SchemaRetriever
from nl_to_sql.services.sql_generator import SQLGeneratorService
from nl_to_sql.services.sql_validator import SQLValidatorService

REWRITTEN = "Show total sales grouped by customer_name"


def _prior_turn(question: str, sql: str) -> MagicMock:
    msg = MagicMock()
    msg.question = question
    msg.sql = sql
    msg.timestamp = datetime(2024, 1, 1, 12, 0, 0)
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
    session_service: MagicMock,
    rewriter: MagicMock,
) -> QueryOrchestrator:
    return QueryOrchestrator(
        retriever=SchemaRetriever(embedder=mock_embedder, vector_store=mock_vector_store, top_k=2),
        generator=SQLGeneratorService(llm_provider=mock_llm_provider, dialect="postgresql"),
        validator=SQLValidatorService(dialect="postgresql"),
        cache=in_memory_cache,
        max_retries=2,
        session_service=session_service,
        correction_detector=CorrectionDetector(),
        query_rewriter=rewriter,
    )


def _rewriter() -> MagicMock:
    r = MagicMock()
    r.apply_correction = AsyncMock(return_value=REWRITTEN)
    return r


def _llm_prompts(mock_llm_provider: MagicMock) -> str:
    """All text passed to the LLM across every complete() call."""
    parts: list[str] = []
    for call in mock_llm_provider.complete.call_args_list:
        parts.extend(str(a) for a in call.args)
        parts.extend(str(v) for v in call.kwargs.values())
    return " ".join(parts)


class TestCorrectionOrchestratorFlow:
    async def test_correction_rewrites_previous_turn_and_regenerates(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_llm_provider: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        prior = _prior_turn(
            "Show total sales grouped by customer_id",
            "SELECT customer_id, SUM(amount) FROM sales GROUP BY customer_id",
        )
        rewriter = _rewriter()
        orch = _build_orchestrator(
            mock_embedder, mock_vector_store, mock_llm_provider,
            in_memory_cache, _session_with([prior]), rewriter,
        )

        response = await orch.run(
            QueryRequest(question="no, I meant customer_name", session_id="s1")
        )

        assert isinstance(response, QueryResponse)
        rewriter.apply_correction.assert_awaited_once()
        kwargs = rewriter.apply_correction.call_args.kwargs
        assert kwargs["previous_question"] == prior.question
        assert kwargs["previous_sql"] == prior.sql
        assert kwargs["correction"] == "no, I meant customer_name"
        # Regeneration used the corrected question, not the raw correction text.
        assert "customer_name" in _llm_prompts(mock_llm_provider)

    async def test_multiple_corrections_chain_to_latest_turn(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_llm_provider: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        first = _prior_turn(
            "Show total sales by region",
            "SELECT region, SUM(amount) FROM sales GROUP BY region",
        )
        latest = _prior_turn(
            "Show total sales by customer_id",
            "SELECT customer_id, SUM(amount) FROM sales GROUP BY customer_id",
        )
        rewriter = _rewriter()
        orch = _build_orchestrator(
            mock_embedder, mock_vector_store, mock_llm_provider,
            in_memory_cache, _session_with([first, latest]), rewriter,
        )

        await orch.run(QueryRequest(question="actually use customer_name", session_id="s1"))

        rewriter.apply_correction.assert_awaited_once()
        # The correction targets the most recent prior turn.
        assert rewriter.apply_correction.call_args.kwargs["previous_question"] == latest.question

    async def test_non_correction_does_not_rewrite(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_llm_provider: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        prior = _prior_turn("Show total sales", "SELECT SUM(amount) FROM sales")
        rewriter = _rewriter()
        orch = _build_orchestrator(
            mock_embedder, mock_vector_store, mock_llm_provider,
            in_memory_cache, _session_with([prior]), rewriter,
        )

        response = await orch.run(
            QueryRequest(question="How many customers are in Canada?", session_id="s1")
        )

        assert isinstance(response, QueryResponse)
        rewriter.apply_correction.assert_not_awaited()

    async def test_correction_without_history_falls_back_to_normal(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_llm_provider: MagicMock,
        in_memory_cache: InMemoryCache,
    ) -> None:
        rewriter = _rewriter()
        orch = _build_orchestrator(
            mock_embedder, mock_vector_store, mock_llm_provider,
            in_memory_cache, _session_with([]), rewriter,
        )

        response = await orch.run(
            QueryRequest(question="use customer_name instead", session_id="s1")
        )

        assert isinstance(response, QueryResponse)
        rewriter.apply_correction.assert_not_awaited()


# ── HTTP surface ────────────────────────────────────────────────────────────

USER_ID = "test-user-corr-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="corr@example.com",
    full_name="Correction Test User",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _null_result() -> MagicMock:
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=None)
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    return r


def _make_app_with_orchestrator(orch: MagicMock | None):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_active_connection_id,
            get_current_user,
            get_metrics_service,
            get_request_orchestrator,
            get_session_service,
        )

        app = create_app()

        async def _fake_user() -> UserPublic:
            return FAKE_USER

        async def _fake_connection_id() -> str:
            return "conn-1"

        mock_db = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        mock_db.execute = AsyncMock(return_value=_null_result())
        mock_svc = MagicMock()
        mock_svc._session_factory = MagicMock(return_value=mock_db)

        mock_metrics_svc = MagicMock()
        mock_metrics_svc.list_certified_for_prompt = AsyncMock(return_value=[])

        app.dependency_overrides[get_current_user] = _fake_user
        app.dependency_overrides[get_session_service] = lambda: mock_svc
        app.dependency_overrides[get_active_connection_id] = _fake_connection_id
        app.dependency_overrides[get_metrics_service] = lambda: mock_metrics_svc
        if orch is not None:
            app.dependency_overrides[get_request_orchestrator] = lambda: orch
        return app


class TestCorrectionEndpoint:
    async def test_endpoint_accepts_is_correction_field(self) -> None:
        orch = MagicMock()
        orch.run = AsyncMock(
            return_value=QueryResponse(
                question="no, I meant customer_name",
                sql="SELECT customer_name FROM sales",
                dialect="postgresql",
                is_valid=True,
            )
        )
        app = _make_app_with_orchestrator(orch)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/v1/query",
                json={"question": "no, I meant customer_name", "is_correction": True},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert r.status_code == 200
        orch.run.assert_awaited_once()
        passed_request = orch.run.call_args.args[0]
        assert passed_request.is_correction is True

    async def test_short_question_returns_error_envelope(self) -> None:
        app = _make_app_with_orchestrator(orch=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/v1/query",
                json={"question": "no", "is_correction": True},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert r.status_code == 422
        assert "message" in r.json()
