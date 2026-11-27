"""Unit tests for QueryOrchestrator._fetch_few_shot_examples (H4).

Regression test: ``example_store`` (P2 — indexed on positive feedback for
semantic few-shot retrieval) was written to but never read anywhere; every
few-shot lookup silently used only the recency-based
``training_data_service`` fallback, making the documented semantic-retrieval
feature entirely non-functional. Builds a bare instance via
``object.__new__`` and sets only the attributes ``_fetch_few_shot_examples``
touches — constructing a full ``QueryOrchestrator`` needs ~15 DI dependencies
unrelated to this method.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nl_to_sql.services.query_orchestrator import QueryOrchestrator


def _orchestrator(
    *,
    few_shot_enabled: bool = True,
    example_store: MagicMock | None = None,
    training_data_service: MagicMock | None = None,
    few_shot_top_k: int = 3,
    connection_id: str | None = "conn-1",
) -> QueryOrchestrator:
    orch = object.__new__(QueryOrchestrator)
    orch._few_shot_enabled = few_shot_enabled
    orch._example_store = example_store
    orch._training_data_service = training_data_service
    orch._few_shot_top_k = few_shot_top_k
    orch._connection_id = connection_id
    return orch


class TestFewShotExampleWiring:
    async def test_semantic_search_used_when_available(self) -> None:
        example_store = MagicMock()
        example_store.search = AsyncMock(
            return_value=[{"question": "similar Q", "sql": "SELECT 1"}]
        )
        training_data_service = MagicMock()
        training_data_service.get_recent_examples = AsyncMock(
            return_value=[{"question": "recent Q", "sql": "SELECT 2"}]
        )
        orch = _orchestrator(
            example_store=example_store, training_data_service=training_data_service
        )

        result = await orch._fetch_few_shot_examples("some question", MagicMock())

        assert result == [{"question": "similar Q", "sql": "SELECT 1"}]
        example_store.search.assert_awaited_once_with(
            "some question", top_k=3, connection_id="conn-1"
        )
        training_data_service.get_recent_examples.assert_not_awaited()

    async def test_falls_back_to_recency_when_semantic_search_empty(self) -> None:
        example_store = MagicMock()
        example_store.search = AsyncMock(return_value=[])
        training_data_service = MagicMock()
        training_data_service.get_recent_examples = AsyncMock(
            return_value=[{"question": "recent Q", "sql": "SELECT 2"}]
        )
        orch = _orchestrator(
            example_store=example_store, training_data_service=training_data_service
        )

        result = await orch._fetch_few_shot_examples("some question", MagicMock())

        assert result == [{"question": "recent Q", "sql": "SELECT 2"}]
        training_data_service.get_recent_examples.assert_awaited_once_with(
            limit=3, connection_id="conn-1"
        )

    async def test_falls_back_to_recency_when_semantic_search_errors(self) -> None:
        example_store = MagicMock()
        example_store.search = AsyncMock(side_effect=RuntimeError("qdrant down"))
        training_data_service = MagicMock()
        training_data_service.get_recent_examples = AsyncMock(
            return_value=[{"question": "recent Q", "sql": "SELECT 2"}]
        )
        orch = _orchestrator(
            example_store=example_store, training_data_service=training_data_service
        )

        result = await orch._fetch_few_shot_examples("some question", MagicMock())

        assert result == [{"question": "recent Q", "sql": "SELECT 2"}]

    async def test_disabled_returns_none_without_calling_anything(self) -> None:
        example_store = MagicMock()
        example_store.search = AsyncMock()
        orch = _orchestrator(few_shot_enabled=False, example_store=example_store)

        result = await orch._fetch_few_shot_examples("some question", MagicMock())

        assert result is None
        example_store.search.assert_not_awaited()

    async def test_no_example_store_uses_recency_directly(self) -> None:
        training_data_service = MagicMock()
        training_data_service.get_recent_examples = AsyncMock(
            return_value=[{"question": "recent Q", "sql": "SELECT 2"}]
        )
        orch = _orchestrator(example_store=None, training_data_service=training_data_service)

        result = await orch._fetch_few_shot_examples("some question", MagicMock())

        assert result == [{"question": "recent Q", "sql": "SELECT 2"}]
