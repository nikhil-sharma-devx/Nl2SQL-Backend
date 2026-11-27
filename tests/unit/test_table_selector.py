"""Unit tests for TableSelectorService (H2: preserve intentional empty results).

``select_tables`` used to do ``return selected or (fallback_tables or
available_tables)`` — Python treats the LLM's deliberate ``[]`` ("no table
can answer this question") as falsy and replaces it with the fallback,
defeating the anti-hallucination purpose: every off-topic question dumped
the entire schema into context instead of correctly retrieving nothing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nl_to_sql.rag.retrieval.table_selector import TableSelectorService


def _llm_returning(content: str) -> MagicMock:
    llm = MagicMock()
    response = MagicMock(content=content, total_tokens=10)
    llm.complete = AsyncMock(return_value=response)
    return llm


class TestEmptySelectionPreserved:
    async def test_deliberate_empty_array_is_honored(self) -> None:
        """The LLM validly says 'no table answers this' — must return []."""
        svc = TableSelectorService(llm_provider=_llm_returning("[]"))
        result = await svc.select_tables(
            "What's the weather today?",
            available_tables=["orders", "customers"],
            fallback_tables=["orders"],
        )
        assert result == []

    async def test_unparseable_response_falls_back(self) -> None:
        """A genuine parse failure (no JSON array at all) must still fall back."""
        svc = TableSelectorService(llm_provider=_llm_returning("I cannot help with that."))
        result = await svc.select_tables(
            "some question",
            available_tables=["orders", "customers"],
            fallback_tables=["orders"],
        )
        assert result == ["orders"]

    async def test_unparseable_falls_back_to_available_with_no_fallback(self) -> None:
        svc = TableSelectorService(llm_provider=_llm_returning("not json"))
        result = await svc.select_tables(
            "some question",
            available_tables=["orders", "customers"],
        )
        assert result == ["orders", "customers"]

    async def test_valid_selection_returned_as_is(self) -> None:
        svc = TableSelectorService(llm_provider=_llm_returning('["orders"]'))
        result = await svc.select_tables(
            "show me orders",
            available_tables=["orders", "customers"],
            fallback_tables=["orders", "customers"],
        )
        assert result == ["orders"]

    async def test_llm_call_exception_falls_back(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        svc = TableSelectorService(llm_provider=llm)
        result = await svc.select_tables(
            "some question",
            available_tables=["orders"],
            fallback_tables=["orders"],
        )
        assert result == ["orders"]
