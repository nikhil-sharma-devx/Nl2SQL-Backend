"""Unit tests for QueryRewriter.apply_correction (LLM path + fallback)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nl_to_sql.core.exceptions import LLMProviderError
from nl_to_sql.core.models.sql_result import LLMResponse
from nl_to_sql.services.query_rewriter import QueryRewriter


def _rewriter(llm: MagicMock) -> QueryRewriter:
    return QueryRewriter(llm_provider=llm, enabled=True)


class TestApplyCorrectionLLMPath:
    async def test_returns_llm_rewrite(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content="Show total sales grouped by customer_name",
                prompt_tokens=10,
                completion_tokens=8,
            )
        )
        rewriter = _rewriter(llm)

        result = await rewriter.apply_correction(
            previous_question="Show total sales grouped by customer_id",
            previous_sql="SELECT customer_id, SUM(amount) FROM sales GROUP BY customer_id",
            correction="no, I meant customer_name",
        )

        assert result == "Show total sales grouped by customer_name"
        llm.complete.assert_awaited_once()

    async def test_correction_works_even_when_rewriting_disabled(self) -> None:
        # apply_correction is independent of the ambiguity-rewrite toggle.
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content="corrected question", prompt_tokens=1, completion_tokens=1
            )
        )
        rewriter = QueryRewriter(llm_provider=llm, enabled=False)

        result = await rewriter.apply_correction("prev", "SELECT 1", "use X instead")
        assert result == "corrected question"
        llm.complete.assert_awaited_once()


class TestApplyCorrectionFallback:
    async def test_falls_back_on_llm_error(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=LLMProviderError("boom"))
        rewriter = _rewriter(llm)

        result = await rewriter.apply_correction(
            previous_question="Show total sales",
            previous_sql="SELECT SUM(amount) FROM sales",
            correction="use OrderDate instead",
        )

        # Deterministic concatenation fallback — never raises.
        assert result == "Show total sales (correction: use OrderDate instead)"

    async def test_falls_back_on_unexpected_error(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("unexpected"))
        rewriter = _rewriter(llm)

        result = await rewriter.apply_correction("prev q", "SELECT 1", "actually use profit")
        assert result == "prev q (correction: actually use profit)"

    async def test_empty_result_uses_fallback(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(content="   ", prompt_tokens=1, completion_tokens=0)
        )
        rewriter = _rewriter(llm)

        result = await rewriter.apply_correction("prev q", "SELECT 1", "use total")
        assert result == "prev q (correction: use total)"

    async def test_empty_correction_returns_previous(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock()
        rewriter = _rewriter(llm)

        result = await rewriter.apply_correction("prev q", "SELECT 1", "   ")
        assert result == "prev q"
        llm.complete.assert_not_awaited()
