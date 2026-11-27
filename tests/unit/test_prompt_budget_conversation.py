"""Unit tests for PromptBudget conversation fitting (multi-turn memory)."""
from __future__ import annotations

from typing import Any

from nl_to_sql.services.prompt_budget import PromptBudget


def _history(n: int) -> list[dict[str, Any]]:
    """Build n chronological turns (oldest first)."""
    return [
        {"question": f"question number {i}", "sql": f"SELECT {i} FROM t;\n-- line 2"}
        for i in range(n)
    ]


class TestFitConversation:
    def test_short_history_kept_verbatim(self) -> None:
        budget = PromptBudget(conversation_max_turns=6)
        hist = _history(3)
        kept, truncations = budget.fit_conversation(hist, budget.budget_tokens)
        assert kept == hist  # unchanged
        assert truncations == []

    def test_recent_turns_kept_older_compressed(self) -> None:
        budget = PromptBudget(conversation_max_turns=6)
        hist = _history(10)
        kept, truncations = budget.fit_conversation(hist, budget.budget_tokens)

        # Length is capped at N recent + 1 summary regardless of session length.
        assert len(kept) == 7
        # First entry is the single compressed summary of the 4 older turns.
        assert kept[0].get("compressed") is True
        assert "Earlier in this session" in kept[0]["question"]
        assert "compressed 4 older turn(s)" in " ".join(truncations)
        # Recency preserved: the most recent turns are kept verbatim, in order.
        assert kept[1:] == hist[-6:]
        assert kept[-1] == hist[-1]

    def test_length_capped_for_very_long_session(self) -> None:
        budget = PromptBudget(conversation_max_turns=6)
        kept, _ = budget.fit_conversation(_history(100), budget.budget_tokens)
        # Bounded: never grows with session length.
        assert len(kept) <= 6 + 1

    def test_budget_respected_drops_oldest(self) -> None:
        budget = PromptBudget(conversation_max_turns=6)
        hist = _history(10)
        tiny_budget = 40  # only room for a couple of turns
        kept, truncations = budget.fit_conversation(hist, tiny_budget)

        total = sum(budget._turn_tokens(t) for t in kept)
        assert total <= tiny_budget
        assert "budget exhausted" in " ".join(truncations)
        # Whatever survives is the most recent turn(s).
        assert kept[-1] == hist[-1]

    def test_empty_history(self) -> None:
        budget = PromptBudget(conversation_max_turns=6)
        kept, truncations = budget.fit_conversation([], budget.budget_tokens)
        assert kept == []
        assert truncations == []

    def test_summary_uses_first_sql_line_only(self) -> None:
        budget = PromptBudget(conversation_max_turns=2)
        hist = _history(4)
        kept, _ = budget.fit_conversation(hist, budget.budget_tokens)
        summary = kept[0]["question"]
        # First SQL line included, second dropped.
        assert "SELECT 0 FROM t;" in summary
        assert "line 2" not in summary


class TestAssembleConversationSlot:
    def test_assemble_includes_conversation_history(self) -> None:
        budget = PromptBudget(conversation_max_turns=6)
        result = budget.assemble(
            system_preamble="sys",
            user_query="latest question",
            custom_instructions=None,
            schema_chunks=["chunk a", "chunk b"],
            conversation_history=_history(10),
        )
        assert "conversation_history" in result
        assert len(result["conversation_history"]) == 7
        assert any("compressed" in t for t in result["truncations"])

    def test_assemble_backwards_compatible_without_history(self) -> None:
        budget = PromptBudget()
        result = budget.assemble(
            system_preamble="sys",
            user_query="q",
            custom_instructions="be concise",
            schema_chunks=["chunk"],
        )
        # No conversation supplied → empty slot, existing behavior unchanged.
        assert result["conversation_history"] == []
        assert result["custom_instructions"] == "be concise"
        assert result["schema_chunks"] == ["chunk"]

    def test_conversation_competes_for_budget_before_schema(self) -> None:
        # Small budget: conversation history is fit before schema chunks.
        budget = PromptBudget(
            model_context_tokens=400, max_completion_tokens=0, safety_margin_tokens=0
        )
        result = budget.assemble(
            system_preamble="s" * 40,
            user_query="u" * 40,
            custom_instructions=None,
            schema_chunks=["x" * 4000],  # huge chunk that won't fit
            conversation_history=_history(8),
        )
        # Conversation kept something; oversized schema chunk dropped.
        assert result["conversation_history"]
        assert result["schema_chunks"] == []
