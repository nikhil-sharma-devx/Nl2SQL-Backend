"""Token budget management for NL-to-SQL prompt assembly.

Priority order (highest → lowest, lower items truncated first):
  1. system preamble  (fixed)
  2. user NL query    (never truncated)
  3. conversation history  (recent turns kept verbatim; older compressed/dropped)
  4. custom instructions  (hard cap 2000 chars ≈ ~500 tokens)
  5. retrieved schema chunks  (drop lowest-similarity chunks first)
  6. glossary  (Phase 2 placeholder)

Log every truncation so missing-context bugs are reproducible.
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

CHARS_PER_TOKEN = 4
INSTRUCTIONS_CHAR_CAP = 2000
# Small fixed per-turn overhead (role labels, separators) added when estimating
# the token cost of a single conversation turn.
CONVERSATION_TURN_OVERHEAD_TOKENS = 8


class PromptBudget:
    """Assembles a prompt within a declared token budget."""

    def __init__(
        self,
        model_context_tokens: int = 8192,
        max_completion_tokens: int = 1024,
        safety_margin_tokens: int = 200,
        conversation_max_turns: int = 6,
    ) -> None:
        self.budget_tokens = model_context_tokens - max_completion_tokens - safety_margin_tokens
        # Most-recent N turns kept verbatim; older turns are compressed to a
        # compact one-line summary before the token budget is applied.
        self._conversation_max_turns = max(0, conversation_max_turns)

    @staticmethod
    def _turn_tokens(turn: dict[str, Any]) -> int:
        """Estimate the token cost of a single rendered conversation turn."""
        q = str(turn.get("question", "") or "")
        sql = str(turn.get("sql", "") or "")
        return (len(q) + len(sql)) // CHARS_PER_TOKEN + CONVERSATION_TURN_OVERHEAD_TOKENS

    @staticmethod
    def _summarize_older(older: list[dict[str, Any]]) -> dict[str, Any]:
        """Compress all older turns into ONE compact summary entry.

        Format per turn: ``Q: <question> → SQL: <first line of SQL>``. Folding
        every older turn into a single entry keeps the total history length
        bounded (recent turns + at most one summary) no matter how long the
        session runs.
        """
        lines: list[str] = []
        for turn in older:
            q = str(turn.get("question", "") or "").strip()
            sql = str(turn.get("sql", "") or "").strip()
            first_line = sql.splitlines()[0].strip() if sql else ""
            if q:
                lines.append(f"Q: {q}" + (f" → SQL: {first_line}" if first_line else ""))
        return {
            "question": "Earlier in this session:\n" + "\n".join(lines),
            "sql": "",
            "compressed": True,
        }

    def fit_conversation(
        self,
        history: list[dict[str, Any]],
        remaining_budget: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Bound conversation history to recent turns within a token budget.

        The most recent ``conversation_max_turns`` turns are kept verbatim;
        every older turn is folded into a single compact summary entry placed
        first. The result (at most ``conversation_max_turns + 1`` entries) is
        then fit into ``remaining_budget`` — when it overflows, the oldest
        entries (summary first) are dropped so recency always wins. The returned
        list is chronological so the generator can render it naturally.

        Args:
            history: Full conversation history (chronological, oldest first),
                     each entry a dict with ``question`` and ``sql`` keys.
            remaining_budget: Token budget available for conversation history.

        Returns:
            ``(kept, truncations)`` — the bounded, chronological history and a
            list of human-readable truncation notes (empty when nothing changed).
        """
        truncations: list[str] = []
        if not history:
            return [], truncations

        n = self._conversation_max_turns
        recent = history[-n:] if n > 0 else []
        older = history[: len(history) - len(recent)]

        entries: list[dict[str, Any]] = []
        if older:
            entries.append(self._summarize_older(older))
            truncations.append(
                f"conversation: compressed {len(older)} older turn(s) into 1 summary"
            )
        entries.extend(recent)

        # Budget fit: drop oldest first (summary, then oldest verbatim) so the
        # most recent turns are always preserved.
        kept = list(entries)
        dropped = 0
        while kept and sum(self._turn_tokens(t) for t in kept) > remaining_budget:
            kept.pop(0)
            dropped += 1
        if dropped:
            truncations.append(
                f"conversation: dropped {dropped} oldest turn(s) (budget exhausted)"
            )

        if truncations:
            logger.info(
                "prompt_budget: conversation history bounded",
                original_turns=len(history),
                kept_turns=len(kept),
                truncations=truncations,
            )

        return kept, truncations

    def assemble(
        self,
        system_preamble: str,
        user_query: str,
        custom_instructions: str | None,
        schema_chunks: list[str],
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Fit prompt components into the budget. Returns the components to use and a truncation log."""
        remaining = self.budget_tokens
        truncations: list[str] = []

        # 1. System preamble (always fits — if it doesn't, the model config is broken)
        remaining -= len(system_preamble) // CHARS_PER_TOKEN

        # 2. User query — never truncated
        remaining -= len(user_query) // CHARS_PER_TOKEN

        # 3. Conversation history — recent-first, older compressed/dropped
        used_conversation: list[dict[str, Any]] = []
        if conversation_history:
            used_conversation, conv_truncations = self.fit_conversation(
                conversation_history, remaining
            )
            truncations.extend(conv_truncations)
            remaining -= sum(self._turn_tokens(t) for t in used_conversation)

        # 4. Custom instructions — hard cap then budget check
        used_instructions: str | None = None
        if custom_instructions:
            if len(custom_instructions) > INSTRUCTIONS_CHAR_CAP:
                custom_instructions = custom_instructions[:INSTRUCTIONS_CHAR_CAP]
                truncations.append(
                    f"custom_instructions truncated to {INSTRUCTIONS_CHAR_CAP} chars"
                )
            instr_tokens = len(custom_instructions) // CHARS_PER_TOKEN
            if instr_tokens <= remaining:
                used_instructions = custom_instructions
                remaining -= instr_tokens
            else:
                truncations.append("custom_instructions dropped (no budget)")

        # 5. Schema chunks — fill remaining budget
        used_chunks: list[str] = []
        for chunk in schema_chunks:
            chunk_tokens = len(chunk) // CHARS_PER_TOKEN
            if chunk_tokens <= remaining:
                used_chunks.append(chunk)
                remaining -= chunk_tokens
            else:
                truncations.append(
                    f"schema chunk dropped (budget exhausted; used {len(used_chunks)} chunks)"
                )
                break

        if truncations:
            logger.warning(
                "prompt_budget: truncations occurred",
                truncations=truncations,
                remaining_tokens=remaining,
            )

        return {
            "system_preamble": system_preamble,
            "custom_instructions": used_instructions,
            "conversation_history": used_conversation,
            "schema_chunks": used_chunks,
            "user_query": user_query,
            "truncations": truncations,
        }
