"""Token budget management for NL-to-SQL prompt assembly.

Priority order (highest → lowest, lower items truncated first):
  1. system preamble  (fixed)
  2. user NL query    (never truncated)
  3. custom instructions  (hard cap 2000 chars ≈ ~500 tokens)
  4. retrieved schema chunks  (drop lowest-similarity chunks first)
  5. glossary  (Phase 2 placeholder)

Log every truncation so missing-context bugs are reproducible.
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

CHARS_PER_TOKEN = 4
INSTRUCTIONS_CHAR_CAP = 2000


class PromptBudget:
    """Assembles a prompt within a declared token budget."""

    def __init__(
        self,
        model_context_tokens: int = 8192,
        max_completion_tokens: int = 1024,
        safety_margin_tokens: int = 200,
    ) -> None:
        self.budget_tokens = model_context_tokens - max_completion_tokens - safety_margin_tokens

    def assemble(
        self,
        system_preamble: str,
        user_query: str,
        custom_instructions: str | None,
        schema_chunks: list[str],
    ) -> dict[str, Any]:
        """Fit prompt components into the budget. Returns the components to use and a truncation log."""
        remaining = self.budget_tokens
        truncations: list[str] = []

        # 1. System preamble (always fits — if it doesn't, the model config is broken)
        remaining -= len(system_preamble) // CHARS_PER_TOKEN

        # 2. User query — never truncated
        remaining -= len(user_query) // CHARS_PER_TOKEN

        # 3. Custom instructions — hard cap then budget check
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

        # 4. Schema chunks — fill remaining budget
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
            "schema_chunks": used_chunks,
            "user_query": user_query,
            "truncations": truncations,
        }
