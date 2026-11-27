"""Token-budget helpers for reasoning ("thinking") models.

Reasoning models — Groq-hosted ``openai/gpt-oss-*`` and ``qwen/qwen3*``, and
current Gemini models — spend hidden reasoning tokens *before* the visible
answer, and those tokens count against the same ``max_tokens`` cap. Callers in
this codebase size ``max_tokens`` for the answer alone (200 tokens for a query
rewrite, 256 for the table selector, …), so on a reasoning model the cap was
used up mid-thought: the reply came back empty or as truncated JSON (Groq
``json_validate_failed``: "max completion tokens reached before generating a
valid document").

Providers therefore treat the caller's ``max_tokens`` as the *answer* budget
and add a fixed reasoning allowance on top for reasoning models. Billing is per
token actually generated, so the extra headroom costs nothing unless used.
"""

from __future__ import annotations

# Observed gpt-oss-120b reasoning on SQL-generation prompts: ~150-500 tokens at
# default effort. 2048 leaves ample margin for harder questions.
REASONING_HEADROOM_TOKENS = 2048

_GROQ_REASONING_PREFIXES = ("openai/gpt-oss", "qwen/qwen3")


def is_groq_reasoning_model(model: str) -> bool:
    """Return True for Groq-hosted models that emit hidden reasoning tokens."""
    return model.lower().startswith(_GROQ_REASONING_PREFIXES)


def with_reasoning_headroom(max_tokens: int, *, reasoning: bool) -> int:
    """Return the completion cap to send: answer budget plus reasoning allowance."""
    return max_tokens + REASONING_HEADROOM_TOKENS if reasoning else max_tokens
