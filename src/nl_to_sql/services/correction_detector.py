"""Correction detector — recognises when a message corrects the previous turn.

Pure, dependency-free heuristics (regex + keyword). Given a natural-language
message, decide whether it is a *correction* of the previous NL→SQL turn — e.g.
"no, I meant customer_name", "use OrderDate instead", "actually use sales_amount",
"not revenue, use profit", "change country to region" — and, when possible,
extract the corrected target term.

The orchestrator uses ``is_correction`` to route the message through the
correction-rewrite path; the full correction text is what actually drives the
LLM rewrite, so ``target_term`` is best-effort and used for logging/telemetry.

SOLID:
  S — Only classifies correction phrasing; no I/O, no LLM, no state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CorrectionSignal:
    """Result of correction detection for a single message.

    Attributes:
        is_correction: True when the message reads as a correction of the prior turn.
        target_term: Best-effort extracted replacement term (e.g. "customer_name"),
            or None when nothing specific could be pulled out.
    """

    is_correction: bool
    target_term: str | None = None


class CorrectionDetector:
    """Heuristic classifier for "please correct the previous turn" messages."""

    # Ordered (pattern, replacement-capture-group) pairs. Order matters: more
    # specific two-part forms ("not X, use Y", "instead of X use Y") are tried
    # before the shorter one-part forms so the *replacement* term wins.
    _PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
        # "no, i meant X" / "i meant X" / "i mean X" / "meant to say X"
        (re.compile(r"\bi\s+mean(?:t)?\s+(?:to\s+say\s+)?(.+)$", re.IGNORECASE), 1),
        # "instead of X, use Y" → Y
        (re.compile(r"\binstead\s+of\s+.+?[,;]?\s+(?:use|it'?s|its)\s+(.+)$", re.IGNORECASE), 1),
        # "not X, use Y" / "not X use Y" → Y
        (re.compile(r"\bnot\s+.+?[,;]?\s+(?:use|but)\s+(.+)$", re.IGNORECASE), 1),
        # "change X to Y" / "swap X for Y" → Y
        (re.compile(r"\b(?:change|swap|replace)\s+.+?\s+(?:to|for|with)\s+(.+)$", re.IGNORECASE), 1),
        # "use X instead" → X
        (re.compile(r"\buse\s+(.+?)\s+instead\b", re.IGNORECASE), 1),
        # "actually use X" / "actually, use X" / "actually it's X"
        (re.compile(r"\bactually[,]?\s+(?:use\s+|it'?s\s+|its\s+)?(.+)$", re.IGNORECASE), 1),
        # "no, use X" / "no use X"
        (re.compile(r"\bno[,.!]?\s+use\s+(.+)$", re.IGNORECASE), 1),
        # "(or) rather use X" / "rather X"
        (re.compile(r"\brather\s+(?:use\s+)?(.+)$", re.IGNORECASE), 1),
        # "(it) should be X" / "it's actually X" handled above; "should be X"
        (re.compile(r"\bshould\s+be\s+(.+)$", re.IGNORECASE), 1),
        # Standalone imperative "use X" (whole message)
        (re.compile(r"^\s*use\s+(.+)$", re.IGNORECASE), 1),
    )

    def detect(self, message: str) -> CorrectionSignal:
        """Classify ``message`` as a correction (or not) of the previous turn."""
        if not message or not message.strip():
            return CorrectionSignal(is_correction=False)

        text = message.strip()
        for pattern, group in self._PATTERNS:
            match = pattern.search(text)
            if match:
                target = self._clean_term(match.group(group))
                logger.debug(
                    "Correction phrasing detected",
                    message=text[:60],
                    target_term=target,
                )
                return CorrectionSignal(is_correction=True, target_term=target)

        return CorrectionSignal(is_correction=False)

    # Characters trimmed from both ends of an extracted term.
    _TRIM_CHARS = " \t\"'`.?!,;:"

    @classmethod
    def _clean_term(cls, raw: str) -> str | None:
        """Trim punctuation/quotes/filler from an extracted target term."""
        term = raw.strip()
        term = re.sub(r"^(?:the|a|an)\s+", "", term, flags=re.IGNORECASE)
        term = term.strip(cls._TRIM_CHARS)
        return term or None
