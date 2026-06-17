"""TableSelectorService — Phase B of the two-phase schema grounding pipeline.

Given a natural language question and the complete list of tables that exist in
the database, asks the LLM to identify which tables are needed to answer the
question.  The LLM is forced to choose only from the provided list, eliminating
one of the primary sources of hallucination (referencing non-existent tables).
"""
from __future__ import annotations

import json
import re
import structlog

from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider

logger = structlog.get_logger(__name__)

# System prompt for the table selector.
# Kept intentionally short and focused — the LLM only needs to output a JSON
# array, not a full SQL query.
_TABLE_SELECTOR_SYSTEM = """\
You are a database assistant helping to identify which database tables are \
needed to answer a natural language question.

RULES:
1. Return ONLY a JSON array of table name strings, e.g. ["orders", "customers"].
2. Choose ONLY from the list of available tables provided below.
3. Do NOT invent or guess table names that are not in the list.
4. Include every table whose columns are likely needed (for JOINs, filters, or \
output).
5. If no tables can answer the question return an empty array: [].
6. Return NOTHING except the JSON array — no explanation, no markdown fences.

AVAILABLE TABLES:
{table_list}
"""

_TABLE_SELECTOR_USER = "Question: {question}"

# Regex to pull a JSON array out of the response in case the LLM adds
# accidental whitespace or newlines around it.
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


class TableSelectorService:
    """Lightweight LLM call that identifies which tables are needed.

    This is Phase B of the two-phase grounding pipeline:
      A — Vector search → candidate tables (coarse, similarity-based)
      B — TableSelector → exact tables (constrained to known list) ← this
      C — Exact schema fetch → precise column context

    SOLID:
      S — Only responsible for table identification via LLM.
      D — Depends on ILLMProvider abstraction.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self._llm = llm_provider
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def select_tables(
        self,
        question: str,
        available_tables: list[str],
        fallback_tables: list[str] | None = None,
    ) -> list[str]:
        """Ask the LLM which tables are needed to answer *question*.

        Args:
            question: The natural language question from the user.
            available_tables: Complete list of ingested table names.  The LLM
                must only pick from this list.
            fallback_tables: Tables to return if the LLM call fails or produces
                an empty / invalid response.  Defaults to *available_tables*.

        Returns:
            List of table names (subset of *available_tables*) that the LLM
            identified as relevant.  Always validated against the known list so
            hallucinated names are silently dropped.
        """
        log = logger.bind(question=question[:80], table_count=len(available_tables))

        if not available_tables:
            log.warning("No available tables — skipping table selection")
            return []

        table_list_str = "\n".join(f"  - {t}" for t in sorted(available_tables))
        system_prompt = _TABLE_SELECTOR_SYSTEM.replace("{table_list}", table_list_str)
        user_prompt = _TABLE_SELECTOR_USER.replace("{question}", question)

        try:
            log.debug("Calling LLM for table selection")
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw = response.content.strip()
            selected = self._parse_table_list(raw, available_tables)
            log.info(
                "Table selection complete",
                selected=selected,
                tokens=response.total_tokens,
            )
            return selected or (fallback_tables or available_tables)

        except Exception as exc:
            log.warning(
                "Table selection LLM call failed — using fallback tables",
                error=str(exc),
            )
            return fallback_tables or available_tables

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_table_list(raw: str, available_tables: list[str]) -> list[str]:
        """Extract and validate a JSON table list from the LLM response.

        Args:
            raw: Raw text from the LLM (expected: a JSON array string).
            available_tables: The ground-truth list of known tables.

        Returns:
            Validated list of table names that exist in *available_tables*.
            Hallucinated / misspelled names are silently dropped.
        """
        # Find the JSON array in the response
        match = _JSON_ARRAY_RE.search(raw)
        if not match:
            logger.warning("Table selector: no JSON array found in response", raw=raw[:200])
            return []

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("Table selector: JSON parse failed", raw=raw[:200])
            return []

        if not isinstance(parsed, list):
            return []

        known = set(available_tables)
        validated: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item in known:
                validated.append(item)
            elif isinstance(item, str):
                logger.debug(
                    "Table selector: dropped unknown table name",
                    name=item,
                )

        return validated
