"""Table describer — P1: LLM-generated natural-language table descriptions.

Raw DDL ("orders.customer_id INT FK→customers.id") embeds poorly against
natural-language questions ("show me all purchases by user"). During ingestion
we ask the LLM to produce a short natural-language summary of each table and
attach it to ``TableInfo.description`` so it is embedded alongside the DDL,
closing the NL↔DDL vocabulary gap (the #1 retrieval failure mode).

Descriptions are only generated for tables that do not already have one, so an
existing DB comment or user-provided description is never overwritten.
"""
from __future__ import annotations

import asyncio

import structlog

from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.models.schema import TableInfo

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You write concise, factual descriptions of database tables to help a retrieval \
system match natural-language questions to the right table.

RULES:
1. Output ONE or TWO plain sentences — no markdown, no lists, no preamble.
2. Describe what business entity/event the table stores and how it relates to \
other tables (via foreign keys), using natural business vocabulary.
3. Do NOT restate every column or its data type verbatim.
4. Output ONLY the description text.
"""

_USER_PROMPT = """\
Table name: {name}
Columns:
{columns}

Write the natural-language description of this table."""


class TableDescriber:
    """Generates a natural-language description for each table via the LLM.

    SOLID:
      S — Only responsible for producing table descriptions.
      D — Depends on the ILLMProvider abstraction.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        concurrency: int = 4,
        max_tokens: int = 160,
        temperature: float = 0.2,
    ) -> None:
        self._llm = llm_provider
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._semaphore = asyncio.Semaphore(max(1, concurrency))

    async def enrich(self, tables: list[TableInfo]) -> int:
        """Populate ``description`` in-place for tables that lack one.

        Failures are swallowed per-table so a single bad LLM call never aborts
        ingestion. Returns the number of descriptions successfully generated.
        """
        targets = [t for t in tables if not (t.description and t.description.strip())]
        if not targets:
            return 0

        log = logger.bind(target_count=len(targets), total=len(tables))
        log.info("Generating LLM table descriptions")
        results = await asyncio.gather(
            *(self._describe(table) for table in targets),
            return_exceptions=True,
        )
        generated = 0
        for table, result in zip(targets, results, strict=True):
            if isinstance(result, str) and result.strip():
                table.description = result.strip()
                generated += 1
            elif isinstance(result, BaseException):
                log.warning(
                    "Table description generation failed — skipping",
                    table=table.name,
                    error=str(result),
                )
        log.info("Table description enrichment complete", generated=generated)
        return generated

    async def _describe(self, table: TableInfo) -> str | None:
        """Generate a single table's description (bounded by the semaphore)."""
        column_lines = "\n".join(
            self._format_column(col) for col in table.columns
        ) or "  (no columns)"
        user_prompt = _USER_PROMPT.replace("{name}", table.name).replace(
            "{columns}", column_lines
        )
        async with self._semaphore:
            response = await self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        text = (response.content or "").strip()
        # Collapse to a single line — descriptions are embedded inline with DDL.
        return " ".join(text.split()) if text else None

    @staticmethod
    def _format_column(col: object) -> str:
        name = getattr(col, "name", "")
        data_type = getattr(col, "data_type", "")
        parts = [f"  - {name} ({data_type})"]
        if getattr(col, "primary_key", False):
            parts.append("[PK]")
        fk = getattr(col, "foreign_key", None)
        if fk:
            parts.append(f"[FK -> {fk}]")
        return " ".join(parts)
