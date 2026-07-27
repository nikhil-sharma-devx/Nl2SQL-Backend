"""SchemaDocService — RAG-powered natural-language explanations for schema.

When a user clicks a table or column in the schema UI, this service produces a
structured explanation grounded in the user's own schema documentation:

  - **Description** — what the table/column is.
  - **Business meaning** — why it exists / what it represents.
  - **Relationships** — FK links (both incoming and outgoing), derived from the
    authoritative per-user catalog (not guessed by the LLM).
  - **Example usage** — a short natural-language usage note.
  - **Common joins** — column-level JOIN clauses derived from FK metadata.
  - **Example SQL** — a runnable sample query.

The factual parts (relationships, common joins, a default example SQL) are
derived deterministically from the catalog + FK metadata *first*, so they are
always available even when the LLM is unreachable — the LLM only *enriches* the
prose. Results are cached per (user_id, schema, table[, column]) with a TTL.

SOLID:
  S — Only responsible for building schema explanations.
  D — Depends on abstractions (IVectorStore, ILLMProvider, ICache) + the
      catalog service; never instantiates infrastructure itself.
"""
from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel, Field

from nl_to_sql.core.exceptions import TableNotFoundError
from nl_to_sql.core.interfaces.i_cache import ICache
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.services.schema_catalog_service import SchemaCatalogService

logger = structlog.get_logger(__name__)


class SchemaExplanation(BaseModel):
    """Structured, cache-friendly explanation of a table or column."""

    table: str
    column: str | None = None
    schema_name: str = "public"
    description: str = ""
    business_meaning: str = ""
    relationships: list[str] = Field(default_factory=list)
    example_usage: str = ""
    common_joins: list[str] = Field(default_factory=list)
    example_sql: str = ""
    # True when served from cache; helps the FE label instant responses.
    cached: bool = False
    # True when the LLM prose was used; False means pure schema-derived fallback.
    llm_generated: bool = False


_SYSTEM_PROMPT = """\
You are a database documentation assistant. Given a database table (and \
optionally a specific column) plus its schema and foreign-key facts, write a \
concise, accurate explanation for an analyst.

Respond with a SINGLE JSON object and nothing else, using exactly these keys:
  "description":      string — what the table/column is (1-2 sentences).
  "business_meaning": string — what it represents for the business (1-2 sentences).
  "example_usage":    string — a short note on how an analyst would use it.
  "example_sql":      string — one runnable example SQL query using this table.

Do not invent columns or tables that are not in the provided schema. Keep each \
field short. Output only the JSON object."""


class SchemaDocService:
    """Builds RAG-grounded explanations for schema tables/columns (per-user)."""

    def __init__(
        self,
        catalog: SchemaCatalogService,
        vector_store: IVectorStore,
        llm_provider: ILLMProvider,
        cache: ICache,
        per_user_isolation: bool = False,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self._catalog = catalog
        self._vector_store = vector_store
        self._llm = llm_provider
        self._cache = cache
        self._isolation = per_user_isolation
        self._cache_ttl = cache_ttl_seconds

    # ── Public API ──────────────────────────────────────────────────────────

    async def explain(
        self, *, user_id: str, connection_id: str, table: str, column: str | None = None
    ) -> SchemaExplanation:
        """Return a cached-or-generated explanation for a table/column.

        Args:
            user_id: The authenticated user (ownership + cache/log scoping).
            connection_id: The active database connection (scopes catalog +
                vector reads so explanations reflect the selected database).
            table: Table name to explain.
            column: Optional column name to focus the explanation on.

        Returns:
            A populated ``SchemaExplanation``.

        Raises:
            TableNotFoundError: If the table (or column) does not exist for the
                connection. Tables from other connections are reported as not found.
        """
        # Resolve the table from the connection's catalog (source of truth). The
        # catalog query is scoped to connection_id, so a table on another
        # connection simply isn't in ``tables`` → reported as not found.
        catalog = await self._catalog.get_catalog(user_id, connection_id)
        all_tables: list[dict[str, Any]] = catalog.get("tables", [])
        target = next((t for t in all_tables if t["table_name"] == table), None)
        if target is None:
            raise TableNotFoundError(f"Table '{table}' was not found.")

        schema_name = target.get("schema_name", "public")
        columns: list[dict[str, Any]] = target.get("columns") or []

        target_column: dict[str, Any] | None = None
        if column is not None:
            target_column = next((c for c in columns if c.get("name") == column), None)
            if target_column is None:
                raise TableNotFoundError(
                    f"Column '{column}' was not found on table '{table}'."
                )

        cache_key = self._cache_key(connection_id, schema_name, table, column)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            log = logger.bind(user_id=user_id, connection_id=connection_id, table=table, column=column)
            log.info("Schema explanation cache hit")
            cached.cached = True
            return cached

        log = logger.bind(user_id=user_id, connection_id=connection_id, table=table, column=column)

        # Deterministic, FK-accurate parts (always available — the fallback base).
        relationships = self._derive_relationships(target, all_tables)
        common_joins = self._derive_joins(target, all_tables)
        default_sql = self._default_example_sql(table, target_column, columns)

        explanation = SchemaExplanation(
            table=table,
            column=column,
            schema_name=schema_name,
            description=self._default_description(target, target_column),
            business_meaning="",
            relationships=relationships,
            example_usage="",
            common_joins=common_joins,
            example_sql=default_sql,
        )

        # Retrieve schema chunk(s) for RAG grounding (connection-scoped).
        chunk_text = await self._retrieve_context(connection_id, table)

        # Enrich the prose via the LLM; on any failure keep the derived parts.
        llm_fields = await self._summarize(
            table=table,
            column=column,
            target=target,
            columns=columns,
            chunk_text=chunk_text,
            relationships=relationships,
            common_joins=common_joins,
            log=log,
        )
        if llm_fields is not None:
            explanation.description = llm_fields.get("description") or explanation.description
            explanation.business_meaning = llm_fields.get("business_meaning") or ""
            explanation.example_usage = llm_fields.get("example_usage") or ""
            explanation.example_sql = llm_fields.get("example_sql") or explanation.example_sql
            explanation.llm_generated = True
            log.info("Schema explanation generated via LLM")
        else:
            log.info("Schema explanation served from schema-derived fallback")

        await self._set_cached(cache_key, explanation)
        return explanation

    # ── Cache ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(
        connection_id: str, schema_name: str, table: str, column: str | None
    ) -> str:
        return f"schema_explain:{connection_id}:{schema_name}:{table}:{column or ''}"

    async def _get_cached(self, key: str) -> SchemaExplanation | None:
        try:
            raw = await self._cache.get(key)
        except Exception as exc:
            logger.warning("Schema explanation cache read failed", error=str(exc))
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return SchemaExplanation.model_validate(raw)
        except Exception:
            return None

    async def _set_cached(self, key: str, value: SchemaExplanation) -> None:
        try:
            # Never persist the transient ``cached`` flag as True.
            payload = value.model_dump()
            payload["cached"] = False
            await self._cache.set(key, payload, ttl=self._cache_ttl)
        except Exception as exc:
            logger.warning("Schema explanation cache write failed", error=str(exc))

    # ── Retrieval (RAG grounding) ─────────────────────────────────────────────

    async def _retrieve_context(self, connection_id: str, table: str) -> str:
        """Fetch the table's schema chunk(s) from the vector store (connection-scoped)."""
        cid = connection_id if self._isolation else None
        try:
            chunks = await self._vector_store.get_chunks_by_table_names([table], connection_id=cid)
        except Exception as exc:
            logger.warning("Schema chunk retrieval failed", table=table, error=str(exc))
            return ""
        return "\n\n---\n\n".join(c.content for c in chunks if c.content)

    # ── FK derivations (deterministic, from the authoritative catalog) ─────────

    @staticmethod
    def _derive_relationships(
        target: dict[str, Any], all_tables: list[dict[str, Any]]
    ) -> list[str]:
        """Return human-readable FK relationships (outgoing + incoming)."""
        table = target["table_name"]
        rels: list[str] = []
        seen: set[str] = set()

        # Outgoing: this table references others.
        for col in target.get("columns") or []:
            fk = col.get("foreign_key")
            if fk:
                text = f"{table}.{col['name']} → {fk}"
                if text not in seen:
                    seen.add(text)
                    rels.append(text)

        # Incoming: other tables reference this table.
        for other in all_tables:
            if other["table_name"] == table:
                continue
            for col in other.get("columns") or []:
                fk = col.get("foreign_key")
                if fk and fk.split(".")[0] == table:
                    text = f"{other['table_name']}.{col['name']} → {fk}"
                    if text not in seen:
                        seen.add(text)
                        rels.append(text)
        return rels

    @staticmethod
    def _derive_joins(
        target: dict[str, Any], all_tables: list[dict[str, Any]]
    ) -> list[str]:
        """Return column-level JOIN clauses derived from FK metadata."""
        table = target["table_name"]
        joins: list[str] = []
        seen: set[str] = set()

        def _add(text: str) -> None:
            if text not in seen:
                seen.add(text)
                joins.append(text)

        # Outgoing FKs → JOIN the referenced table.
        for col in target.get("columns") or []:
            fk = col.get("foreign_key")
            if not fk or "." not in fk:
                continue
            ref_table, ref_col = fk.split(".", 1)
            _add(
                f"{table} JOIN {ref_table} "
                f"ON {table}.{col['name']} = {ref_table}.{ref_col}"
            )

        # Incoming FKs → JOIN the referencing table.
        for other in all_tables:
            if other["table_name"] == table:
                continue
            for col in other.get("columns") or []:
                fk = col.get("foreign_key")
                if not fk or "." not in fk:
                    continue
                ref_table, ref_col = fk.split(".", 1)
                if ref_table == table:
                    _add(
                        f"{table} JOIN {other['table_name']} "
                        f"ON {table}.{ref_col} = {other['table_name']}.{col['name']}"
                    )
        return joins

    @staticmethod
    def _default_description(
        target: dict[str, Any], target_column: dict[str, Any] | None
    ) -> str:
        """Best-effort description from catalog metadata (fallback base)."""
        if target_column is not None:
            desc = target_column.get("description")
            if desc:
                return str(desc)
            dtype = target_column.get("data_type") or "unknown type"
            return (
                f"Column '{target_column['name']}' ({dtype}) on table "
                f"'{target['table_name']}'."
            )
        desc = target.get("description")
        if desc:
            return str(desc)
        n = len(target.get("columns") or [])
        return f"Table '{target['table_name']}' with {n} column(s)."

    @staticmethod
    def _default_example_sql(
        table: str,
        target_column: dict[str, Any] | None,
        columns: list[dict[str, Any]],
    ) -> str:
        # Display-only sample SQL rendered in the UI popover — never executed here.
        if target_column is not None:
            return f"SELECT {target_column['name']} FROM {table} LIMIT 10;"  # noqa: S608
        return f"SELECT * FROM {table} LIMIT 10;"  # noqa: S608

    # ── LLM summarization ──────────────────────────────────────────────────────

    async def _summarize(
        self,
        *,
        table: str,
        column: str | None,
        target: dict[str, Any],
        columns: list[dict[str, Any]],
        chunk_text: str,
        relationships: list[str],
        common_joins: list[str],
        log: structlog.BoundLogger,
    ) -> dict[str, str] | None:
        """Ask the LLM for enriched prose. Returns None on any failure."""
        user_prompt = self._build_user_prompt(
            table=table,
            column=column,
            target=target,
            columns=columns,
            chunk_text=chunk_text,
            relationships=relationships,
            common_joins=common_joins,
        )
        try:
            response = await self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            log.warning("Schema explanation LLM call failed", error=str(exc))
            return None

        parsed = self._parse_json(response.content)
        if parsed is None:
            log.warning("Schema explanation LLM returned unparseable output")
            return None
        return {
            "description": str(parsed.get("description", "")).strip(),
            "business_meaning": str(parsed.get("business_meaning", "")).strip(),
            "example_usage": str(parsed.get("example_usage", "")).strip(),
            "example_sql": str(parsed.get("example_sql", "")).strip(),
        }

    @staticmethod
    def _build_user_prompt(
        *,
        table: str,
        column: str | None,
        target: dict[str, Any],
        columns: list[dict[str, Any]],
        chunk_text: str,
        relationships: list[str],
        common_joins: list[str],
    ) -> str:
        col_lines = []
        for c in columns:
            flags = []
            if c.get("primary_key"):
                flags.append("PK")
            if c.get("foreign_key"):
                flags.append(f"FK→{c['foreign_key']}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            col_lines.append(f"  - {c.get('name')} ({c.get('data_type', '')}){flag_str}")

        parts: list[str] = [f"Table: {table}"]
        if target.get("description"):
            parts.append(f"Table description: {target['description']}")
        parts.append("Columns:\n" + ("\n".join(col_lines) or "  (none)"))
        if column is not None:
            parts.append(f"Focus specifically on the column: {column}")
        if relationships:
            parts.append("Foreign-key relationships:\n" + "\n".join(f"  - {r}" for r in relationships))
        if common_joins:
            parts.append("Common joins:\n" + "\n".join(f"  - {j}" for j in common_joins))
        if chunk_text:
            parts.append("Retrieved schema documentation:\n" + chunk_text)
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any] | None:
        """Parse a JSON object out of an LLM response (tolerant of fences)."""
        text = (content or "").strip()
        if not text:
            return None
        # Strip markdown code fences if present.
        if text.startswith("```"):
            text = text.strip("`")
            # Drop an optional leading "json" language tag.
            if text[:4].lower() == "json":
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Last resort: extract the first {...} block.
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                parsed = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                return None
        return parsed if isinstance(parsed, dict) else None
