"""Per-connection access policy — which tables the AI may use, which columns
are restricted.

A connection owner can:
  - limit the connection to an allow-list of tables (``allowed_tables``;
    ``None`` = every table), and
  - mark ``table.column`` pairs as restricted (``restricted_columns``).

Enforcement happens in three independent layers so no single miss leaks data:

  1. **Schema context** (:meth:`AccessPolicy.filter_chunks`) — disallowed tables
     and restricted column lines are removed before the LLM ever sees them.
  2. **SQL check** (:meth:`AccessPolicy.violations`) — every statement is parsed
     with sqlglot before execution; a reference to a disallowed table or a
     restricted column (in any clause, under any alias) is rejected. If the SQL
     can't be parsed while a policy is active, it is rejected (fail closed).
  3. **Row redaction** (:meth:`AccessPolicy.redact_rows`) — result columns whose
     name matches a restricted column of a referenced table are replaced with
     ``REDACTED``. This catches ``SELECT *`` / ``t.*``, the one shape layer 2
     allows because it can't know the expansion without the catalog.

Pure logic, no I/O. Applied to every execution path by
``infrastructure/database/policy_client.PolicyEnforcingClient``
(lives in ``core/`` because it is pure domain logic every layer may use).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import sqlglot
from sqlglot import exp

from nl_to_sql.core.models.schema import SchemaChunk

REDACTED = "[restricted]"
_DIALECTS = {"postgresql": "postgres", "postgres": "postgres", "mysql": "mysql"}
_COLUMN_LINE_RE = re.compile(r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_EXPLAIN_RE = re.compile(r"^\s*EXPLAIN\b(\s*\([^)]*\))?(\s+ANALYZE)?(\s+VERBOSE)?", re.IGNORECASE)


def _bare(name: str) -> str:
    """'public.Orders' -> 'orders'."""
    return name.strip().strip('"`').split(".")[-1].strip('"`').lower()


@dataclass(frozen=True)
class AccessPolicy:
    """Immutable access rules for one database connection."""

    allowed_tables: frozenset[str] | None = None
    # table -> restricted column names (all lower-case, unqualified)
    restricted: dict[str, frozenset[str]] = field(default_factory=dict)

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        allowed_tables: Iterable[str] | None,
        restricted_columns: Iterable[str] | None,
    ) -> AccessPolicy:
        """Build from the stored JSON columns, normalising names."""
        allowed = (
            frozenset(_bare(t) for t in allowed_tables if t and t.strip())
            if allowed_tables is not None
            else None
        )
        restricted: dict[str, set[str]] = {}
        for entry in restricted_columns or []:
            if not entry or "." not in entry:
                continue
            table, _, column = entry.strip().rpartition(".")
            if table and column:
                restricted.setdefault(_bare(table), set()).add(column.strip('"`').lower())
        return cls(allowed, {t: frozenset(c) for t, c in restricted.items()})

    @property
    def is_empty(self) -> bool:
        return self.allowed_tables is None and not self.restricted

    def table_allowed(self, table: str) -> bool:
        return self.allowed_tables is None or _bare(table) in self.allowed_tables

    def restricted_in(self, table: str) -> frozenset[str]:
        return self.restricted.get(_bare(table), frozenset())

    def is_restricted(self, table: str, column: str) -> bool:
        return column.lower() in self.restricted_in(table)

    # ── Layer 1: schema context ─────────────────────────────────────────────

    def filter_chunks(self, chunks: list[SchemaChunk]) -> list[SchemaChunk]:
        """Drop disallowed tables and strip restricted column lines."""
        if self.is_empty:
            return chunks
        kept: list[SchemaChunk] = []
        for chunk in chunks:
            if not self.table_allowed(chunk.table_name):
                continue
            hidden = self.restricted_in(chunk.table_name)
            if chunk.metadata.get("column") and chunk.metadata["column"].lower() in hidden:
                continue  # a column-level child chunk for a restricted column
            if hidden:
                lines = [
                    line
                    for line in chunk.content.split("\n")
                    if not ((m := _COLUMN_LINE_RE.match(line)) and m.group(1).lower() in hidden)
                ]
                chunk = chunk.model_copy(update={"content": "\n".join(lines)})
            kept.append(chunk)
        return kept

    def filter_table_names(self, tables: Iterable[str]) -> list[str]:
        return [t for t in tables if self.table_allowed(t)]

    # ── Layer 2: SQL check ──────────────────────────────────────────────────

    def violations(self, sql: str, dialect: str = "postgresql") -> list[str]:
        """Return human-readable reasons this SQL breaks the policy ([] = allowed)."""
        if self.is_empty:
            return []
        body = _EXPLAIN_RE.sub("", sql, count=1)
        try:
            statements = [
                s
                for s in sqlglot.parse(body, read=_DIALECTS.get(dialect, dialect))
                if s is not None
            ]
        except Exception:
            return ["The query could not be parsed for an access-policy check, so it was blocked."]
        if not statements:
            return ["The query could not be parsed for an access-policy check, so it was blocked."]

        problems: list[str] = []
        for statement in statements:
            problems.extend(self._statement_violations(cast(exp.Expression, statement)))
        # De-duplicate, keep order.
        return list(dict.fromkeys(problems))

    def _statement_violations(self, statement: exp.Expression) -> list[str]:
        cte_names = {_bare(c.alias_or_name) for c in statement.find_all(exp.CTE)}
        problems: list[str] = []

        alias_to_table: dict[str, str] = {}
        real_tables: list[str] = []
        for table in statement.find_all(exp.Table):
            name = _bare(table.name)
            if not name or name in cte_names:
                continue
            real_tables.append(name)
            alias_to_table[name] = name
            if table.alias:
                alias_to_table[table.alias.lower()] = name
            if not self.table_allowed(name):
                problems.append(f"Table `{name}` is not available on this connection.")

        tables_in_scope = set(real_tables)
        for column in statement.find_all(exp.Column):
            col = column.name.lower() if column.name else ""
            if not col:
                continue
            qualifier = column.table.lower() if column.table else ""
            if qualifier:
                owner = alias_to_table.get(qualifier)
                candidates = {owner} if owner else set()
            else:
                # Unqualified: conservatively check every table in scope.
                candidates = tables_in_scope
            for candidate in candidates:
                if self.is_restricted(candidate, col):
                    problems.append(f"Column `{candidate}.{col}` is restricted on this connection.")
        return problems

    def referenced_tables(self, sql: str, dialect: str = "postgresql") -> set[str]:
        """Best-effort set of real tables a statement reads (for redaction)."""
        try:
            parsed = sqlglot.parse(sql, read=_DIALECTS.get(dialect, dialect))
        except Exception:
            return set(self.restricted)  # unknown → assume every restricted table
        names: set[str] = set()
        for statement in parsed:
            if statement is None:
                continue
            ctes = {_bare(c.alias_or_name) for c in statement.find_all(exp.CTE)}
            names.update(
                _bare(t.name) for t in statement.find_all(exp.Table) if _bare(t.name) not in ctes
            )
        return names

    # ── Layer 3: row redaction ──────────────────────────────────────────────

    def redact_rows(
        self, rows: list[dict[str, Any]], referenced_tables: Iterable[str]
    ) -> list[dict[str, Any]]:
        """Mask values of restricted columns that surfaced via ``*`` expansion."""
        if not self.restricted or not rows:
            return rows
        hidden: set[str] = set()
        for table in referenced_tables:
            hidden |= self.restricted_in(table)
        if not hidden:
            return rows
        return [
            {k: (REDACTED if k.lower() in hidden else v) for k, v in row.items()} for row in rows
        ]
