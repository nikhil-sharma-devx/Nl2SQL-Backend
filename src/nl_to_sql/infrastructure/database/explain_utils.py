"""Pure helpers for parsing PostgreSQL ``EXPLAIN (FORMAT JSON)`` output.

These functions are intentionally side-effect free (no DB, no I/O) so the plan
parsing + warning derivation can be unit-tested against canned EXPLAIN JSON.
The DB-touching part lives on ``AsyncDatabaseClient.explain``.

SOLID: S — this module only interprets a plan; it never runs one.
"""
from __future__ import annotations

from typing import Any

# A join node whose total cost meets this threshold is flagged as expensive.
EXPENSIVE_COST_THRESHOLD = 10_000.0
# A Nested Loop over at least this many estimated rows is flagged as a risky join
# (Nested Loops degrade badly on large row counts).
LARGE_ROWS_THRESHOLD = 100_000

# Leading keywords that mark a statement as a write / DDL. Used to make sure we
# never run ``EXPLAIN ANALYZE`` (which executes the statement) against a write.
_WRITE_KEYWORDS: frozenset[str] = frozenset(
    {
        "insert", "update", "delete", "merge", "truncate", "drop", "alter",
        "create", "grant", "revoke", "comment", "refresh", "reindex", "vacuum",
        "call", "do",
    }
)

_JOIN_NODES: frozenset[str] = frozenset({"Nested Loop", "Hash Join", "Merge Join"})


def _strip_leading_noise(sql: str) -> str:
    """Drop leading whitespace and ``--`` / ``/* */`` comments from ``sql``."""
    s = sql.lstrip()
    while s.startswith("--") or s.startswith("/*"):
        if s.startswith("--"):
            newline = s.find("\n")
            s = "" if newline == -1 else s[newline + 1 :].lstrip()
        else:
            end = s.find("*/")
            s = "" if end == -1 else s[end + 2 :].lstrip()
    return s


def is_write_statement(sql: str) -> bool:
    """Return True when the SQL's leading keyword indicates a write/DDL statement.

    A deliberately simple leading-keyword check — enough to guarantee we never
    ask PostgreSQL to ``EXPLAIN ANALYZE`` (i.e. execute) a mutating statement.
    """
    cleaned = _strip_leading_noise(sql or "")
    if not cleaned:
        return False
    first = cleaned.split(None, 1)[0].lower().rstrip("(;")
    return first in _WRITE_KEYWORDS


def _simplify_node(node: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw plan node to the fields the UI cares about (recursively)."""
    simplified: dict[str, Any] = {
        "node_type": node.get("Node Type"),
        "relation": node.get("Relation Name"),
        "total_cost": node.get("Total Cost"),
        "plan_rows": node.get("Plan Rows"),
    }
    children = node.get("Plans") or []
    if children:
        simplified["children"] = [_simplify_node(c) for c in children]
    return simplified


def _collect_warnings(root: dict[str, Any]) -> list[dict[str, str]]:
    """Walk the plan tree and derive human-readable performance warnings."""
    warnings: list[dict[str, str]] = []
    seen: set[tuple[Any, ...]] = set()

    def visit(node: dict[str, Any]) -> None:
        node_type = str(node.get("Node Type", ""))
        relation = node.get("Relation Name")
        rows = int(node.get("Plan Rows", 0) or 0)
        cost = float(node.get("Total Cost", 0.0) or 0.0)

        if node_type == "Seq Scan":
            scan_key: tuple[Any, ...] = ("seq_scan", relation)
            if scan_key not in seen:
                seen.add(scan_key)
                target = f" on {relation}" if relation else ""
                warnings.append(
                    {
                        "type": "seq_scan",
                        "message": (
                            f"Sequential (full-table) scan{target} — "
                            f"~{rows:,} rows estimated. Consider an index."
                        ),
                    }
                )

        if node_type in _JOIN_NODES:
            expensive = cost >= EXPENSIVE_COST_THRESHOLD or (
                node_type == "Nested Loop" and rows >= LARGE_ROWS_THRESHOLD
            )
            if expensive:
                join_key: tuple[Any, ...] = ("expensive_join", node_type, round(cost))
                if join_key not in seen:
                    seen.add(join_key)
                    warnings.append(
                        {
                            "type": "expensive_join",
                            "message": (
                                f"Potentially expensive {node_type} "
                                f"(cost ~{cost:,.0f}, ~{rows:,} rows)."
                            ),
                        }
                    )

        for child in node.get("Plans") or []:
            visit(child)

    visit(root)
    return warnings


def parse_explain_plan(plan_json: Any) -> dict[str, Any]:
    """Parse ``EXPLAIN (FORMAT JSON)`` output into a typed preview structure.

    Args:
        plan_json: The decoded JSON (a list ``[{"Plan": {...}}]`` per the
            PostgreSQL format, or the equivalent dict).

    Returns:
        A dict with ``estimated_rows`` (int), ``estimated_cost`` (float),
        ``plan`` (simplified node tree) and ``warnings`` (list of dicts).
    """
    if isinstance(plan_json, list):
        top: Any = plan_json[0] if plan_json else {}
    else:
        top = plan_json
    root: dict[str, Any] = {}
    if isinstance(top, dict):
        candidate = top.get("Plan", top)
        if isinstance(candidate, dict):
            root = candidate

    estimated_rows = int(root.get("Plan Rows", 0) or 0)
    estimated_cost = float(root.get("Total Cost", 0.0) or 0.0)

    return {
        "estimated_rows": estimated_rows,
        "estimated_cost": estimated_cost,
        "plan": _simplify_node(root) if root else None,
        "warnings": _collect_warnings(root),
    }
