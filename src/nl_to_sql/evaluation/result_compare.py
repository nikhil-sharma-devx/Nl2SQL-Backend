"""Execution-equivalence comparison of two SQL result sets.

Two result sets are considered equivalent when they contain the same rows of
the same values, regardless of:

- column names / aliases (``SELECT count(*)`` vs ``SELECT count(*) AS n``),
- column order (``SELECT name, price`` vs ``SELECT price, name``),
- row order — *unless* the gold SQL has a top-level ``ORDER BY``, in which
  case the question asked for an ordering and row order must match too.

Rows are compared as a multiset (duplicates count). Values are normalised
before comparison: numbers (int / float / Decimal) are rounded to 4 decimal
places, dates / datetimes / times become ISO strings, strings are stripped,
and ``None`` stays ``None``. A differing column count is always a mismatch.

Note: rows arrive as ``list[dict]``; a query projecting two columns with the
same name collapses to one key, so gold SQL should alias duplicates.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from collections import Counter
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import sqlglot
from sqlglot import exp

# Upper bound on column-permutation candidates tried before giving up. Columns
# are pre-filtered by their value multiset so real result sets almost always
# have exactly one candidate; this only guards pathological inputs.
_MAX_PERMUTATIONS = 10_000

NUMERIC_PRECISION = 4

Row = tuple[Hashable, ...]


@dataclass(frozen=True)
class CompareResult:
    """Outcome of comparing a generated result set against the gold one."""

    match: bool
    reason: str


def normalize_value(value: Any) -> Hashable:
    """Normalise one cell so semantically-equal values compare (and hash) equal."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass — keep it distinct
        return value
    if isinstance(value, int | float | Decimal):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return str(number)
        rounded = round(number, NUMERIC_PRECISION)
        return 0.0 if rounded == 0 else rounded  # fold -0.0 into 0.0
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return round(value.total_seconds(), NUMERIC_PRECISION)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value).hex()
    if isinstance(value, list | tuple):
        return tuple(normalize_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), repr(normalize_value(v))) for k, v in value.items()))
    return str(value)


def has_top_level_order_by(sql: str, dialect: str = "postgres") -> bool:
    """True when the outermost query has an ``ORDER BY`` (subqueries/CTEs ignored).

    Unparseable SQL is treated as unordered (the lenient choice).
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return False
    if parsed is None:
        return False
    return isinstance(parsed, exp.Query) and parsed.args.get("order") is not None


def _normalize_rows(rows: Sequence[dict[str, Any]]) -> list[Row]:
    return [tuple(normalize_value(v) for v in row.values()) for row in rows]


def _column_candidates(gold: list[Row], pred: list[Row], width: int) -> list[list[int]]:
    """For each gold column, the predicted columns holding the same value multiset."""
    gold_cols = [Counter(r[i] for r in gold) for i in range(width)]
    pred_cols = [Counter(r[j] for r in pred) for j in range(width)]
    return [[j for j in range(width) if pred_cols[j] == gold_cols[i]] for i in range(width)]


def _assignments(candidates: list[list[int]]) -> Iterator[list[int]]:
    """Yield injective gold→pred column mappings, depth-first, capped."""
    width = len(candidates)
    chosen: list[int] = []
    used: set[int] = set()
    budget = [_MAX_PERMUTATIONS]

    def walk(i: int) -> Iterator[list[int]]:
        if budget[0] <= 0:
            return
        if i == width:
            budget[0] -= 1
            yield list(chosen)
            return
        for j in candidates[i]:
            if j in used:
                continue
            used.add(j)
            chosen.append(j)
            yield from walk(i + 1)
            chosen.pop()
            used.discard(j)

    yield from walk(0)


def compare_result_sets(
    gold_rows: Sequence[dict[str, Any]],
    pred_rows: Sequence[dict[str, Any]],
    *,
    order_matters: bool = False,
) -> CompareResult:
    """Compare a generated result set against the gold result set.

    Args:
        gold_rows: Rows returned by the hand-written gold SQL.
        pred_rows: Rows returned by the generated SQL.
        order_matters: Require identical row order (set when the gold SQL has a
            top-level ``ORDER BY`` — see :func:`has_top_level_order_by`).
    """
    if len(gold_rows) != len(pred_rows):
        return CompareResult(
            False, f"row count differs: gold={len(gold_rows)} generated={len(pred_rows)}"
        )
    if not gold_rows:
        return CompareResult(True, "match (both empty)")

    gold = _normalize_rows(gold_rows)
    pred = _normalize_rows(pred_rows)
    gold_width = len(gold[0])
    pred_width = len(pred[0])
    if gold_width != pred_width:
        return CompareResult(
            False, f"column count differs: gold={gold_width} generated={pred_width}"
        )

    candidates = _column_candidates(gold, pred, gold_width)
    if any(not c for c in candidates):
        return CompareResult(False, "values differ")

    gold_counter = Counter(gold) if not order_matters else None
    same_rows_wrong_order = False
    for mapping in _assignments(candidates):
        permuted = [tuple(row[j] for j in mapping) for row in pred]
        if order_matters:
            if permuted == gold:
                return CompareResult(True, "match")
            if Counter(permuted) == Counter(gold):
                same_rows_wrong_order = True
        elif Counter(permuted) == gold_counter:
            return CompareResult(True, "match")

    if same_rows_wrong_order:
        return CompareResult(False, "row order differs (gold SQL has ORDER BY)")
    return CompareResult(False, "values differ")


def compare_for_gold_sql(
    gold_sql: str,
    gold_rows: Sequence[dict[str, Any]],
    pred_rows: Sequence[dict[str, Any]],
    dialect: str = "postgres",
) -> CompareResult:
    """Compare result sets, deriving order-sensitivity from the gold SQL."""
    return compare_result_sets(
        gold_rows, pred_rows, order_matters=has_top_level_order_by(gold_sql, dialect)
    )
