"""Execution-accuracy benchmark runner.

For each golden case: generate SQL for the question (injected ``generate``),
execute both the gold SQL and the generated SQL (injected ``execute``), and
compare the result sets with :mod:`nl_to_sql.evaluation.result_compare`.

Scoring rules:

- A generation exception, empty SQL, SQL that failed validation, or a
  generated-SQL execution error is a *miss* — never a crash.
- A gold-SQL execution error means the benchmark itself is broken for that
  case: it is reported (``gold_error``) and excluded from the accuracy
  denominator rather than silently penalising the pipeline.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from nl_to_sql.evaluation.result_compare import compare_for_gold_sql

logger = structlog.get_logger(__name__)

GenerateFn = Callable[[str], Awaitable[tuple[str, bool, dict[str, Any]]]]
"""async (question) -> (sql, is_valid, meta)."""

ExecuteFn = Callable[[str], Awaitable[list[dict[str, Any]]]]
"""async (sql) -> rows. Must run against a read-only connection."""

_REQUIRED_KEYS = ("id", "question", "gold_sql", "category")
_ERROR_MAX_LEN = 300


@dataclass(frozen=True)
class GoldenCase:
    """One benchmark question with its hand-written gold SQL."""

    id: str
    question: str
    gold_sql: str
    category: str


@dataclass
class CaseResult:
    """Outcome of one benchmark case."""

    id: str
    question: str
    category: str
    match: bool
    reason: str
    generated_sql: str = ""
    is_valid: bool = False
    latency_ms: float | None = None
    error: str | None = None
    gold_error: bool = False


@dataclass(frozen=True)
class CategoryStats:
    """Accuracy for one category (gold-error cases excluded)."""

    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results."""

    total: int
    scored: int
    correct: int
    gold_errors: int
    execution_accuracy: float
    valid_sql_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    category_accuracy: dict[str, CategoryStats] = field(default_factory=dict)
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (for ``--out report.json``)."""
        data = asdict(self)
        data["category_accuracy"] = {
            name: {"total": s.total, "correct": s.correct, "accuracy": round(s.accuracy, 4)}
            for name, s in self.category_accuracy.items()
        }
        return data


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    """Load and validate a golden-set JSON file (a list of case objects)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Golden set must be a JSON list of cases")
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Golden case #{index} is not an object")
        missing = [k for k in _REQUIRED_KEYS if not str(item.get(k, "")).strip()]
        if missing:
            raise ValueError(f"Golden case #{index} missing keys: {', '.join(missing)}")
        case_id = str(item["id"])
        if case_id in seen:
            raise ValueError(f"Duplicate golden case id: {case_id}")
        seen.add(case_id)
        cases.append(
            GoldenCase(
                id=case_id,
                question=str(item["question"]).strip(),
                gold_sql=str(item["gold_sql"]).strip(),
                category=str(item["category"]).strip(),
            )
        )
    return cases


def _short(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text if len(text) <= _ERROR_MAX_LEN else text[: _ERROR_MAX_LEN - 3] + "..."


def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


async def _run_case(
    case: GoldenCase,
    generate: GenerateFn,
    execute: ExecuteFn,
    dialect: str,
    clock: Callable[[], float],
) -> CaseResult:
    result = CaseResult(
        id=case.id, question=case.question, category=case.category, match=False, reason=""
    )

    # Gold first: if the benchmark itself is broken, don't spend an LLM call.
    try:
        gold_rows = await execute(case.gold_sql)
    except Exception as exc:
        result.gold_error = True
        result.reason = "gold SQL failed to execute"
        result.error = _short(exc)
        return result

    started = clock()
    try:
        sql, is_valid, _meta = await generate(case.question)
    except Exception as exc:
        result.latency_ms = (clock() - started) * 1000
        result.reason = "generation failed"
        result.error = _short(exc)
        return result
    result.latency_ms = (clock() - started) * 1000
    result.generated_sql = (sql or "").strip()
    result.is_valid = bool(is_valid)

    if not result.generated_sql:
        result.reason = "no SQL generated"
        return result
    if not result.is_valid:
        result.reason = "generated SQL failed validation"
        return result

    try:
        pred_rows = await execute(result.generated_sql)
    except Exception as exc:
        result.reason = "generated SQL failed to execute"
        result.error = _short(exc)
        return result

    comparison = compare_for_gold_sql(case.gold_sql, gold_rows, pred_rows, dialect)
    result.match = comparison.match
    result.reason = comparison.reason
    return result


async def run_benchmark(
    cases: Iterable[GoldenCase],
    generate: GenerateFn,
    execute: ExecuteFn,
    *,
    dialect: str = "postgres",
    clock: Callable[[], float] = time.perf_counter,
    on_case: Callable[[CaseResult], None] | None = None,
) -> BenchmarkReport:
    """Run every case sequentially and aggregate an execution-accuracy report.

    Args:
        cases: Golden cases to evaluate.
        generate: async question -> (sql, is_valid, meta).
        execute: async sql -> rows (read-only connection).
        dialect: sqlglot dialect of the gold SQL (for ORDER BY detection).
        clock: Monotonic seconds clock (injectable for tests).
        on_case: Optional progress callback invoked after each case.
    """
    results: list[CaseResult] = []
    for case in cases:
        result = await _run_case(case, generate, execute, dialect, clock)
        results.append(result)
        logger.info(
            "Benchmark case evaluated",
            case_id=case.id,
            category=case.category,
            match=result.match,
            reason=result.reason,
        )
        if on_case is not None:
            on_case(result)

    scored = [r for r in results if not r.gold_error]
    correct = sum(1 for r in scored if r.match)
    valid = sum(1 for r in scored if r.is_valid)
    latencies = [r.latency_ms for r in scored if r.latency_ms is not None]

    per_category: dict[str, list[CaseResult]] = {}
    for r in scored:
        per_category.setdefault(r.category, []).append(r)
    category_accuracy = {
        name: CategoryStats(total=len(rs), correct=sum(1 for r in rs if r.match))
        for name, rs in sorted(per_category.items())
    }

    return BenchmarkReport(
        total=len(results),
        scored=len(scored),
        correct=correct,
        gold_errors=len(results) - len(scored),
        execution_accuracy=correct / len(scored) if scored else 0.0,
        valid_sql_rate=valid / len(scored) if scored else 0.0,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p95_latency_ms=_p95(latencies),
        category_accuracy=category_accuracy,
        cases=results,
    )
