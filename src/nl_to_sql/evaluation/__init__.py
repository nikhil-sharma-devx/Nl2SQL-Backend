"""Offline evaluation of the NL→SQL pipeline (execution-accuracy benchmark).

Pure logic only: no ``api/`` imports and no infrastructure construction. The
LLM pipeline and the database are injected as async callables so everything
here is unit-testable without either.
"""

from nl_to_sql.evaluation.benchmark import (
    BenchmarkReport,
    CaseResult,
    CategoryStats,
    GoldenCase,
    load_golden_set,
    run_benchmark,
)
from nl_to_sql.evaluation.result_compare import (
    CompareResult,
    compare_result_sets,
    has_top_level_order_by,
    normalize_value,
)

__all__ = [
    "BenchmarkReport",
    "CaseResult",
    "CategoryStats",
    "CompareResult",
    "GoldenCase",
    "compare_result_sets",
    "has_top_level_order_by",
    "load_golden_set",
    "normalize_value",
    "run_benchmark",
]
