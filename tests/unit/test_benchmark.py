"""Unit tests for the execution-accuracy benchmark runner (no LLM, no DB)."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import sqlglot
from sqlglot import exp

from nl_to_sql.evaluation.benchmark import GoldenCase, load_golden_set, run_benchmark

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "golden_sql.json"

# Fake "database": SQL text -> rows.
_DB: dict[str, list[dict[str, Any]]] = {
    "GOLD_COUNT": [{"count": 5}],
    "GOOD_COUNT": [{"n": 5}],
    "GOLD_NAMES": [{"name": "Alice"}, {"name": "Bob"}],
    "GOOD_NAMES": [{"first_name": "Bob"}, {"first_name": "Alice"}],
    "BAD_NAMES": [{"first_name": "Alice"}],
    "SELECT id FROM top_ids ORDER BY id": [{"id": 1}, {"id": 2}],
    "WRONG_ORDER": [{"id": 2}, {"id": 1}],
}


async def fake_execute(sql: str) -> list[dict[str, Any]]:
    if sql not in _DB:
        raise RuntimeError(f"relation does not exist: {sql}")
    return _DB[sql]


def make_generate(answers: dict[str, Any]):
    async def generate(question: str) -> tuple[str, bool, dict[str, Any]]:
        answer = answers[question]
        if isinstance(answer, Exception):
            raise answer
        sql, valid = answer
        return sql, valid, {}

    return generate


def make_clock(step: float = 0.1):
    t = [0.0]

    def clock() -> float:
        t[0] += step
        return t[0]

    return clock


CASES = [
    GoldenCase("c1", "q_count", "GOLD_COUNT", "aggregation"),
    GoldenCase("c2", "q_names_ok", "GOLD_NAMES", "filter"),
    GoldenCase("c3", "q_names_bad", "GOLD_NAMES", "filter"),
    GoldenCase("c4", "q_crash", "GOLD_COUNT", "aggregation"),
    GoldenCase("c5", "q_invalid", "GOLD_COUNT", "join"),
    GoldenCase("c6", "q_exec_err", "GOLD_COUNT", "join"),
    GoldenCase("c7", "q_order", "SELECT id FROM top_ids ORDER BY id", "order_limit"),
]

ANSWERS: dict[str, Any] = {
    "q_count": ("GOOD_COUNT", True),
    "q_names_ok": ("GOOD_NAMES", True),
    "q_names_bad": ("BAD_NAMES", True),
    "q_crash": RuntimeError("LLM timeout"),
    "q_invalid": ("GOOD_COUNT", False),
    "q_exec_err": ("SELECT * FROM nope", True),
    "q_order": ("WRONG_ORDER", True),
}


async def test_accuracy_and_per_category_numbers() -> None:
    report = await run_benchmark(CASES, make_generate(ANSWERS), fake_execute, clock=make_clock())

    assert report.total == 7
    assert report.scored == 7
    assert report.gold_errors == 0
    assert report.correct == 2  # c1, c2
    assert report.execution_accuracy == pytest.approx(2 / 7)
    # c4 crashed (not valid), c5 invalid; all others reported valid.
    assert report.valid_sql_rate == pytest.approx(5 / 7)

    cats = report.category_accuracy
    assert (cats["aggregation"].correct, cats["aggregation"].total) == (1, 2)
    assert (cats["filter"].correct, cats["filter"].total) == (1, 2)
    assert (cats["join"].correct, cats["join"].total) == (0, 2)
    assert (cats["order_limit"].correct, cats["order_limit"].total) == (0, 1)
    assert cats["filter"].accuracy == pytest.approx(0.5)

    by_id = {c.id: c for c in report.cases}
    assert by_id["c3"].reason.startswith("row count")
    assert "order" in by_id["c7"].reason
    assert by_id["c5"].reason == "generated SQL failed validation"
    assert by_id["c6"].reason == "generated SQL failed to execute"
    assert by_id["c6"].error and "nope" in by_id["c6"].error
    assert by_id["c1"].generated_sql == "GOOD_COUNT"


async def test_generation_error_is_a_miss_not_a_crash() -> None:
    cases = [GoldenCase("c4", "q_crash", "GOLD_COUNT", "aggregation")]
    report = await run_benchmark(cases, make_generate(ANSWERS), fake_execute, clock=make_clock())
    case = report.cases[0]
    assert report.execution_accuracy == 0.0
    assert not case.match
    assert case.reason == "generation failed"
    assert case.error is not None and "LLM timeout" in case.error
    assert case.latency_ms is not None


async def test_gold_error_excluded_from_denominator() -> None:
    cases = [
        GoldenCase("ok", "q_count", "GOLD_COUNT", "aggregation"),
        GoldenCase("broken", "q_count", "MISSING_GOLD", "aggregation"),
    ]
    called: list[str] = []

    async def generate(question: str) -> tuple[str, bool, dict[str, Any]]:
        called.append(question)
        return "GOOD_COUNT", True, {}

    report = await run_benchmark(cases, generate, fake_execute, clock=make_clock())
    assert report.total == 2
    assert report.scored == 1
    assert report.gold_errors == 1
    assert report.execution_accuracy == 1.0
    assert len(called) == 1  # no LLM call spent on a broken gold case
    assert report.cases[1].gold_error


async def test_latency_stats_and_report_serialisable() -> None:
    cases = [GoldenCase(f"c{i}", "q_count", "GOLD_COUNT", "aggregation") for i in range(20)]
    report = await run_benchmark(
        cases, make_generate(ANSWERS), fake_execute, clock=make_clock(0.05)
    )
    assert report.mean_latency_ms == pytest.approx(50.0)
    assert report.p95_latency_ms == pytest.approx(50.0)
    data = json.loads(json.dumps(report.to_dict()))
    assert data["category_accuracy"]["aggregation"]["accuracy"] == 1.0
    assert len(data["cases"]) == 20


async def test_empty_case_list() -> None:
    report = await run_benchmark([], make_generate({}), fake_execute)
    assert report.total == 0
    assert report.execution_accuracy == 0.0


def test_load_golden_set_validates(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    path.write_text(json.dumps([{"id": "a", "question": "q", "gold_sql": "SELECT 1"}]))
    with pytest.raises(ValueError, match="category"):
        load_golden_set(path)
    row = {"id": "a", "question": "q", "gold_sql": "SELECT 1", "category": "x"}
    path.write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_golden_set(path)


def test_shipped_golden_set_is_well_formed() -> None:
    cases = load_golden_set(GOLDEN_PATH)
    assert 40 <= len(cases) <= 60
    categories = Counter(c.category for c in cases)
    assert len(categories) >= 8
    seed_tables = {"customers", "categories", "products", "orders", "order_items"}
    for case in cases:
        parsed = sqlglot.parse_one(case.gold_sql, read="postgres")
        assert isinstance(parsed, exp.Select), case.id
        tables = {t.name for t in parsed.find_all(exp.Table)}
        assert tables and tables <= seed_tables, case.id
