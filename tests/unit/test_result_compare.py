"""Unit tests for execution-equivalence result-set comparison."""

import datetime as dt
from decimal import Decimal

import pytest

from nl_to_sql.evaluation.result_compare import (
    compare_for_gold_sql,
    compare_result_sets,
    has_top_level_order_by,
    normalize_value,
)


class TestNormalizeValue:
    def test_numbers_round_to_four_places_and_unify_types(self) -> None:
        assert normalize_value(Decimal("238.11500")) == normalize_value(238.115)
        assert normalize_value(5) == normalize_value(Decimal("5.00")) == normalize_value(5.0)
        assert normalize_value(1 / 3) == normalize_value(Decimal("0.33333333"))

    def test_dates_become_iso_strings(self) -> None:
        assert normalize_value(dt.date(2026, 1, 2)) == "2026-01-02"
        assert normalize_value(dt.datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05"

    def test_strings_stripped_and_none_preserved(self) -> None:
        assert normalize_value("  Alice ") == "Alice"
        assert normalize_value(None) is None


class TestCompareResultSets:
    def test_aliases_ignored(self) -> None:
        gold = [{"count": 5}]
        pred = [{"customer_count": 5}]
        assert compare_result_sets(gold, pred).match

    def test_column_order_ignored(self) -> None:
        gold = [{"name": "Lamp", "price": Decimal("49.99")}, {"name": "Hub", "price": 29.99}]
        pred = [{"p": 29.99, "n": "Hub"}, {"p": 49.99, "n": "Lamp"}]
        assert compare_result_sets(gold, pred).match

    def test_column_permutation_must_be_consistent_across_rows(self) -> None:
        gold = [{"a": 1, "b": 2}, {"a": 2, "b": 1}]
        pred = [{"x": 1, "y": 2}, {"x": 2, "y": 1}]
        assert compare_result_sets(gold, pred).match
        bad = [{"x": 1, "y": 1}, {"x": 2, "y": 2}]
        assert not compare_result_sets(gold, bad).match

    def test_row_order_ignored_without_order_by(self) -> None:
        gold = [{"id": 1}, {"id": 2}, {"id": 3}]
        pred = [{"id": 3}, {"id": 1}, {"id": 2}]
        assert compare_result_sets(gold, pred, order_matters=False).match

    def test_row_order_enforced_with_order_by(self) -> None:
        gold = [{"id": 1}, {"id": 2}]
        pred = [{"id": 2}, {"id": 1}]
        result = compare_result_sets(gold, pred, order_matters=True)
        assert not result.match
        assert "order" in result.reason
        assert compare_result_sets(gold, list(gold), order_matters=True).match

    def test_order_by_detected_from_gold_sql(self) -> None:
        gold = [{"id": 1}, {"id": 2}]
        pred = [{"id": 2}, {"id": 1}]
        assert not compare_for_gold_sql("SELECT id FROM t ORDER BY id", gold, pred).match
        assert compare_for_gold_sql("SELECT id FROM t", gold, pred).match

    def test_numeric_and_date_normalisation(self) -> None:
        gold = [{"total": Decimal("1534.96"), "day": dt.date(2026, 1, 1)}]
        pred = [{"total": 1534.9600000001, "day": "2026-01-01"}]
        assert compare_result_sets(gold, pred).match

    def test_duplicates_are_a_multiset(self) -> None:
        gold = [{"n": "Alice"}, {"n": "Alice"}, {"n": "Bob"}]
        assert compare_result_sets(gold, [{"n": "Bob"}, {"n": "Alice"}, {"n": "Alice"}]).match
        # Same distinct values, different multiplicities -> mismatch.
        assert not compare_result_sets(gold, [{"n": "Alice"}, {"n": "Bob"}, {"n": "Bob"}]).match
        # DISTINCT-collapsed answer -> row count mismatch.
        result = compare_result_sets(gold, [{"n": "Alice"}, {"n": "Bob"}])
        assert not result.match
        assert "row count" in result.reason

    def test_column_count_mismatch(self) -> None:
        gold = [{"name": "Lamp"}]
        pred = [{"id": 5, "name": "Lamp"}]
        result = compare_result_sets(gold, pred)
        assert not result.match
        assert "column count" in result.reason

    def test_value_mismatch(self) -> None:
        result = compare_result_sets([{"n": 5}], [{"n": 6}])
        assert not result.match
        assert result.reason == "values differ"

    def test_both_empty_match(self) -> None:
        assert compare_result_sets([], []).match

    def test_null_values(self) -> None:
        assert compare_result_sets([{"a": None, "b": 1}], [{"b": 1, "a": None}]).match
        assert not compare_result_sets([{"a": None}], [{"a": 0}]).match


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT name FROM products ORDER BY price DESC LIMIT 3", True),
        ("SELECT name FROM products", False),
        ("SELECT name FROM (SELECT name FROM products ORDER BY price) s", False),
        ("SELECT * FROM c WHERE id = (SELECT id FROM o ORDER BY amt DESC LIMIT 1)", False),
        ("WITH x AS (SELECT 1 AS a) SELECT a FROM x ORDER BY a", True),
        ("SELECT a FROM t UNION SELECT a FROM u ORDER BY a", True),
        ("not sql at all (((", False),
    ],
)
def test_has_top_level_order_by(sql: str, expected: bool) -> None:
    assert has_top_level_order_by(sql) is expected
