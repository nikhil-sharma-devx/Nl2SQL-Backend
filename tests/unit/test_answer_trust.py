"""Answer, assumptions, confidence, and sample values — the trust features.

Covers the pure logic (confidence levels, assumption cleaning, row-payload
bounds, sample-value selection) and the failure behaviour of the LLM-backed
summarizer (never fatal, never invents an answer from nothing).
"""

from __future__ import annotations

from typing import Any

import pytest

from nl_to_sql.core.models.schema import ColumnInfo, SchemaMetadata, TableInfo
from nl_to_sql.core.models.sql_result import LLMResponse
from nl_to_sql.rag.ingestion.doc_builder import DocBuilder
from nl_to_sql.rag.ingestion.sample_values import (
    MAX_DISTINCT,
    apply_sample_values,
    is_profileable,
    profile_sample_values,
)
from nl_to_sql.services.answer_confidence import assess_confidence
from nl_to_sql.services.answer_summarizer import (
    MAX_PAYLOAD_CHARS,
    MAX_ROWS,
    AnswerSummarizer,
    build_result_payload,
)
from nl_to_sql.services.sql_generator import _clean_assumptions


def _assess(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "sql": "SELECT 1",
        "is_valid": True,
        "validation_errors": [],
        "attempts": 1,
        "retrieved_tables": ["orders"],
        "used_tables": ["orders"],
        "execute_requested": True,
        "execution_result": [{"n": 1}],
        "execution_error": None,
        "empty_result_warning": None,
        "assumptions": [],
    }
    base.update(overrides)
    return assess_confidence(**base)


class TestConfidence:
    def test_clean_first_attempt_is_high(self) -> None:
        a = _assess()
        assert (a.level, a.reasons) == ("high", [])

    def test_cannot_answer_is_low(self) -> None:
        a = _assess(sql="-- CANNOT_ANSWER", is_valid=False)
        assert a.level == "low"
        assert "don't appear to contain" in a.reasons[0]

    def test_invalid_after_retries_is_low_and_counts_attempts(self) -> None:
        a = _assess(is_valid=False, validation_errors=["bad column"], attempts=3)
        assert a.level == "low"
        assert "3 attempts" in a.reasons[0]

    def test_execution_error_is_low(self) -> None:
        assert _assess(execution_error="boom").level == "low"

    def test_no_tables_is_low(self) -> None:
        assert _assess(retrieved_tables=[]).level == "low"

    def test_self_correction_is_medium(self) -> None:
        a = _assess(attempts=2)
        assert a.level == "medium"
        assert "attempt 2" in a.reasons[0]

    def test_empty_result_is_medium_and_uses_warning_text(self) -> None:
        a = _assess(execution_result=[], empty_result_warning="No rows matched your filters")
        assert a.level == "medium"
        assert a.reasons == ["No rows matched your filters"]

    def test_unretrieved_table_is_medium(self) -> None:
        a = _assess(used_tables=["orders", "payroll"])
        assert a.level == "medium"
        assert "payroll" in a.reasons[0]

    def test_many_assumptions_is_medium(self) -> None:
        assert _assess(assumptions=["a", "b", "c"]).level == "medium"

    def test_not_executed_is_not_penalised(self) -> None:
        assert _assess(execute_requested=False, execution_result=None).level == "high"


class TestCleanAssumptions:
    def test_non_list_dropped(self) -> None:
        assert _clean_assumptions("revenue is sum") == []
        assert _clean_assumptions(None) == []

    def test_filters_non_strings_whitespace_and_caps(self) -> None:
        raw = ["  Revenue  =  `total`  ", 42, "", "b", "c", "d", "e"]
        out = _clean_assumptions(raw)
        assert out[0] == "Revenue = `total`"
        assert len(out) == 4

    def test_long_items_truncated(self) -> None:
        out = _clean_assumptions(["x" * 1000])
        assert len(out[0]) == 240 and out[0].endswith("…")


class TestResultPayload:
    def test_caps_rows(self) -> None:
        rows = [{"i": i} for i in range(MAX_ROWS + 5)]
        text, truncated = build_result_payload(rows)
        assert truncated
        assert text.count('"i"') == MAX_ROWS

    def test_caps_characters(self) -> None:
        rows = [{"blob": "y" * 3000} for _ in range(5)]
        text, truncated = build_result_payload(rows)
        assert truncated
        assert len(text) <= MAX_PAYLOAD_CHARS


class _FakeLLM:
    def __init__(self, content: str = "", raises: bool = False) -> None:
        self.content = content
        self.raises = raises
        self.prompts: list[str] = []

    async def complete(self, system_prompt: str, user_prompt: str, **_: Any) -> LLMResponse:
        self.prompts.append(user_prompt)
        if self.raises:
            raise RuntimeError("provider down")
        return LLMResponse(content=self.content, model="fake", prompt_tokens=1, completion_tokens=1, total_tokens=2)


class TestSummarizer:
    async def test_no_rows_means_no_call_and_no_answer(self) -> None:
        llm = _FakeLLM("should not be used")
        assert await AnswerSummarizer(llm).summarize("q", []) is None  # type: ignore[arg-type]
        assert llm.prompts == []

    async def test_answer_is_normalised(self) -> None:
        llm = _FakeLLM("  Northwind   had the\n highest revenue.  ")
        answer = await AnswerSummarizer(llm).summarize("top customer?", [{"name": "Northwind"}])  # type: ignore[arg-type]
        assert answer == "Northwind had the highest revenue."
        assert "<question>top customer?</question>" in llm.prompts[0]

    async def test_provider_failure_is_not_fatal(self) -> None:
        llm = _FakeLLM(raises=True)
        assert await AnswerSummarizer(llm).summarize("q", [{"a": 1}]) is None  # type: ignore[arg-type]


class TestSampleValues:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("status", True),
            ("platform", True),
            ("region", True),
            ("email", False),
            ("customerEmail", False),
            ("first_name", False),
            ("user_id", False),
            ("shipping_address", False),
        ],
    )
    def test_profileable_names(self, name: str, expected: bool) -> None:
        assert is_profileable(name, "varchar") is expected

    def test_numeric_columns_skipped(self) -> None:
        assert not is_profileable("status_code", "integer")

    async def test_profiles_low_cardinality_and_skips_others(self) -> None:
        schema = SchemaMetadata(
            database_name="d",
            dialect="postgresql",
            tables=[
                TableInfo(
                    name="orders",
                    columns=[
                        ColumnInfo(name="status", data_type="varchar"),
                        ColumnInfo(name="channel", data_type="text"),
                        ColumnInfo(name="blocked", data_type="text"),
                        ColumnInfo(name="email", data_type="text"),
                    ],
                )
            ],
        )

        class FakeClient:
            def __init__(self) -> None:
                self.sql: list[str] = []

            async def execute_sql(self, sql: str) -> list[dict[str, Any]]:
                self.sql.append(sql)
                if '"status"' in sql:
                    return [{"v": "shipped"}, {"v": "cancelled"}]
                if '"channel"' in sql:  # high cardinality → dropped
                    return [{"v": str(i)} for i in range(MAX_DISTINCT + 1)]
                raise RuntimeError("policy blocked")  # e.g. restricted column

        client = FakeClient()
        values = await profile_sample_values(schema, client)  # type: ignore[arg-type]
        assert values == {("orders", "status"): ["cancelled", "shipped"]}
        assert not any('"email"' in s for s in client.sql)  # sensitive name never queried

        chunk = DocBuilder.build_chunks(apply_sample_values(schema, values).tables)[0]
        assert "status (varchar) (values: 'cancelled', 'shipped')" in chunk.content

    async def test_rejects_unsafe_identifiers(self) -> None:
        schema = SchemaMetadata(
            database_name="d",
            dialect="postgresql",
            tables=[TableInfo(name='x"; DROP TABLE y; --', columns=[ColumnInfo(name="status", data_type="text")])],
        )

        class Boom:
            async def execute_sql(self, sql: str) -> list[dict[str, Any]]:
                raise AssertionError("must not run SQL for an unsafe identifier")

        assert await profile_sample_values(schema, Boom()) == {}  # type: ignore[arg-type]
