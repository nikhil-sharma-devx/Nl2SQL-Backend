"""Unit tests for the CorrectionDetector (correction-phrasing recognition)."""
from __future__ import annotations

import pytest

from nl_to_sql.services.correction_detector import CorrectionDetector, CorrectionSignal


@pytest.fixture
def detector() -> CorrectionDetector:
    return CorrectionDetector()


class TestPositivePhrasings:
    @pytest.mark.parametrize(
        ("message", "expected_term"),
        [
            ("no, I meant customer_name", "customer_name"),
            ("I meant sales_amount", "sales_amount"),
            ("use OrderDate instead", "OrderDate"),
            ("actually use sales_amount", "sales_amount"),
            ("not revenue, use profit", "profit"),
            ("change country to region", "region"),
            ("no, use customer_id", "customer_id"),
            ("rather use gross_margin", "gross_margin"),
            ("it should be total_price", "total_price"),
            ("use the created_at column", "created_at column"),
            ("instead of email, use phone", "phone"),
        ],
    )
    def test_detects_correction_and_extracts_term(
        self, detector: CorrectionDetector, message: str, expected_term: str
    ) -> None:
        result = detector.detect(message)
        assert result.is_correction is True
        assert result.target_term == expected_term


class TestNegativePhrasings:
    @pytest.mark.parametrize(
        "message",
        [
            "Show me the top 5 customers by total order value",
            "How many orders were placed last month?",
            "What is the average revenue per region?",
            "List all products in the electronics category",
            "Count the number of active users",
            "",
            "   ",
        ],
    )
    def test_plain_questions_are_not_corrections(
        self, detector: CorrectionDetector, message: str
    ) -> None:
        result = detector.detect(message)
        assert result.is_correction is False
        assert result.target_term is None


class TestSignalShape:
    def test_returns_correction_signal(self, detector: CorrectionDetector) -> None:
        result = detector.detect("use total instead")
        assert isinstance(result, CorrectionSignal)

    def test_strips_quotes_and_trailing_punctuation(
        self, detector: CorrectionDetector
    ) -> None:
        result = detector.detect('I meant "customer_name".')
        assert result.is_correction is True
        assert result.target_term == "customer_name"
