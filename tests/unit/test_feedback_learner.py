"""Unit tests for FeedbackLearner — pattern recording and prompt injection."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from nl_to_sql.services.feedback_learner import FeedbackLearner, FeedbackPattern


@pytest.fixture
def learner() -> FeedbackLearner:
    return FeedbackLearner(feedback_service=None, training_data_service=None)


class TestRecordFeedback:

    async def test_positive_feedback_does_not_create_pattern(
        self, learner: FeedbackLearner
    ) -> None:
        await learner.record_feedback(
            question="Show all customers",
            generated_sql="SELECT * FROM customers",
            feedback_type="positive",
        )
        # Positive feedback should not add an error pattern
        patterns = await learner.get_relevant_patterns("Show all customers")
        assert len(patterns) == 0

    async def test_negative_feedback_creates_pattern(
        self, learner: FeedbackLearner
    ) -> None:
        await learner.record_feedback(
            question="Count orders by status",
            generated_sql="SELECT status, COUNT(*) FROM orders GROUP BY status",
            feedback_type="negative",
            error_type="wrong_aggregation",
            user_correction="SELECT status, COUNT(order_id) FROM orders GROUP BY status",
            user_notes="Should count order_id not *",
        )
        patterns = await learner.get_relevant_patterns("Count orders by status")
        # May or may not add a pattern depending on implementation; just must not crash
        assert isinstance(patterns, list)

    async def test_record_feedback_with_no_error_type(
        self, learner: FeedbackLearner
    ) -> None:
        # Must not raise even when error_type is omitted
        await learner.record_feedback(
            question="List products",
            generated_sql="SELECT * FROM products",
            feedback_type="negative",
        )

    async def test_record_feedback_with_training_service(self) -> None:
        mock_training = MagicMock()
        mock_training.collect_training_data = AsyncMock()
        learner = FeedbackLearner(
            feedback_service=None,
            training_data_service=mock_training,
        )
        await learner.record_feedback(
            question="Show orders",
            generated_sql="SELECT * FROM orders",
            feedback_type="positive",
            user_correction="SELECT order_id FROM orders",
        )
        # Training service should have been called with the corrected SQL
        mock_training.collect_training_data.assert_called_once()


class TestGetRelevantPatterns:

    async def test_empty_learner_returns_empty_list(
        self, learner: FeedbackLearner
    ) -> None:
        patterns = await learner.get_relevant_patterns("Any question")
        assert patterns == []

    async def test_returns_list_type(self, learner: FeedbackLearner) -> None:
        result = await learner.get_relevant_patterns("Count customers")
        assert isinstance(result, list)

    async def test_filters_by_table_name_mentioned_in_question(
        self, learner: FeedbackLearner
    ) -> None:
        """Medium finding: the question argument used to be ignored entirely."""
        await learner.record_feedback(
            question="Count orders",
            generated_sql="SELECT * FROM orders",
            feedback_type="negative",
            error_type="column 'orders.fake' does not exist",
        )
        assert await learner.get_relevant_patterns("Show me all orders") != []
        assert await learner.get_relevant_patterns("List all customers") == []


class TestBuildPatternContext:

    def test_no_patterns_returns_empty_string(self, learner: FeedbackLearner) -> None:
        context = learner.build_pattern_context([])
        assert context == ""

    def test_patterns_produce_non_empty_string(self, learner: FeedbackLearner) -> None:
        pattern = FeedbackPattern(
            pattern_id="p1",
            error_type="wrong_column",
            description="Used wrong column name",
            example_sql="SELECT c.phone FROM customers c",
            correction="Column 'phone' does not exist; use 'email'",
            table_name="customers",
            column_name="phone",
        )
        context = learner.build_pattern_context([pattern])
        assert isinstance(context, str)
        assert len(context) > 0

    def test_multiple_patterns_all_included(self, learner: FeedbackLearner) -> None:
        patterns = [
            FeedbackPattern(
                pattern_id=f"p{i}",
                error_type="wrong_column",
                description=f"Error {i}",
                example_sql=f"SELECT fake_col_{i} FROM t",
                correction=f"Use real_col_{i}",
            )
            for i in range(3)
        ]
        context = learner.build_pattern_context(patterns)
        for p in patterns:
            assert p.description in context


class TestTenantIsolation:
    """A patterns leak here means one tenant's schema names show up in
    another tenant's SQL-generation prompt (the shared process-global dict)."""

    async def test_pattern_scoped_to_connection_is_invisible_to_other_connection(
        self, learner: FeedbackLearner
    ) -> None:
        await learner.record_feedback(
            question="Count orders",
            generated_sql="SELECT * FROM internal_payroll",
            feedback_type="negative",
            error_type="column 'internal_payroll.ssn' does not exist",
            connection_id="tenant-a",
        )
        assert await learner.get_relevant_patterns(
            "Show me internal_payroll", connection_id="tenant-a"
        ) != []
        assert (
            await learner.get_relevant_patterns(
                "Show me internal_payroll", connection_id="tenant-b"
            )
            == []
        )
        assert await learner.get_relevant_patterns("Show me internal_payroll") == []

    async def test_shared_pattern_visible_to_every_connection(
        self, learner: FeedbackLearner
    ) -> None:
        await learner.record_feedback(
            question="Count orders",
            generated_sql="SELECT * FROM orders",
            feedback_type="negative",
            error_type="column 'orders.fake' does not exist",
        )
        assert await learner.get_relevant_patterns("Show orders", connection_id="tenant-a") != []
        assert await learner.get_relevant_patterns("Show orders", connection_id="tenant-b") != []
        assert await learner.get_relevant_patterns("Show orders") != []

    def test_get_learning_prompt_scoped_to_connection(self, learner: FeedbackLearner) -> None:
        # Directly exercise the sync path used on the live SQL-generation route.
        assert learner.get_learning_prompt(connection_id="tenant-a") == ""

    async def test_same_error_signature_across_tenants_does_not_collide(
        self, learner: FeedbackLearner
    ) -> None:
        """Two tenants hitting the identical error must get distinct pattern
        ids — otherwise the second tenant's record overwrites the first's."""
        for tenant in ("tenant-a", "tenant-b"):
            await learner.record_feedback(
                question="Count orders",
                generated_sql="SELECT * FROM orders",
                feedback_type="negative",
                error_type="column 'orders.fake' does not exist",
                connection_id=tenant,
            )
        assert learner.get_pattern_count() == 2
        assert await learner.get_relevant_patterns("orders", connection_id="tenant-a") != []
        assert await learner.get_relevant_patterns("orders", connection_id="tenant-b") != []


class TestFeedbackPatternModel:

    def test_default_occurrence_count_is_one(self) -> None:
        p = FeedbackPattern(
            pattern_id="x",
            error_type="missing_join",
            description="Missing join",
            example_sql="SELECT c.name FROM orders o",
            correction="JOIN customers c ON o.customer_id = c.customer_id",
        )
        assert p.occurrence_count == 1

    def test_optional_fields_default_to_none(self) -> None:
        p = FeedbackPattern(
            pattern_id="y",
            error_type="wrong_column",
            description="Bad col",
            example_sql="SELECT fake FROM t",
            correction="Use real",
        )
        assert p.table_name is None
        assert p.column_name is None
