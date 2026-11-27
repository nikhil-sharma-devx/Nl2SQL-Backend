"""Unit tests for SQLGeneratorService."""
from unittest.mock import MagicMock

import pytest

from nl_to_sql.services.sql_generator import SQLGeneratorService


class TestSQLGeneratorService:

    @pytest.fixture
    def generator(self, mock_llm_provider: MagicMock) -> SQLGeneratorService:
        return SQLGeneratorService(
            llm_provider=mock_llm_provider,
            dialect="postgresql",
            temperature=0.0,
            max_tokens=512,
        )

    async def test_generate_returns_cleaned_sql(
        self, generator: SQLGeneratorService
    ) -> None:
        result = await generator.generate(
            question="List all customers",
            schema_context="Table: customers\nColumns: customer_id, email",
        )
        assert result.cleaned_sql != ""
        assert result.dialect == "postgresql"

    async def test_generate_strips_markdown_fences(
        self, generator: SQLGeneratorService, mock_llm_provider: MagicMock
    ) -> None:
        from unittest.mock import AsyncMock

        from nl_to_sql.core.models.sql_result import LLMResponse

        mock_llm_provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="```sql\nSELECT * FROM customers;\n```",
                prompt_tokens=10,
                completion_tokens=5,
            )
        )
        result = await generator.generate(
            question="Get all customers",
            schema_context="Table: customers",
        )
        assert "```" not in result.cleaned_sql
        assert "SELECT" in result.cleaned_sql

    async def test_generate_with_error_feedback(
        self, generator: SQLGeneratorService, mock_llm_provider: MagicMock
    ) -> None:
        """Verifies error_feedback is passed through (retry loop support)."""
        await generator.generate(
            question="List customers",
            schema_context="...",
            error_feedback="Syntax error near 'SELCT'",
        )
        call_kwargs = mock_llm_provider.complete.call_args
        user_prompt: str = call_kwargs.kwargs.get("user_prompt", "")
        assert "fix the SQL" in user_prompt.lower() or "Syntax error" in user_prompt

    def test_clean_sql_strips_fence(self) -> None:
        raw = "```sql\nSELECT 1;\n```"
        assert SQLGeneratorService._clean_sql(raw) == "SELECT 1;"

    def test_clean_sql_no_fence(self) -> None:
        raw = "  SELECT 1;  "
        assert SQLGeneratorService._clean_sql(raw) == "SELECT 1;"

    async def test_certified_metrics_substituted_into_prompt(
        self, generator: SQLGeneratorService, mock_llm_provider: MagicMock
    ) -> None:
        await generator.generate(
            question="What is net revenue?",
            schema_context="Table: orders",
            certified_metrics="- Net Revenue: SUM(amount) - SUM(refunds)\n  SQL: SELECT ...",
        )
        system_prompt = mock_llm_provider.complete.call_args.kwargs["system_prompt"]
        assert "CERTIFIED METRICS" in system_prompt
        assert "Net Revenue" in system_prompt
        assert "{certified_metrics_section}" not in system_prompt

    async def test_no_certified_metrics_falls_back_to_default_text(
        self, generator: SQLGeneratorService, mock_llm_provider: MagicMock
    ) -> None:
        await generator.generate(
            question="List customers",
            schema_context="Table: customers",
        )
        system_prompt = mock_llm_provider.complete.call_args.kwargs["system_prompt"]
        assert "No certified metrics defined for this connection." in system_prompt

    async def test_certified_metrics_does_not_affect_custom_instructions_section(
        self, generator: SQLGeneratorService, mock_llm_provider: MagicMock
    ) -> None:
        """Regression guard: certified metrics is additive/parallel to glossary's
        custom_instructions path, never folded into or displacing it."""
        await generator.generate(
            question="List customers",
            schema_context="Table: customers",
            custom_instructions="Business glossary:\n- churn: customers who cancelled",
            certified_metrics="- Net Revenue: SUM(amount)",
        )
        system_prompt = mock_llm_provider.complete.call_args.kwargs["system_prompt"]
        assert "Business glossary" in system_prompt
        assert "churn" in system_prompt
        assert "Net Revenue" in system_prompt


class TestMaxTokensHonored:
    """H3: settings.llm_max_tokens was stored but the generation call
    hardcoded 768/512 regardless — raising the setting had zero effect."""

    async def test_first_attempt_uses_half_of_configured_max_tokens(
        self, mock_llm_provider: MagicMock
    ) -> None:
        generator = SQLGeneratorService(
            llm_provider=mock_llm_provider, dialect="postgresql", max_tokens=2000
        )
        await generator.generate(question="List customers", schema_context="Table: customers")
        assert mock_llm_provider.complete.call_args.kwargs["max_tokens"] == 1000

    async def test_retry_attempt_uses_full_configured_max_tokens(
        self, mock_llm_provider: MagicMock
    ) -> None:
        generator = SQLGeneratorService(
            llm_provider=mock_llm_provider, dialect="postgresql", max_tokens=2000
        )
        await generator.generate(
            question="List customers",
            schema_context="Table: customers",
            error_feedback="Syntax error",
        )
        assert mock_llm_provider.complete.call_args.kwargs["max_tokens"] == 2000

    async def test_default_max_tokens_matches_previous_hardcoded_values(
        self, mock_llm_provider: MagicMock
    ) -> None:
        """With the default max_tokens=1024, behavior matches the old
        hardcoded 512 (first attempt); retry is now 1024, not 768 — the old
        768 was itself an arbitrary hardcode this fix supersedes."""
        generator = SQLGeneratorService(llm_provider=mock_llm_provider, dialect="postgresql")
        await generator.generate(question="List customers", schema_context="Table: customers")
        assert mock_llm_provider.complete.call_args.kwargs["max_tokens"] == 512
