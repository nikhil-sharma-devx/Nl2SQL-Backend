"""Question Suggestion Service — Generates related follow-up questions.

After answering a query, this service generates 2-3 suggested follow-up questions
that are contextually related to the current query. This helps users explore the
data more effectively and discover insights they might not have thought to ask.

SOLID:
  S — Only generates question suggestions
  D — Depends on LLM provider for generation
"""
from pydantic import BaseModel, Field

import structlog

from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider

logger = structlog.get_logger(__name__)


class SuggestionRequest(BaseModel):
    """Request for generating follow-up questions."""

    original_question: str = Field(..., description="The user's original question")
    generated_sql: str = Field(..., description="The SQL that was generated")
    retrieved_tables: list[str] = Field(
        default_factory=list,
        description="Tables used in the query",
    )


class SuggestionResponse(BaseModel):
    """Response containing suggested follow-up questions."""

    suggestions: list[str] = Field(
        ...,
        description="List of 2-3 suggested follow-up questions",
        min_items=2,
        max_items=3,
    )


class QuestionSuggestionService:
    """Generates contextually relevant follow-up questions."""

    def __init__(self, llm_provider: ILLMProvider) -> None:
        """Initialize with LLM provider.

        Args:
            llm_provider: LLM provider for generating suggestions.
        """
        self._llm_provider = llm_provider

    async def generate_suggestions(
        self,
        request: SuggestionRequest,
    ) -> SuggestionResponse:
        """Generate 2-3 follow-up questions based on the current query.

        Args:
            request: Suggestion request with context.

        Returns:
            Response with suggested questions.
        """
        system_prompt = (
            "You are a data analyst assistant. Your job is to suggest "
            "relevant follow-up questions that help users explore data deeper.\n\n"
            "Rules:\n"
            "1. Generate exactly 3 questions\n"
            "2. Questions should be natural extensions of the original query\n"
            "3. Questions should be answerable with SQL\n"
            "4. Vary the type of analysis (aggregations, filters, joins, etc.)\n"
            "5. Keep questions concise (10-20 words)\n"
            "6. Return ONLY the questions, one per line, no numbering\n"
        )

        user_prompt = (
            f"Original question: {request.original_question}\n"
            f"Tables used: {', '.join(request.retrieved_tables)}\n"
            f"Generated SQL:\n{request.generated_sql}\n\n"
            "Suggest 3 related follow-up questions:"
        )

        try:
            response = await self._llm_provider.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=256,
            )

            # Parse suggestions from response
            suggestions = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip() and not line.strip().startswith(("#", "-", "*", "1.", "2.", "3."))
            ]

            # Ensure we have 2-3 suggestions
            suggestions = suggestions[:3]
            if len(suggestions) < 2:
                # Fallback: generate generic suggestions based on tables
                suggestions = self._generate_fallback_suggestions(
                    request.original_question,
                    request.retrieved_tables,
                )

            logger.info(
                "Generated question suggestions",
                count=len(suggestions),
                suggestions=suggestions,
            )

            return SuggestionResponse(suggestions=suggestions)

        except Exception as exc:
            logger.warning(
                "Failed to generate suggestions, using fallback",
                error=str(exc),
            )
            # Return fallback suggestions
            return SuggestionResponse(
                suggestions=self._generate_fallback_suggestions(
                    request.original_question,
                    request.retrieved_tables,
                )
            )

    def _generate_fallback_suggestions(
        self,
        question: str,
        tables: list[str],
    ) -> list[str]:
        """Generate basic fallback suggestions when LLM fails.

        These are template-based suggestions that are always relevant.
        """
        suggestions = []

        # Suggestion 1: Add aggregation
        if "count" not in question.lower() and "sum" not in question.lower():
            suggestions.append(
                f"Can you show me the total count or sum for these results?"
            )

        # Suggestion 2: Add filtering
        if "where" not in question.lower() and "filter" not in question.lower():
            suggestions.append(
                f"How can I filter these results by a specific condition?"
            )

        # Suggestion 3: Time-based analysis
        if "time" not in question.lower() and "date" not in question.lower():
            suggestions.append(
                f"Can you show me how this changes over time?"
            )

        # If we still don't have enough, add generic ones
        if len(suggestions) < 2:
            suggestions.append(
                f"What are the top results in this category?"
            )
        if len(suggestions) < 3:
            suggestions.append(
                f"Can you compare this across different groups?"
            )

        return suggestions[:3]
