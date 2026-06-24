"""SQL Explanation Service — generates plain-English explanations of SQL queries."""
from __future__ import annotations

import structlog

from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider

logger = structlog.get_logger(__name__)


class SQLExplanationService:
    """Generates human-readable explanations for SQL queries.

    SOLID:
      S — Only responsible for explaining SQL queries.
      D — Depends on ILLMProvider interface, not concrete implementation.
    """

    def __init__(self, llm_provider: ILLMProvider) -> None:
        self._llm_provider = llm_provider

    async def explain(self, sql: str) -> str:
        """Generate a plain-English explanation of what the SQL query does.

        Args:
            sql: The SQL query to explain.

        Returns:
            A natural-language explanation of the query.
        """
        log = logger.bind(sql_preview=sql[:100])
        log.info("Generating SQL explanation")

        system_prompt = (
            "You are a SQL expert who explains complex queries in simple, "
            "easy-to-understand language. Your explanations should be clear and concise, "
            "suitable for users who may not have a technical background.\n\n"
            "Explain what the query does step by step:\n"
            "1. What data is being retrieved\n"
            "2. Which tables are involved and how they're connected\n"
            "3. What filters or conditions are applied\n"
            "4. How the results are organized (GROUP BY, ORDER BY, etc.)\n"
            "5. Any limits or aggregations\n\n"
            "Use plain English and avoid technical jargon where possible."
        )

        user_prompt = f"Please explain this SQL query in plain English:\n\n```sql\n{sql}\n```"

        try:
            response = await self._llm_provider.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=512,
            )

            explanation = str(response.content).strip()
            log.info("SQL explanation generated", explanation_length=len(explanation))
            return explanation

        except Exception as exc:
            log.error("Failed to generate SQL explanation", error=str(exc))
            raise Exception(f"Failed to explain SQL: {exc}") from exc
