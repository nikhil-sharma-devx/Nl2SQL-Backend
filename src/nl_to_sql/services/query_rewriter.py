"""Query Rewriter — Uses LLM to clarify ambiguous questions."""
import structlog

from nl_to_sql.core.exceptions import LLMProviderError
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider

logger = structlog.get_logger(__name__)

_REWRITE_PROMPT = """You are a query clarification expert for database systems.

Your task is to rewrite ambiguous natural language questions into clear, specific database queries.

Rules:
1. Make implicit entities explicit (e.g., "sales" → "total sales amount")
2. Clarify time ranges if mentioned (e.g., "recent" → "last 30 days")
3. Specify aggregation if implied (e.g., "customer orders" → "count of orders per customer")
4. Keep the original intent
5. Output ONLY the clarified question, nothing else

Examples:
Q: "Show me sales"
A: "Show me the total sales amount grouped by month for the last 12 months"

Q: "Who are the top customers?"
A: "List the top 10 customers by total purchase amount in descending order"

Q: "What products sold well?"
A: "List the top 20 products by total quantity sold in the last quarter"

Original Question: {question}

Clarified Question:"""


_CORRECTION_PROMPT = """You are a query correction assistant for a natural-language-to-SQL system.

A user previously asked a question that produced a SQL query. They are now giving \
a short CORRECTION — changing a column, table, filter, metric, or value \
(e.g. "no, I meant customer_name", "use OrderDate instead", "not revenue, use profit").

Rewrite the ORIGINAL question into a single, self-contained natural-language \
question that incorporates the correction. Preserve everything else about the \
original intent. Do NOT return SQL. Output ONLY the corrected question, nothing else.

Original question: {previous_question}

Previous SQL:
{previous_sql}

User correction: {correction}

Corrected question:"""


class QueryRewriter:
    """Uses LLM to clarify ambiguous questions before processing.

    Features:
    - Detects ambiguous queries
    - Rewrites with explicit entities and operations
    - Only activates for low-confidence classifications

    SOLID:
      S — Only handles query rewriting
      D — Depends on ILLMProvider abstraction
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        enabled: bool = True,
        temperature: float = 0.3,
    ) -> None:
        self._llm = llm_provider
        self._enabled = enabled
        self._temperature = temperature
        self._logger = logger.bind(component="QueryRewriter")

    async def rewrite(self, question: str) -> str:
        """Rewrite an ambiguous question to make it more specific.

        Args:
            question: The original natural language question.

        Returns:
            Clarified question string, or original if rewriting fails/disabled.
        """
        if not self._enabled:
            return question

        try:
            self._logger.debug("Rewriting query", original=question[:80])

            prompt = _REWRITE_PROMPT.replace("{question}", question)

            response = await self._llm.complete(
                system_prompt="You are a helpful query clarification assistant.",
                user_prompt=prompt,
                temperature=self._temperature,
                max_tokens=200,
            )

            clarified = str(response.content).strip()

            # Validate that the response is reasonable (not too long, not empty)
            if clarified and len(clarified) < 500:
                self._logger.info(
                    "Query rewritten",
                    original=question[:50],
                    clarified=clarified[:80],
                )
                return clarified

            self._logger.debug("Rewriting produced invalid result — using original")
            return question

        except LLMProviderError as exc:
            self._logger.warning(
                "Query rewriting failed — using original",
                error=str(exc),
            )
            return question
        except Exception as exc:
            self._logger.warning(
                "Unexpected error in query rewriting — using original",
                error=str(exc),
            )
            return question

    async def apply_correction(
        self,
        previous_question: str,
        previous_sql: str,
        correction: str,
    ) -> str:
        """Rewrite the previous turn's question by applying a user correction.

        Combines the previous question (and its SQL for grounding) with a short
        correction into a single self-contained question, so the user needn't
        restate the whole ask. LLM-backed, with a deterministic concatenation
        fallback on any failure — mirroring ``rewrite()`` error handling.

        Args:
            previous_question: The original NL question from the prior turn.
            previous_sql: The SQL generated for the prior turn (grounding context).
            correction: The user's new correction message.

        Returns:
            A corrected, self-contained NL question. Never raises.
        """
        fallback = self._deterministic_correction(previous_question, correction)
        if not correction.strip():
            return previous_question

        try:
            self._logger.debug(
                "Applying correction",
                previous=previous_question[:60],
                correction=correction[:60],
            )

            prompt = (
                _CORRECTION_PROMPT
                .replace("{previous_question}", previous_question)
                .replace("{previous_sql}", previous_sql or "(none)")
                .replace("{correction}", correction)
            )

            response = await self._llm.complete(
                system_prompt="You rewrite a question by applying a user's correction.",
                user_prompt=prompt,
                temperature=self._temperature,
                max_tokens=250,
            )

            corrected = str(response.content).strip()

            if corrected and len(corrected) < 1000:
                self._logger.info(
                    "Correction applied",
                    correction=correction[:50],
                    corrected=corrected[:80],
                )
                return corrected

            self._logger.debug("Correction produced invalid result — using fallback")
            return fallback

        except LLMProviderError as exc:
            self._logger.warning(
                "Correction rewrite failed — using deterministic fallback",
                error=str(exc),
            )
            return fallback
        except Exception as exc:
            self._logger.warning(
                "Unexpected error in correction rewrite — using fallback",
                error=str(exc),
            )
            return fallback

    @staticmethod
    def _deterministic_correction(previous_question: str, correction: str) -> str:
        """Fallback: fold the correction into the previous question textually."""
        prev = previous_question.strip()
        corr = correction.strip()
        if not corr:
            return prev
        if not prev:
            return corr
        return f"{prev} (correction: {corr})"

    def is_ambiguous(self, question: str) -> bool:
        """Quick heuristic check for ambiguity.

        Args:
            question: The natural language question.

        Returns:
            True if the question appears ambiguous and needs rewriting.
        """
        question_lower = question.lower()

        # Indicators of ambiguity
        ambiguous_patterns = [
            "show me",
            "tell me",
            "what about",
            "how about",
            "give me",
            "i want",
        ]

        # Very short questions are often ambiguous
        if len(question.split()) < 4:
            return True

        # Questions with vague terms
        vague_terms = ["stuff", "things", "data", "info", "something", "recent", "good", "bad"]
        if any(term in question_lower for term in vague_terms):
            return True

        # Questions starting with ambiguous patterns
        if any(question_lower.startswith(pattern) for pattern in ambiguous_patterns):
            return True

        return False
