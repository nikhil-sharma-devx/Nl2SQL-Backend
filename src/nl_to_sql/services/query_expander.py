"""Query Expander — Auto-expand queries with synonyms and related terms."""
import structlog

logger = structlog.get_logger(__name__)

# Domain-specific synonym dictionary for common database queries
_SYNONYM_DICT = {
    "revenue": ["revenue", "total sales", "income", "earnings", "turnover"],
    "customer": ["customer", "client", "buyer", "user", "account"],
    "product": ["product", "item", "goods", "merchandise", "sku"],
    "order": ["order", "purchase", "transaction", "sale", "booking"],
    "employee": ["employee", "worker", "staff", "personnel", "user"],
    "department": ["department", "division", "team", "unit", "group"],
    "profit": ["profit", "margin", "earnings", "net income", "gain"],
    "cost": ["cost", "expense", "spending", "expenditure", "outlay"],
    "quantity": ["quantity", "amount", "count", "number", "volume"],
    "date": ["date", "time", "timestamp", "day", "period"],
    "top": ["top", "highest", "best", "leading", "most"],
    "recent": ["recent", "latest", "newest", "last", "current"],
}


class QueryExpander:
    """Expands queries with synonyms and related terms for better retrieval.

    Features:
    - Maintains domain-specific synonym dictionary
    - Expands key terms to improve schema matching
    - Configurable expansion depth

    SOLID:
      S — Only handles query expansion logic
      O — Can be extended with ML-based expansion
    """

    def __init__(self, use_synonyms: bool = True, max_expansions: int = 5) -> None:
        self._use_synonyms = use_synonyms
        self._max_expansions = max_expansions
        self._logger = logger.bind(component="QueryExpander")

    def expand(self, question: str) -> list[str]:
        """Expand a query with synonyms and related terms.

        Args:
            question: The original natural language question.

        Returns:
            List of expanded query variations (including original).
        """
        if not self._use_synonyms:
            return [question]

        expanded_queries = {question}
        question_lower = question.lower()

        # Find matching synonyms
        for term, synonyms in _SYNONYM_DICT.items():
            if term in question_lower:
                # Replace term with each synonym
                for synonym in synonyms:
                    if synonym != term:
                        expanded_query = question_lower.replace(term, synonym)
                        expanded_queries.add(expanded_query)

        # Limit expansions
        result = list(expanded_queries)[: self._max_expansions]
        self._logger.debug(
            "Query expanded",
            original=question[:50],
            expansions=len(result) - 1,
        )

        return result

    def get_expanded_terms(self, question: str) -> dict[str, list[str]]:
        """Get expanded terms for each keyword in the question.

        Args:
            question: The natural language question.

        Returns:
            Dictionary mapping original terms to their synonyms.
        """
        question_lower = question.lower()
        expanded_terms = {}

        for term, synonyms in _SYNONYM_DICT.items():
            if term in question_lower:
                expanded_terms[term] = [s for s in synonyms if s != term]

        return expanded_terms
