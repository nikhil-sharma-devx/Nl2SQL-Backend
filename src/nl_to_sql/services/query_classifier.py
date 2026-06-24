"""Lightweight query classifier for detecting greetings and off-topic queries."""
from __future__ import annotations

import re
from typing import ClassVar


class QueryClassifier:
    """Classifies user questions into greeting, off_topic, or database_query.

    Uses fast keyword and regex-based heuristics — no LLM call required.
    This keeps the classification lightweight and suitable for running
    before the expensive RAG pipeline.
    """

    # Common greeting patterns (case-insensitive)
    GREETING_PATTERNS: ClassVar[list[str]] = [
        r"^\s*hi\s*$",
        r"^\s*hello\s*$",
        r"^\s*hey\s*$",
        r"^\s*howdy\s*$",
        r"^\s*sup\s*$",
        r"^\s*what's up\s*$",
        r"^\s*whats up\s*$",
        r"^\s*greetings\s*$",
        r"^\s*good morning\s*$",
        r"^\s*good afternoon\s*$",
        r"^\s*good evening\s*$",
        r"^\s*good day\s*$",
        r"^\s*hi there\s*$",
        r"^\s*hello there\s*$",
        r"^\s*hey there\s*$",
        r"^\s*yo\s*$",
        r"^\s*hiya\s*$",
        r"^\s*hi assistant\s*$",
        r"^\s*hello assistant\s*$",
        r"^\s*who are you\s*$",
        r"^\s*what can you do\s*$",
        r"^\s*help\s*$",
    ]

    # Strong database-related keywords that clearly indicate a data question
    # These are words that are very unlikely to appear in casual conversation
    STRONG_DB_KEYWORDS: ClassVar[list[str]] = [
        # SQL operations (strong signal)
        "select", "from", "where", "join", "inner", "outer", "left", "right",
        "group", "having", "limit", "offset", "distinct",
        "insert", "update", "delete", "create", "drop", "alter",
        "table", "tables", "column", "columns", "row", "rows",
        "database", "db", "schema", "query", "queries", "sql",
        # Common query verbs (strong signal)
        "show", "list", "find", "get", "display", "retrieve", "fetch",
        "search", "look up", "lookup",
        # Aggregations (strong signal)
        "count", "sum", "avg", "average", "min", "max", "total",
        "how many", "number of", "amount of", "quantity",
        "group by", "order by", "sorted by", "arranged by",
        "per", "each", "every",
        # Comparisons (strong signal)
        "greater than", "less than", "more than", "fewer than",
        "above", "below", "over", "under", "between", "equal to",
        "highest", "lowest", "largest", "smallest", "most", "least",
        "top", "bottom", "first", "latest", "recent", "oldest",
        # Business/data entities (strong signal)
        "customer", "customers", "client", "clients", "user", "users",
        "product", "products", "item", "items", "order", "orders",
        "sale", "sales", "revenue", "profit", "income", "price", "prices",
        "cost", "amount", "quantity", "inventory", "stock",
        "employee", "employees", "staff", "worker", "workers",
        "department", "departments", "category", "categories",
        "region", "regions", "location", "locations", "city", "cities",
        "country", "countries", "state", "states",
        "transaction", "transactions", "payment", "payments",
        "invoice", "invoices", "bill", "bills",
        "account", "accounts", "id", "identifier", "code",
    ]

    # Context keywords that support a data question but aren't strong on their own
    CONTEXT_KEYWORDS: ClassVar[list[str]] = [
        # Time-related (weak signal on their own)
        "date", "dates", "day", "days", "week", "weeks", "month", "months",
        "year", "years", "today", "yesterday", "tomorrow",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        # Ordering (weak signal on their own)
        "order", "by", "last",
    ]

    GREETING_RESPONSE = (
        "Hello! I'm your SQL assistant. I can help you convert natural language "
        "questions into SQL queries. Try asking me something like 'Show me the top 5 "
        "customers by revenue' or 'How many orders were placed last month?'"
    )

    OFF_TOPIC_RESPONSE = (
        "I can only help with database-related questions. Please ask me a question "
        "about your data, such as 'List all products with price above 100' or "
        "'What is the total revenue by category?'"
    )

    def __init__(self) -> None:
        """Initialize the classifier with compiled regex patterns."""
        self._greeting_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in self.GREETING_PATTERNS]
        self._strong_db_keywords_set = set(self.STRONG_DB_KEYWORDS)
        self._context_keywords_set = set(self.CONTEXT_KEYWORDS)

    def classify(self, question: str) -> str:
        """Classify a user question.

        Args:
            question: The natural language question from the user.

        Returns:
            One of: "greeting", "off_topic", "database_query"
        """
        stripped = question.strip()

        # Check for greetings first
        if self._is_greeting(stripped):
            return "greeting"

        # Check if it's a database-related question
        if self._is_database_query(stripped):
            return "database_query"

        # If neither greeting nor database query, it's off-topic
        return "off_topic"

    def _is_greeting(self, question: str) -> bool:
        """Check if the question is a greeting.

        Args:
            question: The stripped question string.

        Returns:
            True if the question matches a greeting pattern.
        """
        for regex in self._greeting_regexes:
            if regex.match(question):
                return True
        return False

    def _is_database_query(self, question: str) -> bool:
        """Check if the question appears to be database-related.

        Uses keyword matching and simple heuristics to detect data questions.
        A question is considered a database query if:
        1. It contains strong database keywords (SQL terms, business entities, etc.)
        2. It contains BOTH context keywords AND follows a data question pattern

        Args:
            question: The stripped question string.

        Returns:
            True if the question appears to be about database data.
        """
        lower_question = question.lower()
        words = re.findall(r"\b\w+\b", lower_question)

        # Check for strong database keywords (immediate match)
        for word in words:
            if word in self._strong_db_keywords_set:
                return True

        # Check for multi-word strong keywords
        for keyword in self.STRONG_DB_KEYWORDS:
            if " " in keyword and keyword in lower_question:
                return True

        # Check for context keywords + specific data question patterns
        # Only match patterns that are clearly about data retrieval
        has_context = any(word in self._context_keywords_set for word in words)

        if has_context:
            # These patterns combined with context keywords strongly suggest a DB query
            # "how many" and "how much" are very commonly used for data aggregation queries
            data_question_patterns = [
                "how many ", "how much ",
            ]
            for pattern in data_question_patterns:
                if lower_question.startswith(pattern):
                    return True

        return False

    def get_response_message(self, classification: str) -> str:
        """Get the appropriate response message for a classification.

        Args:
            classification: The classification result ("greeting" or "off_topic").

        Returns:
            The response message to return to the user.

        Raises:
            ValueError: If classification is not "greeting" or "off_topic".
        """
        if classification == "greeting":
            return self.GREETING_RESPONSE
        if classification == "off_topic":
            return self.OFF_TOPIC_RESPONSE
        raise ValueError(f"No response message for classification: {classification}")
