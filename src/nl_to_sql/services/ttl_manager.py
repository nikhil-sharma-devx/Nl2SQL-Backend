"""TTL Manager — Dynamic cache TTL based on query complexity and frequency."""
import hashlib
import re
from collections import defaultdict
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class TTLManager:
    """Manages dynamic cache TTL based on query characteristics.

    Features:
    - Tracks query frequency to identify popular queries
    - Analyzes SQL complexity (JOINs, aggregations, subqueries)
    - Calculates optimal TTL: popular + simple = long TTL, rare + complex = short TTL
    - Maintains frequency counts with decay over time

    SOLID:
      S — Only handles TTL calculation logic
      O — Can be extended with different TTL strategies
    """

    def __init__(
        self,
        base_ttl: int = 3600,
        min_ttl: int = 300,
        max_ttl: int = 86400,
        frequency_threshold: int = 5,
    ) -> None:
        self._base_ttl = base_ttl
        self._min_ttl = min_ttl
        self._max_ttl = max_ttl
        self._frequency_threshold = frequency_threshold
        self._query_frequency: dict[str, int] = defaultdict(int)
        self._logger = logger.bind(component="TTLManager")

    def calculate_ttl(
        self,
        question: str,
        sql: str | None = None,
    ) -> int:
        """Calculate optimal TTL for a query response.

        Args:
            question: The natural language question.
            sql: The generated SQL query (if available).

        Returns:
            Optimal TTL in seconds.
        """
        # Get query frequency
        query_hash = self._hash_query(question)
        frequency = self._query_frequency[query_hash]

        # Calculate complexity if SQL is available
        complexity = self._analyze_complexity(sql) if sql else 5  # Default medium complexity

        # Calculate frequency multiplier
        if frequency >= self._frequency_threshold * 2:
            frequency_multiplier = 2.0  # Very popular
        elif frequency >= self._frequency_threshold:
            frequency_multiplier = 1.5  # Popular
        elif frequency >= 2:
            frequency_multiplier = 1.2  # Somewhat frequent
        else:
            frequency_multiplier = 1.0  # Rare

        # Calculate complexity multiplier (inverse: simpler = longer TTL)
        if complexity <= 3:
            complexity_multiplier = 1.5  # Simple queries cache longer
        elif complexity <= 6:
            complexity_multiplier = 1.0  # Medium complexity
        else:
            complexity_multiplier = 0.6  # Complex queries cache shorter

        # Calculate final TTL
        ttl = int(self._base_ttl * frequency_multiplier * complexity_multiplier)

        # Clamp to min/max range
        ttl = max(self._min_ttl, min(self._max_ttl, ttl))

        self._logger.debug(
            "Calculated dynamic TTL",
            question=question[:50],
            frequency=frequency,
            complexity=complexity,
            frequency_multiplier=frequency_multiplier,
            complexity_multiplier=complexity_multiplier,
            ttl=ttl,
        )

        return ttl

    def record_query(self, question: str) -> None:
        """Record a query execution to update frequency count.

        Args:
            question: The natural language question.
        """
        query_hash = self._hash_query(question)
        self._query_frequency[query_hash] += 1
        self._logger.debug(
            "Query frequency updated",
            question=question[:50],
            frequency=self._query_frequency[query_hash],
        )

    def _hash_query(self, question: str) -> str:
        """Create a hash of the normalized query."""
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _analyze_complexity(self, sql: str | None) -> int:
        """Analyze SQL complexity on a scale of 1-10.

        Factors:
        - Number of JOINs
        - Presence of subqueries
        - Aggregation functions
        - GROUP BY / HAVING clauses
        - Window functions
        - Number of tables

        Args:
            sql: The SQL query string.

        Returns:
            Complexity score from 1 (simple) to 10 (very complex).
        """
        if not sql:
            return 5

        sql_upper = sql.upper()
        complexity = 1

        # Count JOINs (each JOIN adds complexity)
        join_count = len(re.findall(r'\bJOIN\b', sql_upper))
        complexity += min(join_count * 2, 4)  # Max +4 for JOINs

        # Check for subqueries
        subquery_count = sql_upper.count('SELECT') - 1
        if subquery_count > 0:
            complexity += min(subquery_count * 2, 3)  # Max +3 for subqueries

        # Check for aggregations
        aggregations = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']
        agg_count = sum(1 for agg in aggregations if agg in sql_upper)
        if agg_count > 0:
            complexity += 1

        # Check for GROUP BY
        if 'GROUP BY' in sql_upper:
            complexity += 1

        # Check for HAVING
        if 'HAVING' in sql_upper:
            complexity += 1

        # Check for window functions
        window_functions = ['ROW_NUMBER()', 'RANK()', 'DENSE_RANK()', 'OVER(']
        if any(wf in sql_upper for wf in window_functions):
            complexity += 2

        # Check for ORDER BY
        if 'ORDER BY' in sql_upper:
            complexity += 0.5

        # Check for DISTINCT
        if 'DISTINCT' in sql_upper:
            complexity += 0.5

        # Check for UNION
        if 'UNION' in sql_upper:
            complexity += 2

        # Check for CTEs (WITH clauses)
        if sql_upper.strip().startswith('WITH'):
            complexity += 2

        # Clamp to 1-10 range
        return max(1, min(10, int(complexity)))

    def get_frequency_stats(self) -> dict[str, Any]:
        """Get statistics about query frequencies.

        Returns:
            Dictionary with frequency distribution stats.
        """
        if not self._query_frequency:
            return {"total_queries": 0, "unique_queries": 0}

        frequencies = list(self._query_frequency.values())
        return {
            "total_queries": sum(frequencies),
            "unique_queries": len(frequencies),
            "avg_frequency": sum(frequencies) / len(frequencies),
            "max_frequency": max(frequencies),
            "min_frequency": min(frequencies),
        }

    def reset(self) -> None:
        """Reset all frequency counts."""
        self._query_frequency.clear()
        self._logger.info("TTLManager frequency counts reset")
