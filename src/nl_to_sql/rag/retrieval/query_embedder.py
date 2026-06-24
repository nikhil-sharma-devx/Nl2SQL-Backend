"""Query embedder â€” embeds the user's NL question for retrieval.

Uses the same embedding model as the ingestion pipeline to ensure
compatible vector spaces. Caches results by query hash to avoid
redundant API calls.
"""
import hashlib

import structlog

from nl_to_sql.core.exceptions import SchemaRetrievalError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder

logger = structlog.get_logger(__name__)


class QueryEmbedder:
    """Embeds user questions for vector search during retrieval.

    SOLID:
      S â€” Only embeds queries; does not search or rank.
      D â€” Depends on IEmbedder abstraction.
    """

    def __init__(self, embedder: IEmbedder) -> None:
        self._embedder = embedder
        # Simple in-memory cache keyed by query hash
        self._cache: dict[str, list[float]] = {}

    async def embed(self, question: str) -> list[float]:
        """Embed a user question into a dense vector.

        Results are cached by a SHA-256 hash of the question text.

        Args:
            question: The user's natural-language question.

        Returns:
            Dense vector embedding.

        Raises:
            SchemaRetrievalError: If embedding fails.
        """
        cache_key = hashlib.sha256(question.strip().lower().encode()).hexdigest()

        if cache_key in self._cache:
            logger.debug("Query embedding cache hit", question=question[:60])
            return self._cache[cache_key]

        try:
            embedding = await self._embedder.embed(question)
        except Exception as exc:
            raise SchemaRetrievalError(
                f"Failed to embed question: {exc}", detail=str(exc)
            ) from exc

        self._cache[cache_key] = embedding
        logger.debug("Query embedded and cached", question=question[:60])
        return embedding  # type: ignore[no-any-return]
