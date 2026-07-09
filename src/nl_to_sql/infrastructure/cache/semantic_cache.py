"""Semantic cache â€” caches semantically similar queries using vector similarity."""
import asyncio
import hashlib
import time
from typing import Any

import structlog

from nl_to_sql.core.interfaces.i_cache import ICache
from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


class SemanticCache(ICache):  # type: ignore[misc]
    """Vector-based semantic cache for query responses.

    Instead of exact matching, this cache uses embedding similarity to find
    semantically similar queries and return their cached responses.

    Features:
    - Embeds queries and stores in vector store
    - Similarity search with configurable threshold
    - TTL support for cache expiration
    - Falls back to exact match cache if semantic cache misses

    SOLID:
      S â€” Only handles semantic caching logic
      D â€” Depends on IEmbedder, IVectorStore, and ICache abstractions
    """

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        exact_cache: ICache,
        threshold: float = 0.95,
        default_ttl: int = 3600,
        collection_name: str = "semantic_cache",
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._exact_cache = exact_cache  # Fallback exact match cache
        self._threshold = threshold
        self._default_ttl = default_ttl
        self._collection_name = collection_name
        self._logger = logger.bind(component="SemanticCache")

    async def get(self, key: str) -> Any | None:
        """Get from exact match cache (delegated)."""
        return await self._exact_cache.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set in exact match cache (delegated)."""
        await self._exact_cache.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        """Delete from exact match cache (delegated)."""
        await self._exact_cache.delete(key)

    async def clear(self) -> None:
        """Clear both caches."""
        await self._exact_cache.clear()
        await self._vector_store.delete_collection()

    async def get_semantic(
        self,
        question: str,
        threshold: float | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve cached response for semantically similar query.

        Args:
            question: The user's natural language question.
            threshold: Similarity threshold (0-1). Defaults to instance threshold.
            user_id: When provided, only returns results cached by this user.

        Returns:
            Cached response dict if similarity > threshold, else None.
        """
        effective_threshold = threshold if threshold is not None else self._threshold
        log = self._logger.bind(question=question[:80], threshold=effective_threshold)

        try:
            # Embed the question
            query_embedding = await self._embedder.embed(question)

            # Retrieve more candidates so we can filter by user_id when needed
            top_k = 10 if user_id else 1
            similar_chunks = await self._vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            if not similar_chunks:
                log.debug("No semantically similar queries found")
                return None

            # Filter by user_id before scoring â€” prevents cross-user cache leakage
            if user_id:
                similar_chunks = [
                    c for c in similar_chunks
                    if c.metadata.get("user_id") == user_id
                ]
                if not similar_chunks:
                    log.debug("No semantic cache entries for this user")
                    return None

            best_match = similar_chunks[0]
            similarity_score = 1.0 - best_match.metadata.get("distance", 1.0)

            log.debug(
                "Best semantic match found",
                similarity=similarity_score,
                cached_question=best_match.metadata.get("original_question", "")[:50],
            )

            # Check if similarity exceeds threshold
            if similarity_score < effective_threshold:
                log.debug(
                    "Similarity below threshold â€” cache miss",
                    similarity=similarity_score,
                )
                return None

            # Check TTL
            cached_at = best_match.metadata.get("cached_at")
            ttl = best_match.metadata.get("ttl", self._default_ttl)

            if cached_at and ttl > 0:
                age = time.time() - cached_at
                if age > ttl:
                    log.debug("Cached response expired â€” cache miss", age=age)
                    return None

            # Return cached response
            cached_response_str = best_match.metadata.get("response")
            if cached_response_str:
                try:
                    import json
                    cached_response = json.loads(cached_response_str) if isinstance(cached_response_str, str) else cached_response_str
                except Exception:
                    cached_response = cached_response_str

                log.info(
                    "Semantic cache hit",
                    similarity=similarity_score,
                    cached_question=best_match.metadata.get("original_question", "")[:50],
                )
                return cached_response  # type: ignore[no-any-return]

            return None

        except Exception as exc:
            log.warning("Semantic cache lookup failed â€” falling back", error=str(exc))
            return None

    async def set_semantic(
        self,
        question: str,
        response: dict[str, Any],
        ttl: int | None = None,
        user_id: str | None = None,
    ) -> None:
        """Store query response in semantic cache (non-blocking â€” write happens in background)."""
        task = asyncio.create_task(self._write_semantic(question, response, ttl, user_id))
        task.add_done_callback(lambda t: None)  # prevent GC

    async def _write_semantic(
        self,
        question: str,
        response: dict[str, Any],
        ttl: int | None = None,
        user_id: str | None = None,
    ) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        log = self._logger.bind(question=question[:80], ttl=effective_ttl)

        try:
            embedding = await self._embedder.embed(question)

            id_namespace = f"{user_id}:" if user_id else ""
            chunk_id = f"semantic:{id_namespace}{hashlib.sha256(question.encode()).hexdigest()}"
            import json
            chunk = SchemaChunk(
                chunk_id=chunk_id,
                table_name="_semantic_cache",
                schema_name="_cache",
                content=question,
                embedding=embedding,
                metadata={
                    "original_question": question,
                    # default=str so datetime/Decimal values in execution_result
                    # (from the target DB) serialize instead of raising and
                    # silently disabling the L2 semantic cache.
                    "response": json.dumps(response, default=str),
                    "cached_at": time.time(),
                    "ttl": effective_ttl,
                    "type": "semantic_cache",
                    "user_id": user_id,
                },
            )

            await self._vector_store.upsert([chunk])
            log.info("Semantic cache entry stored")

        except Exception as exc:
            log.warning("Failed to store in semantic cache â€” skipping", error=str(exc))
