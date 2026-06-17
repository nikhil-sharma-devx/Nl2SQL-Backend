"""Semantic cache — caches semantically similar queries using vector similarity."""
import hashlib
import time
from typing import Any

import structlog

from nl_to_sql.core.interfaces.i_cache import ICache
from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


class SemanticCache(ICache):
    """Vector-based semantic cache for query responses.

    Instead of exact matching, this cache uses embedding similarity to find
    semantically similar queries and return their cached responses.

    Features:
    - Embeds queries and stores in vector store
    - Similarity search with configurable threshold
    - TTL support for cache expiration
    - Falls back to exact match cache if semantic cache misses

    SOLID:
      S — Only handles semantic caching logic
      D — Depends on IEmbedder, IVectorStore, and ICache abstractions
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
    ) -> dict[str, Any] | None:
        """Retrieve cached response for semantically similar query.

        Args:
            question: The user's natural language question.
            threshold: Similarity threshold (0-1). Defaults to instance threshold.

        Returns:
            Cached response dict if similarity > threshold, else None.
        """
        effective_threshold = threshold if threshold is not None else self._threshold
        log = self._logger.bind(question=question[:80], threshold=effective_threshold)

        try:
            # Embed the question
            query_embedding = await self._embedder.embed(question)

            # Search for similar cached queries
            similar_chunks = await self._vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=1,
            )

            if not similar_chunks:
                log.debug("No semantically similar queries found")
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
                    "Similarity below threshold — cache miss",
                    similarity=similarity_score,
                )
                return None

            # Check TTL
            cached_at = best_match.metadata.get("cached_at")
            ttl = best_match.metadata.get("ttl", self._default_ttl)

            if cached_at and ttl > 0:
                age = time.time() - cached_at
                if age > ttl:
                    log.debug("Cached response expired — cache miss", age=age)
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
                return cached_response

            return None

        except Exception as exc:
            log.warning("Semantic cache lookup failed — falling back", error=str(exc))
            return None

    async def set_semantic(
        self,
        question: str,
        response: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Store query response in semantic cache.

        Args:
            question: The original natural language question.
            response: The query response to cache.
            ttl: Time-to-live in seconds. Defaults to instance TTL.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        log = self._logger.bind(question=question[:80], ttl=effective_ttl)

        try:
            # Embed the question
            embedding = await self._embedder.embed(question)

            # Create a chunk with the response in metadata
            chunk_id = f"semantic:{hashlib.sha256(question.encode()).hexdigest()}"
            import json
            chunk = SchemaChunk(
                chunk_id=chunk_id,
                table_name="_semantic_cache",
                schema_name="_cache",
                content=question,
                embedding=embedding,
                metadata={
                    "original_question": question,
                    "response": json.dumps(response),
                    "cached_at": time.time(),
                    "ttl": effective_ttl,
                    "type": "semantic_cache",
                },
            )

            # Upsert into vector store
            await self._vector_store.upsert([chunk])
            log.info("Semantic cache entry stored")

        except Exception as exc:
            log.warning("Failed to store in semantic cache — skipping", error=str(exc))
