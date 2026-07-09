"""FAISS vector store — implements IVectorStore (in-memory, no persistence)."""
from __future__ import annotations

import numpy as np
import structlog

from nl_to_sql.core.exceptions import VectorStoreError
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


class FAISSVectorStore(IVectorStore):  # type: ignore[misc]
    """FAISS-backed in-memory vector store.

    Useful for environments where ChromaDB is not available or for ultra-fast
    local similarity search. Does NOT persist between restarts.

    SOLID:
      L — Drop-in substitute for ChromaVectorStore.
      O — FAISS index type can be changed via subclass without touching callers.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        try:
            import faiss
            self._faiss = faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        self._dimensions = dimensions
        self._index = self._faiss.IndexFlatIP(dimensions)  # Inner-product (cosine with normalised vecs)
        self._chunks: list[SchemaChunk] = []  # parallel list to index rows
        logger.info("FAISS vector store initialised", dimensions=dimensions)

    async def upsert(self, chunks: list[SchemaChunk], user_id: str | None = None) -> None:
        """Add chunks to the FAISS index (full rebuild on each call).

        Note: per-user isolation (``user_id``) is only implemented for
        QdrantVectorStore; here the parameter is accepted but ignored.
        """
        if not chunks:
            return
        valid = [c for c in chunks if c.embedding is not None]
        if not valid:
            raise VectorStoreError("No embeddings found on chunks to upsert.")
        try:
            # Rebuild index for simplicity (suitable for schema-scale data)
            self._index.reset()
            self._chunks = valid
            matrix = np.array([c.embedding for c in valid], dtype="float32")
            # Normalize for cosine similarity
            self._faiss.normalize_L2(matrix)
            self._index.add(matrix)
            logger.info("FAISS index rebuilt", count=len(valid))
        except Exception as exc:
            raise VectorStoreError(f"FAISS upsert failed: {exc}") from exc

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Run an inner-product (cosine) search."""
        if self._index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype="float32")
        self._faiss.normalize_L2(q)
        k = min(top_k, self._index.ntotal)
        _, indices = self._index.search(q, k)
        return [self._chunks[i] for i in indices[0] if i >= 0]

    async def delete_collection(self) -> None:
        """Clear the in-memory index."""
        self._index.reset()
        self._chunks = []

    async def count(self, user_id: str | None = None) -> int:
        return self._index.ntotal

    async def health_check(self) -> bool:
        return True

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Hybrid search — falls back to vector search for FAISS.

        FAISS doesn't support native BM25, so we use pure vector search.
        For true hybrid search, use QdrantVectorStore.

        Args:
            query_text: The original query text (unused in FAISS).
            query_embedding: The query vector for semantic search.
            top_k: Number of results to return.
            alpha: Weight parameter (unused in fallback).

        Returns:
            Ordered list of SchemaChunk from vector search.
        """
        # Fall back to pure vector search
        return await self.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    async def get_chunks_by_table_names(
        self,
        table_names: list[str],
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Fetch schema chunks whose table_name is in the given list.

        Performs an in-memory scan of the stored chunk list — fast because
        schema chunk counts are always small (one per table).

        Args:
            table_names: Exact table names to fetch.

        Returns:
            List of matching SchemaChunk objects.
        """
        if not table_names:
            return []
        name_set = set(table_names)
        return [c for c in self._chunks if c.table_name in name_set]

    async def get_all_table_names(self, user_id: str | None = None) -> list[str]:
        """Return every unique table name currently stored in the FAISS index.

        Returns:
            Sorted list of unique table name strings.
        """
        return sorted({c.table_name for c in self._chunks if c.table_name})
