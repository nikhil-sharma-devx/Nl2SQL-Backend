"""IVectorStore — Abstract interface for vector store backends."""
from abc import ABC, abstractmethod

from nl_to_sql.core.models.schema import SchemaChunk


class IVectorStore(ABC):
    """Contract for storing and retrieving schema chunks by vector similarity.

    SOLID: Open/Closed — new backends (Pinecone, Weaviate) added without
           modifying callers.
    """

    @abstractmethod
    async def upsert(self, chunks: list[SchemaChunk], user_id: str | None = None) -> None:
        """Store schema chunks in the vector store.

        Args:
            chunks: Schema chunks with pre-computed embeddings.
            user_id: When provided, tag each point with this owner so reads can
                be filtered per user (per-user isolation). ``None`` writes
                un-tagged points (shared-collection behaviour).
        """
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Retrieve the top-k most similar schema chunks.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            user_id: When provided, restrict results to this user's chunks.

        Returns:
            Ordered list of SchemaChunk (most relevant first).
        """
        ...

    @abstractmethod
    async def delete_collection(self) -> None:
        """Remove all chunks from the collection (used during re-ingestion)."""
        ...

    @abstractmethod
    async def count(self, user_id: str | None = None) -> int:
        """Return how many chunks are currently stored (optionally per user)."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the store is reachable."""
        ...

    @abstractmethod
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Hybrid search combining vector similarity + BM25 keyword search.

        Args:
            query_text: The original query text for keyword matching.
            query_embedding: The query vector for semantic search.
            top_k: Number of results to return.
            alpha: Weight for vector search (1-alpha for BM25).
            user_id: When provided, restrict results to this user's chunks.

        Returns:
            Ordered list of SchemaChunk (most relevant first).
        """
        ...

    @abstractmethod
    async def get_chunks_by_table_names(
        self,
        table_names: list[str],
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Fetch schema chunks whose table_name is in the given list.

        Used by the two-phase schema grounding pipeline to deterministically
        retrieve exact column definitions for a known set of tables without
        relying on vector similarity.

        Args:
            table_names: List of table names to fetch (exact, case-sensitive).
            user_id: When provided, restrict results to this user's chunks.

        Returns:
            List of matching SchemaChunk objects (order not guaranteed).
        """
        ...

    @abstractmethod
    async def get_all_table_names(self, user_id: str | None = None) -> list[str]:
        """Return the names of every table currently ingested in the store.

        Used to populate the list shown to the LLM during table selection so
        it can only pick from tables that actually exist.

        Args:
            user_id: When provided, restrict to this user's tables.

        Returns:
            Sorted list of unique table name strings.
        """
        ...
