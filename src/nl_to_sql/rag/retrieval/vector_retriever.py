"""Vector retriever â€” performs ANN search in the vector database.

Uses the query embedding to find the top-K most similar schema chunks
via approximate nearest-neighbour (ANN) search with cosine similarity.
"""
import structlog

from nl_to_sql.core.exceptions import EmptySchemaError, SchemaRetrievalError
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


class VectorRetriever:
    """Retrieves schema chunks via dense vector similarity search.

    SOLID:
      S â€” Only performs vector search; does not embed or rerank.
      D â€” Depends on IVectorStore abstraction.
      L â€” Returns list[SchemaChunk] â€” same shape as BM25Retriever.
    """

    def __init__(
        self,
        vector_store: IVectorStore,
        top_k: int = 5,
        use_hybrid_search: bool = False,
        hybrid_alpha: float = 0.5,
    ) -> None:
        self._vector_store = vector_store
        self._top_k = top_k
        self._use_hybrid_search = use_hybrid_search
        self._hybrid_alpha = hybrid_alpha

    async def retrieve(
        self,
        query_embedding: list[float],
        query_text: str = "",
    ) -> list[SchemaChunk]:
        """Search the vector store for the most relevant schema chunks.

        Args:
            query_embedding: Dense vector of the user's question.
            query_text: Original question text (used for hybrid search).

        Returns:
            Ordered list of SchemaChunk (most relevant first).

        Raises:
            EmptySchemaError: If the vector store contains no chunks.
            SchemaRetrievalError: On search failure.
        """
        log = logger.bind(top_k=self._top_k)

        count = await self._vector_store.count()
        if count == 0:
            raise EmptySchemaError(
                "Vector store is empty. Run the ingestion pipeline to load the schema."
            )

        try:
            if (
                self._use_hybrid_search
                and hasattr(self._vector_store, "hybrid_search")
                and query_text
            ):
                log.debug("Using hybrid search (vector + BM25)")
                chunks = await self._vector_store.hybrid_search(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    top_k=self._top_k,
                    alpha=self._hybrid_alpha,
                )
            else:
                log.debug("Using pure vector search")
                chunks = await self._vector_store.similarity_search(
                    query_embedding=query_embedding,
                    top_k=self._top_k,
                )
        except Exception as exc:
            raise SchemaRetrievalError(
                f"Vector store search failed: {exc}", detail=str(exc)
            ) from exc

        log.info(
            "Vector retrieval complete",
            count=len(chunks),
            tables=[c.table_name for c in chunks],
        )
        return chunks  # type: ignore[no-any-return]

    async def get_schema_for_tables(
        self,
        table_names: list[str],
    ) -> list[SchemaChunk]:
        """Deterministically fetch exact schema chunks by table name.

        Bypasses vector similarity. Used in Phase C of two-phase
        schema grounding for precise column context.

        Args:
            table_names: List of exact table names.

        Returns:
            List of SchemaChunk objects.
        """
        if not table_names:
            return []
        chunks = await self._vector_store.get_chunks_by_table_names(table_names)
        logger.info(
            "Exact schema chunks fetched",
            requested=len(table_names),
            found=len(chunks),
        )
        return chunks  # type: ignore[no-any-return]

    async def get_all_table_names(self) -> list[str]:
        """Return all table names in the vector store."""
        return await self._vector_store.get_all_table_names()  # type: ignore[no-any-return]
