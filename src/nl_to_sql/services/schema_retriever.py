"""Schema retriever â€” RAG step: embed query â†’ find relevant schema chunks.

NOTE: This module is kept for backward compatibility. New code should prefer
``rag.retrieval.retrieval_chain.RetrievalChain`` which adds BM25 sparse
search, RRF fusion, and cross-encoder reranking on top of vector search.
"""
import structlog

from nl_to_sql.core.exceptions import EmptySchemaError, SchemaRetrievalError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.rag.retrieval.context_builder import ContextBuilder  # noqa: F401
from nl_to_sql.rag.retrieval.retrieval_chain import RetrievalChain  # noqa: F401

logger = structlog.get_logger(__name__)


class SchemaRetriever:
    """Retrieves the most relevant schema chunks for a given NL query.

    This is the R in RAG: embed the user's question and find the schema
    tables/columns that are most relevant to answering it.

    Supports both pure vector search and hybrid search (vector + BM25).

    SOLID:
      S â€” Purely responsible for schema retrieval; not generation.
      D â€” Depends on IEmbedder and IVectorStore abstractions.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        top_k: int = 5,
        use_hybrid_search: bool = False,
        hybrid_alpha: float = 0.5,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._use_hybrid_search = use_hybrid_search
        self._hybrid_alpha = hybrid_alpha

    async def retrieve(self, question: str, top_k: int | None = None) -> list[SchemaChunk]:
        """Retrieve top-k schema chunks most relevant to the question.

        Args:
            question: The natural language query from the user.
            top_k: Per-call override for top_k; uses instance default when None.

        Returns:
            Ordered list of SchemaChunk (most relevant first).

        Raises:
            EmptySchemaError: If the vector store contains no chunks.
            SchemaRetrievalError: On embedding or retrieval errors.
        """
        effective_k = top_k if top_k is not None else self._top_k
        log = logger.bind(question=question[:80], top_k=effective_k)

        count = await self._vector_store.count()
        if count == 0:
            raise EmptySchemaError(
                "Vector store is empty. Run 'make ingest' to load the schema."
            )

        log.debug("Embedding user question")
        try:
            query_embedding = await self._embedder.embed(question)
        except Exception as exc:
            raise SchemaRetrievalError(
                f"Failed to embed question: {exc}", detail=str(exc)
            ) from exc

        log.debug("Searching vector store")
        try:
            # Use hybrid search if enabled and available
            if self._use_hybrid_search and hasattr(self._vector_store, 'hybrid_search'):
                log.debug("Using hybrid search (vector + BM25)")
                chunks = await self._vector_store.hybrid_search(
                    query_text=question,
                    query_embedding=query_embedding,
                    top_k=effective_k,
                    alpha=self._hybrid_alpha,
                )
            else:
                log.debug("Using pure vector search")
                chunks = await self._vector_store.similarity_search(
                    query_embedding=query_embedding,
                    top_k=effective_k,
                )
        except Exception as exc:
            raise SchemaRetrievalError(
                f"Vector store search failed: {exc}", detail=str(exc)
            ) from exc

        log.info("Schema chunks retrieved", count=len(chunks),
                 tables=[c.table_name for c in chunks])
        return chunks  # type: ignore[no-any-return]

    def build_schema_context(self, chunks: list[SchemaChunk]) -> str:
        """Concatenate chunk contents into a single schema context block.

        Args:
            chunks: Retrieved schema chunks.

        Returns:
            A formatted multi-table schema description for LLM injection.
        """
        if not chunks:
            return "No relevant schema context was found."
        sections = "\n\n---\n\n".join(c.to_text() for c in chunks)
        return f"RELEVANT DATABASE SCHEMA:\n\n{sections}"

    async def get_schema_for_tables(
        self,
        table_names: list[str],
    ) -> list[SchemaChunk]:
        """Deterministically fetch exact schema chunks for the given tables.

        Bypasses vector similarity and retrieves chunks directly by table name
        from the underlying vector store. Used in Phase C of the two-phase
        schema grounding pipeline to ensure precise column context.

        Args:
            table_names: List of exact table names (from TableSelectorService).

        Returns:
            List of SchemaChunk objects â€” one per matched table.
        """
        if not table_names:
            return []
        log = logger.bind(tables=table_names)
        log.debug("Fetching exact schema chunks by table name")
        chunks = await self._vector_store.get_chunks_by_table_names(table_names)
        log.info(
            "Exact schema chunks fetched",
            requested=len(table_names),
            found=len(chunks),
        )
        return chunks  # type: ignore[no-any-return]

    async def get_all_table_names(self) -> list[str]:
        """Return all table names currently ingested in the vector store.

        Used to build the authoritative list shown to the LLM table selector
        so it can only choose from tables that actually exist.

        Returns:
            Sorted list of unique table name strings.
        """
        return await self._vector_store.get_all_table_names()  # type: ignore[no-any-return]
