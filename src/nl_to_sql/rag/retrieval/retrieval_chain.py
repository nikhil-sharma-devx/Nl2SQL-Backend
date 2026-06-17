"""Retrieval chain — composes the full online retrieval pipeline.

This is the ONLY file that services/ needs to import from rag/retrieval/.
Composes: embed → vector search + BM25 search (parallel) → rerank → build context.
"""
import asyncio

import structlog

from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.rag.retrieval.context_builder import ContextBuilder
from nl_to_sql.rag.retrieval.query_embedder import QueryEmbedder
from nl_to_sql.rag.retrieval.reranker import Reranker
from nl_to_sql.rag.retrieval.vector_retriever import VectorRetriever
from nl_to_sql.infrastructure.observability.tracing import trace_function, set_span_attribute

logger = structlog.get_logger(__name__)


class RetrievalChain:
    """End-to-end retrieval pipeline for the online query path.

    Pipeline steps:
      1. Embed the user's question.
      2. Run vector search + BM25 search in parallel.
      3. Rerank the merged results.
      4. Build the context string for the LLM prompt.

    This class is the single entry point that chat_service / query_orchestrator
    should import.

    SOLID:
      S — Only orchestrates retrieval; does not generate SQL or call the LLM.
      D — Depends on abstractions (IEmbedder, IVectorStore).
    """

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        bm25_store: object | None = None,
        top_k: int = 5,
        use_hybrid_search: bool = False,
        hybrid_alpha: float = 0.5,
        bm25_enabled: bool = False,
        bm25_top_k: int = 5,
        reranker_enabled: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_k: int = 10,
    ) -> None:
        self._query_embedder = QueryEmbedder(embedder)
        self._vector_retriever = VectorRetriever(
            vector_store=vector_store,
            top_k=top_k,
            use_hybrid_search=use_hybrid_search,
            hybrid_alpha=hybrid_alpha,
        )
        self._context_builder = ContextBuilder()
        self._reranker = Reranker(
            model_name=reranker_model,
            top_k=reranker_top_k,
            enabled=reranker_enabled,
        )

        # BM25 retriever (optional)
        self._bm25_retriever = None
        self._bm25_enabled = bm25_enabled
        if bm25_enabled and bm25_store is not None:
            try:
                from nl_to_sql.rag.retrieval.bm25_retriever import BM25Retriever
                self._bm25_retriever = BM25Retriever(
                    store=bm25_store,
                    top_k=bm25_top_k,
                )
            except Exception as exc:
                logger.warning("BM25 retriever init failed", error=str(exc))

        # Keep reference for direct access
        self._vector_store = vector_store

    @trace_function("retrieval.retrieve")
    async def retrieve(self, question: str) -> list[SchemaChunk]:
        """Run the full retrieval pipeline and return ranked schema chunks.

        Args:
            question: The user's natural-language question.

        Returns:
            Reranked list of SchemaChunk (most relevant first).
        """
        log = logger.bind(question=question[:80])
        set_span_attribute("retrieval.question", question)

        # Step 1: Embed the question
        log.debug("Step 1: Embedding question")
        query_embedding = await self._query_embedder.embed(question)

        # Step 2: Run dense + sparse search in parallel
        log.debug("Step 2: Running parallel retrieval")
        dense_task = self._vector_retriever.retrieve(
            query_embedding=query_embedding,
            query_text=question,
        )

        dense_chunks: list[SchemaChunk]
        sparse_chunks: list[SchemaChunk] | None = None

        if self._bm25_retriever is not None and self._bm25_enabled:
            # Run in parallel
            dense_chunks, sparse_chunks_result = await asyncio.gather(
                dense_task,
                asyncio.to_thread(self._bm25_retriever.retrieve, question),
                return_exceptions=True,
            )
            # Handle potential exceptions from parallel tasks
            if isinstance(dense_chunks, Exception):
                raise dense_chunks
            if isinstance(sparse_chunks_result, Exception):
                log.warning("BM25 search failed", error=str(sparse_chunks_result))
                sparse_chunks = None
            else:
                sparse_chunks = sparse_chunks_result
        else:
            dense_chunks = await dense_task

        # Step 3: Rerank
        log.debug("Step 3: Reranking results")
        reranked = await self._reranker.rerank(
            query=question,
            dense_chunks=dense_chunks,
            sparse_chunks=sparse_chunks,
        )

        log.info(
            "Retrieval chain complete",
            dense_count=len(dense_chunks),
            sparse_count=len(sparse_chunks) if sparse_chunks else 0,
            reranked_count=len(reranked),
            tables=[c.table_name for c in reranked],
        )
        set_span_attribute("retrieval.results_count", len(reranked))
        return reranked

    def build_context(self, chunks: list[SchemaChunk]) -> str:
        """Format retrieved chunks into LLM context string.

        Args:
            chunks: Reranked schema chunks.

        Returns:
            Formatted schema context string.
        """
        return self._context_builder.build(chunks)

    async def get_schema_for_tables(
        self,
        table_names: list[str],
    ) -> list[SchemaChunk]:
        """Fetch exact schema chunks by table name (deterministic).

        Used in Phase C of two-phase schema grounding.
        """
        return await self._vector_retriever.get_schema_for_tables(table_names)

    async def get_all_table_names(self) -> list[str]:
        """Return all table names in the vector store."""
        return await self._vector_retriever.get_all_table_names()
