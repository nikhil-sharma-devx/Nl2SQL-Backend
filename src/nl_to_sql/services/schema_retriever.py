"""Schema retriever â€” RAG step: embed query â†’ find relevant schema chunks.

NOTE: This module is kept for backward compatibility. New code should prefer
``rag.retrieval.retrieval_chain.RetrievalChain`` which adds BM25 sparse
search, RRF fusion, and cross-encoder reranking on top of vector search.
"""
import asyncio
import time

import structlog

from nl_to_sql.core.exceptions import EmptySchemaError, SchemaRetrievalError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.infrastructure.observability.tracing import set_span_attribute
from nl_to_sql.rag.retrieval.context_builder import (  # noqa: F401
    ContextBuilder,
    dedupe_parent_chunks,
)
from nl_to_sql.rag.retrieval.retrieval_chain import RetrievalChain  # noqa: F401

logger = structlog.get_logger(__name__)

# HyDE (P5): ask the LLM for a hypothetical schema fragment that *would* answer
# the question. Its embedding sits much closer to real schema chunks than the
# question embedding does.
_HYDE_SYSTEM = """\
You generate a short, hypothetical database schema fragment that would be used \
to answer the user's question. Output 1-3 lines naming plausible tables and \
columns (e.g. "orders(order_id, customer_id, status, shipped_date, total)"). \
Do NOT answer the question or write SQL. Output only the schema fragment."""


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
        user_id: str | None = None,
        query_expander: object | None = None,
        multi_query_enabled: bool = False,
        multi_query_max: int = 3,
        llm_provider: ILLMProvider | None = None,
        hyde_enabled: bool = False,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._use_hybrid_search = use_hybrid_search
        self._hybrid_alpha = hybrid_alpha
        # When set (per-user isolation), every vector-store read is scoped to
        # this user so retrieval never sees another user's tables.
        self._user_id = user_id
        # P3 — multi-query retrieval (original + synonym expansions).
        self._query_expander = query_expander
        self._multi_query_enabled = multi_query_enabled
        self._multi_query_max = multi_query_max
        # P5 — HyDE (hypothetical document embedding) for the dense query vector.
        self._llm_provider = llm_provider
        self._hyde_enabled = hyde_enabled

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

        count = await self._vector_store.count(user_id=self._user_id)
        if count == 0:
            raise EmptySchemaError(
                "Vector store is empty. Run 'make ingest' to load the schema."
            )

        # Build the list of (query_text, dense_embedding) to search. This is one
        # entry normally, or several when multi-query (P3) is enabled. HyDE (P5)
        # swaps the *original* question's dense embedding for a hypothetical-doc
        # embedding while keeping the original text for sparse/keyword matching.
        log.debug("Embedding query variants")
        _embed_start = time.perf_counter()
        try:
            query_specs = await self._build_query_specs(question, log)
        except Exception as exc:
            raise SchemaRetrievalError(
                f"Failed to embed question: {exc}", detail=str(exc)
            ) from exc
        embed_ms = round((time.perf_counter() - _embed_start) * 1000, 2)

        log.debug("Searching vector store", variants=len(query_specs))
        _search_start = time.perf_counter()
        try:
            results = await asyncio.gather(
                *(self._search_one(text, emb, effective_k) for text, emb in query_specs)
            )
        except Exception as exc:
            raise SchemaRetrievalError(
                f"Vector store search failed: {exc}", detail=str(exc)
            ) from exc
        chunks = self._merge_dedupe(results)
        search_ms = round((time.perf_counter() - _search_start) * 1000, 2)

        # Per-stage timers for the live retrieval path (item 2 / observability):
        # mirrors the instrumentation in rag.retrieval.RetrievalChain so both
        # retrieval implementations report comparable span/log timings.
        set_span_attribute("retrieval.embed_ms", embed_ms)
        set_span_attribute("retrieval.search_ms", search_ms)
        set_span_attribute("retrieval.hybrid", bool(self._use_hybrid_search))
        set_span_attribute("retrieval.query_variants", len(query_specs))
        log.info("Schema chunks retrieved", count=len(chunks),
                 embed_ms=embed_ms, search_ms=search_ms, variants=len(query_specs),
                 tables=[c.table_name for c in chunks])
        return chunks

    async def _build_query_specs(
        self, question: str, log: "structlog.BoundLogger"
    ) -> list[tuple[str, list[float]]]:
        """Return (query_text, dense_embedding) pairs to search (P3 + P5)."""
        if self._hyde_enabled and self._llm_provider is not None:
            hyde_text = await self._generate_hyde(question, log)
            base_embedding = await self._embedder.embed(hyde_text or question)
        else:
            base_embedding = await self._embedder.embed(question)
        specs: list[tuple[str, list[float]]] = [(question, base_embedding)]

        if self._multi_query_enabled and self._query_expander is not None:
            try:
                variants = self._query_expander.expand(question)  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning("Query expansion failed — using original only", error=str(exc))
                variants = []
            original_norm = question.strip().lower()
            extras = [
                v for v in variants
                if v and v.strip().lower() != original_norm
            ][: self._multi_query_max]
            if extras:
                extra_embeddings = await self._embedder.embed_batch(extras)
                specs.extend(zip(extras, extra_embeddings, strict=True))
        return specs

    async def _search_one(
        self, query_text: str, query_embedding: list[float], top_k: int
    ) -> list[SchemaChunk]:
        """Run a single dense (or hybrid) search for one query variant."""
        if self._use_hybrid_search and hasattr(self._vector_store, "hybrid_search"):
            return await self._vector_store.hybrid_search(  # type: ignore[no-any-return]
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                alpha=self._hybrid_alpha,
                user_id=self._user_id,
            )
        return await self._vector_store.similarity_search(  # type: ignore[no-any-return]
            query_embedding=query_embedding,
            top_k=top_k,
            user_id=self._user_id,
        )

    @staticmethod
    def _merge_dedupe(results: list[list[SchemaChunk]]) -> list[SchemaChunk]:
        """Union results from all query variants, deduped by chunk_id (P3)."""
        seen: set[str] = set()
        merged: list[SchemaChunk] = []
        for chunk_list in results:
            for chunk in chunk_list:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                merged.append(chunk)
        return merged

    async def _generate_hyde(
        self, question: str, log: "structlog.BoundLogger"
    ) -> str | None:
        """Generate a hypothetical schema fragment for HyDE embedding (P5)."""
        try:
            response = await self._llm_provider.complete(  # type: ignore[union-attr]
                system_prompt=_HYDE_SYSTEM,
                user_prompt=question,
                temperature=0.0,
                max_tokens=200,
            )
            text = (response.content or "").strip()
            if text:
                log.debug("HyDE fragment generated", fragment=text[:80])
                return text
        except Exception as exc:
            log.warning("HyDE generation failed — falling back to question", error=str(exc))
        return None

    def build_schema_context(self, chunks: list[SchemaChunk]) -> str:
        """Concatenate chunk contents into a single schema context block.

        Args:
            chunks: Retrieved schema chunks.

        Returns:
            A formatted multi-table schema description for LLM injection.
        """
        if not chunks:
            return "No relevant schema context was found."
        # Collapse column-level child chunks back to their parent table (P4).
        chunks = dedupe_parent_chunks(chunks)
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
        chunks = await self._vector_store.get_chunks_by_table_names(
            table_names, user_id=self._user_id
        )
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
        return await self._vector_store.get_all_table_names(  # type: ignore[no-any-return]
            user_id=self._user_id
        )
