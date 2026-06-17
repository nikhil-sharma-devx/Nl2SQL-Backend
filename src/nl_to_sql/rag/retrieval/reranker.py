"""Reranker — merges dense and sparse results into a single ranked list.

Supports Reciprocal Rank Fusion (RRF) and optional cross-encoder reranking.
"""
from typing import Any, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


class Reranker:
    """Merges and reranks schema chunks from multiple retrieval sources.

    Supports two reranking strategies:
      1. RRF (Reciprocal Rank Fusion) — fast, no model required.
      2. Cross-encoder — more accurate, requires a model.

    SOLID:
      S — Only reranks; does not retrieve or embed.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        enabled: bool = True,
        use_cross_encoder: bool = True,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._enabled = enabled
        self._use_cross_encoder = use_cross_encoder
        self._model: "CrossEncoder | None" = None

    def _get_model(self) -> "CrossEncoder":
        """Lazy-load the cross-encoder model."""
        from sentence_transformers import CrossEncoder
        if self._model is None:
            try:
                logger.debug("Loading cross-encoder model", model=self._model_name)
                self._model = CrossEncoder(self._model_name)
                logger.info("Cross-encoder model loaded")
            except Exception as exc:
                logger.error(
                    "Failed to load cross-encoder model",
                    error=str(exc),
                )
                raise
        return self._model

    async def rerank(
        self,
        query: str,
        dense_chunks: list[SchemaChunk],
        sparse_chunks: list[SchemaChunk] | None = None,
    ) -> list[SchemaChunk]:
        """Merge and rerank chunks from dense and sparse retrieval.

        Args:
            query: The user's natural language question.
            dense_chunks: Results from vector search.
            sparse_chunks: Results from BM25 search (optional).

        Returns:
            Re-ranked list of SchemaChunk (most relevant first).
        """
        if not self._enabled:
            return dense_chunks

        # Step 1: Merge dense and sparse results
        if sparse_chunks:
            merged = self._rrf_merge(dense_chunks, sparse_chunks)
        else:
            merged = dense_chunks

        if not merged:
            return merged

        # Step 2: Optionally re-score with cross-encoder
        if self._use_cross_encoder:
            try:
                return self._cross_encoder_rerank(query, merged)
            except Exception as exc:
                logger.warning(
                    "Cross-encoder reranking failed — using RRF order",
                    error=str(exc),
                )
                return merged[:self._top_k]

        return merged[:self._top_k]

    def _rrf_merge(
        self,
        dense_chunks: list[SchemaChunk],
        sparse_chunks: list[SchemaChunk],
        k: int = 60,
    ) -> list[SchemaChunk]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank) across all lists where the item appears.

        Args:
            dense_chunks: Dense retrieval results (ordered by relevance).
            sparse_chunks: Sparse retrieval results (ordered by relevance).
            k: RRF constant (default 60, standard in literature).

        Returns:
            Merged list sorted by RRF score.
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, SchemaChunk] = {}

        for rank, chunk in enumerate(dense_chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        merged = [chunk_map[cid] for cid in sorted_ids]
        logger.info(
            "RRF merge complete",
            dense_count=len(dense_chunks),
            sparse_count=len(sparse_chunks),
            merged_count=len(merged),
        )
        return merged

    def _cross_encoder_rerank(
        self,
        query: str,
        chunks: list[SchemaChunk],
    ) -> list[SchemaChunk]:
        """Re-score chunks using a cross-encoder model.

        Args:
            query: The user's question.
            chunks: Merged chunks to rescore.

        Returns:
            Reranked list limited to top_k.
        """
        model = self._get_model()

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        chunks_with_scores = list(zip(chunks, scores))
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)

        reranked = [chunk for chunk, score in chunks_with_scores[:self._top_k]]

        logger.info(
            "Cross-encoder reranking complete",
            input_count=len(chunks),
            output_count=len(reranked),
            top_score=float(scores.max()) if len(scores) > 0 else 0.0,
        )
        return reranked

    def get_scores(
        self,
        query: str,
        chunks: list[SchemaChunk],
    ) -> list[tuple[SchemaChunk, float]]:
        """Get relevance scores for chunks without reranking.

        Args:
            query: The user's question.
            chunks: Retrieved schema chunks.

        Returns:
            List of (chunk, score) tuples.
        """
        if not self._enabled or not chunks:
            return [(chunk, 0.0) for chunk in chunks]

        try:
            model = self._get_model()
            pairs = [(query, chunk.content) for chunk in chunks]
            scores = model.predict(pairs, show_progress_bar=False)
            return list(zip(chunks, scores.tolist()))
        except Exception as exc:
            logger.warning("Failed to get reranking scores", error=str(exc))
            return [(chunk, 0.0) for chunk in chunks]
