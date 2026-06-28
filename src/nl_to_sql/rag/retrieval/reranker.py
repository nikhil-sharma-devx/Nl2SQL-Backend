"""Reranker — merges dense and sparse results into a single ranked list.

Supports Reciprocal Rank Fusion (RRF) and optional FlashRank cross-encoder
reranking (ONNX-based, no PyTorch required).

TO SWITCH BACK TO sentence-transformers (needs PyTorch + ~500MB RAM):
  1. pip install sentence-transformers
  2. In _get_model(): comment FlashRank block, uncomment sentence-transformers block
  3. In _cross_encoder_rerank(): comment FlashRank block, uncomment sentence-transformers block
  4. In get_scores(): same swap
  5. Change default model_name to "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from flashrank import Ranker

from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


class Reranker:
    """Merges and reranks schema chunks from multiple retrieval sources.

    Supports two reranking strategies:
      1. RRF (Reciprocal Rank Fusion) — fast, no model required.
      2. FlashRank cross-encoder — more accurate, ONNX-based, no PyTorch.

    SOLID:
      S — Only reranks; does not retrieve or embed.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        # sentence-transformers alternative: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        top_k: int = 10,
        enabled: bool = True,
        use_cross_encoder: bool = True,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._enabled = enabled
        self._use_cross_encoder = use_cross_encoder
        self._model: Ranker | None = None

    def _get_model(self) -> "Ranker":
        if self._model is None:
            try:
                # ── FlashRank (active — ONNX, no PyTorch) ────────────────────
                from flashrank import Ranker
                logger.debug("Loading FlashRank model", model=self._model_name)
                self._model = Ranker(model_name=self._model_name)

                # ── sentence-transformers (commented — needs PyTorch ~500MB) ──
                # from sentence_transformers import CrossEncoder
                # self._model = CrossEncoder(self._model_name)

                logger.info("Reranker model loaded")
            except Exception as exc:
                logger.error("Failed to load reranker model", error=str(exc))
                raise
        return self._model

    async def rerank(
        self,
        query: str,
        dense_chunks: list[SchemaChunk],
        sparse_chunks: list[SchemaChunk] | None = None,
    ) -> list[SchemaChunk]:
        if not self._enabled:
            return dense_chunks

        merged = self._rrf_merge(dense_chunks, sparse_chunks) if sparse_chunks else dense_chunks

        if not merged:
            return merged

        if self._use_cross_encoder:
            try:
                return self._cross_encoder_rerank(query, merged)
            except Exception as exc:
                logger.warning("Cross-encoder reranking failed — using RRF order", error=str(exc))
                return merged[: self._top_k]

        return merged[: self._top_k]

    def _rrf_merge(
        self,
        dense_chunks: list[SchemaChunk],
        sparse_chunks: list[SchemaChunk],
        k: int = 60,
    ) -> list[SchemaChunk]:
        scores: dict[str, float] = {}
        chunk_map: dict[str, SchemaChunk] = {}

        for rank, chunk in enumerate(dense_chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        merged = [chunk_map[cid] for cid in sorted_ids]
        logger.info(
            "RRF merge complete",
            dense_count=len(dense_chunks),
            sparse_count=len(sparse_chunks),
            merged_count=len(merged),
        )
        return merged

    def _cross_encoder_rerank(self, query: str, chunks: list[SchemaChunk]) -> list[SchemaChunk]:
        # ── FlashRank (active) ────────────────────────────────────────────────
        from flashrank import RerankRequest
        model = self._get_model()
        passages = [{"id": i, "text": chunk.content} for i, chunk in enumerate(chunks)]
        results = model.rerank(RerankRequest(query=query, passages=passages))
        reranked = [chunks[r["id"]] for r in results[: self._top_k]]

        # ── sentence-transformers (commented) ─────────────────────────────────
        # model = self._get_model()
        # pairs = [(query, chunk.content) for chunk in chunks]
        # scores = model.predict(pairs, show_progress_bar=False)
        # chunks_with_scores = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        # reranked = [chunk for chunk, _ in chunks_with_scores[: self._top_k]]

        logger.info(
            "Cross-encoder reranking complete",
            input_count=len(chunks),
            output_count=len(reranked),
            top_score=results[0]["score"] if results else 0.0,
        )
        return reranked

    def get_scores(
        self,
        query: str,
        chunks: list[SchemaChunk],
    ) -> list[tuple[SchemaChunk, float]]:
        if not self._enabled or not chunks:
            return [(chunk, 0.0) for chunk in chunks]
        try:
            # ── FlashRank (active) ────────────────────────────────────────────
            from flashrank import RerankRequest
            model = self._get_model()
            passages = [{"id": i, "text": chunk.content} for i, chunk in enumerate(chunks)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
            score_map: dict[int, float] = {r["id"]: r["score"] for r in results}
            return [(chunk, score_map.get(i, 0.0)) for i, chunk in enumerate(chunks)]

            # ── sentence-transformers (commented) ─────────────────────────────
            # model = self._get_model()
            # pairs = [(query, chunk.content) for chunk in chunks]
            # scores = model.predict(pairs, show_progress_bar=False)
            # return list(zip(chunks, scores.tolist()))

        except Exception as exc:
            logger.warning("Failed to get reranking scores", error=str(exc))
            return [(chunk, 0.0) for chunk in chunks]
