"""Re-ranker — Uses FlashRank (ONNX, no PyTorch) to re-rank retrieved schema chunks.

NOTE: This module is kept for backward compatibility. New code should prefer
``rag.retrieval.reranker.Reranker`` which adds RRF (Reciprocal Rank Fusion)
for merging dense + sparse results in addition to cross-encoder reranking.
"""
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from flashrank import Ranker

from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.rag.retrieval.reranker import Reranker  # noqa: F401 — re-export

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Re-ranks schema chunks using FlashRank (ONNX cross-encoder, no PyTorch).

    SOLID:
      S — Only handles re-ranking logic.
      D — Depends on flashrank library (ONNX-based, memory-light).
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        top_k: int = 10,
        enabled: bool = True,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._enabled = enabled
        self._model: Ranker | None = None
        self._logger = logger.bind(component="CrossEncoderReranker")

    def _get_model(self) -> "Ranker":
        if self._model is None:
            try:
                from flashrank import Ranker
                self._logger.debug("Loading FlashRank model", model=self._model_name)
                self._model = Ranker(model_name=self._model_name)
                self._logger.info("FlashRank model loaded")
            except Exception as exc:
                self._logger.error("Failed to load FlashRank model", error=str(exc))
                raise
        return self._model

    async def rerank(self, query: str, chunks: list[SchemaChunk]) -> list[SchemaChunk]:
        if not self._enabled or not chunks:
            return chunks
        try:
            from flashrank import RerankRequest
            model = self._get_model()
            passages = [{"id": i, "text": chunk.content} for i, chunk in enumerate(chunks)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
            reranked = [chunks[r["id"]] for r in results[: self._top_k]]
            self._logger.info(
                "Chunks re-ranked",
                original_count=len(chunks),
                reranked_count=len(reranked),
                top_score=results[0]["score"] if results else 0.0,
            )
            return reranked
        except Exception as exc:
            self._logger.warning("Re-ranking failed — returning original order", error=str(exc))
            return chunks

    def get_scores(self, query: str, chunks: list[SchemaChunk]) -> list[tuple[SchemaChunk, float]]:
        if not self._enabled or not chunks:
            return [(chunk, 0.0) for chunk in chunks]
        try:
            from flashrank import RerankRequest
            model = self._get_model()
            passages = [{"id": i, "text": chunk.content} for i, chunk in enumerate(chunks)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
            score_map: dict[int, float] = {r["id"]: r["score"] for r in results}
            return [(chunk, score_map.get(i, 0.0)) for i, chunk in enumerate(chunks)]
        except Exception as exc:
            self._logger.warning("Failed to get re-ranking scores", error=str(exc))
            return [(chunk, 0.0) for chunk in chunks]
