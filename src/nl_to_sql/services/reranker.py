"""Re-ranker — Uses cross-encoder to re-rank retrieved schema chunks.

NOTE: This module is kept for backward compatibility. New code should prefer
``rag.retrieval.reranker.Reranker`` which adds RRF (Reciprocal Rank Fusion)
for merging dense + sparse results in addition to cross-encoder reranking.
"""
from typing import Any, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.rag.retrieval.reranker import Reranker  # noqa: F401 — re-export

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Re-ranks schema chunks using a cross-encoder model.

    Features:
    - Loads cross-encoder model lazily
    - Re-scores (query, chunk) pairs for more accurate ranking
    - Returns re-ranked chunks sorted by relevance

    SOLID:
      S — Only handles re-ranking logic
      D — Depends on sentence-transformers library
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        enabled: bool = True,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._enabled = enabled
        self._model: "CrossEncoder | None" = None
        self._logger = logger.bind(component="CrossEncoderReranker")

    def _get_model(self) -> "CrossEncoder":
        """Lazy-load the cross-encoder model."""
        from sentence_transformers import CrossEncoder
        if self._model is None:
            try:
                self._logger.debug("Loading cross-encoder model", model=self._model_name)
                self._model = CrossEncoder(self._model_name)
                self._logger.info("Cross-encoder model loaded")
            except Exception as exc:
                self._logger.error(
                    "Failed to load cross-encoder model",
                    error=str(exc),
                )
                raise
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: list[SchemaChunk],
    ) -> list[SchemaChunk]:
        """Re-rank schema chunks by relevance to the query.

        Args:
            query: The user's natural language question.
            chunks: Retrieved schema chunks from vector search.

        Returns:
            Re-ranked list of SchemaChunk (most relevant first).
        """
        if not self._enabled or not chunks:
            return chunks

        try:
            model = self._get_model()

            # Create (query, chunk) pairs for scoring
            pairs = [(query, chunk.content) for chunk in chunks]

            # Score all pairs
            scores = model.predict(pairs, show_progress_bar=False)

            # Attach scores to chunks
            chunks_with_scores: list[tuple[SchemaChunk, float]] = list(
                zip(chunks, scores)
            )

            # Sort by score descending
            chunks_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top_k chunks
            reranked = [chunk for chunk, score in chunks_with_scores[: self._top_k]]

            self._logger.info(
                "Chunks re-ranked",
                original_count=len(chunks),
                reranked_count=len(reranked),
                top_score=float(scores.max()) if len(scores) > 0 else 0.0,
            )

            return reranked

        except Exception as exc:
            self._logger.warning(
                "Re-ranking failed — returning original order",
                error=str(exc),
            )
            return chunks

    def get_scores(
        self,
        query: str,
        chunks: list[SchemaChunk],
    ) -> list[tuple[SchemaChunk, float]]:
        """Get relevance scores for chunks without re-ranking.

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
            self._logger.warning("Failed to get re-ranking scores", error=str(exc))
            return [(chunk, 0.0) for chunk in chunks]
