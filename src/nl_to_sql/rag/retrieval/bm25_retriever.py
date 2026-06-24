"""BM25 retriever — performs keyword search over the schema corpus.

Handles exact column and table name matches that dense retrieval sometimes
misses. Loads the persisted BM25 index from the BM25Store.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from nl_to_sql.core.models.schema import SchemaChunk

if TYPE_CHECKING:
    from nl_to_sql.infrastructure.bm25_store import BM25Store

logger = structlog.get_logger(__name__)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + lowercase tokeniser (must match ingestion)."""
    import re
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


class BM25Retriever:
    """Retrieves schema chunks via BM25 keyword search.

    SOLID:
      S — Only performs sparse search; does not embed or rerank.
      L — Returns list[SchemaChunk] — same shape as VectorRetriever.
    """

    def __init__(
        self,
        store: BM25Store,
        top_k: int = 5,
    ) -> None:
        self._store: BM25Store = store
        self._top_k = top_k

    def retrieve(self, question: str) -> list[SchemaChunk]:
        """Search the BM25 index for schema chunks matching the question.

        Args:
            question: The user's natural-language question.

        Returns:
            List of SchemaChunk ordered by BM25 relevance score.
            Returns empty list if the index is not loaded.
        """
        index_data = self._store.load()
        if index_data is None:
            logger.warning("BM25 index not loaded — returning empty results")
            return []

        bm25, _corpus, chunk_metadata = index_data

        # Tokenise the query using the same tokeniser as ingestion
        query_tokens = _tokenise(question)
        if not query_tokens:
            return []

        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)

        # Rank by score and take top-K
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:self._top_k]

        chunks: list[SchemaChunk] = []
        for idx in scored_indices:
            if scores[idx] <= 0:
                continue  # Skip zero-score results
            meta = chunk_metadata[idx]
            chunks.append(
                SchemaChunk(
                    chunk_id=meta["chunk_id"],
                    table_name=meta["table_name"],
                    schema_name=meta["schema_name"],
                    content=meta["content"],
                )
            )

        logger.info(
            "BM25 retrieval complete",
            count=len(chunks),
            tables=[c.table_name for c in chunks],
        )
        return chunks
