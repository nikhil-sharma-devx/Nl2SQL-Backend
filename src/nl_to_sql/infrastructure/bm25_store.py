"""BM25 store — handles loading and saving the serialised BM25 index.

The ingestion pipeline writes here; the retrieval pipeline reads from here
on startup. Uses pickle serialisation for the BM25 model.
"""
import os
import pickle
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BM25Store:
    """Persistent storage for the BM25 sparse index.

    SOLID:
      S — Only handles serialisation/deserialisation of the BM25 index.
    """

    def __init__(self, index_path: str = "./data/bm25_index.pkl") -> None:
        self._index_path = index_path
        self._cached_data: tuple[Any, ...] | None = None

    def save(
        self,
        bm25_index: Any,
        corpus: list[list[str]],
        chunk_metadata: list[dict[str, Any]],
    ) -> None:
        """Persist the BM25 index, corpus, and metadata to disk.

        Args:
            bm25_index: The BM25Okapi instance.
            corpus: Tokenised corpus used to build the index.
            chunk_metadata: Metadata for each chunk (chunk_id, table_name, etc.).
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)

        data = {
            "bm25": bm25_index,
            "corpus": corpus,
            "chunk_metadata": chunk_metadata,
        }

        with open(self._index_path, "wb") as f:
            pickle.dump(data, f)

        # Update cache
        self._cached_data = (bm25_index, corpus, chunk_metadata)

        logger.info(
            "BM25 index saved",
            path=self._index_path,
            chunk_count=len(chunk_metadata),
        )

    def load(self) -> tuple[Any, ...] | None:
        """Load the BM25 index from disk.

        Returns:
            Tuple of (bm25_index, corpus, chunk_metadata) or None if not found.
        """
        # Return cached if available
        if self._cached_data is not None:
            return self._cached_data

        if not os.path.exists(self._index_path):
            logger.debug("BM25 index file not found", path=self._index_path)
            return None

        try:
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301 — index is self-generated, not user input

            self._cached_data = (
                data["bm25"],
                data["corpus"],
                data["chunk_metadata"],
            )
            logger.info(
                "BM25 index loaded from disk",
                path=self._index_path,
                chunk_count=len(data["chunk_metadata"]),
            )
            return self._cached_data
        except Exception as exc:
            logger.warning(
                "Failed to load BM25 index",
                path=self._index_path,
                error=str(exc),
            )
            return None

    def clear_cache(self) -> None:
        """Clear the in-memory cache, forcing a reload from disk on next load()."""
        self._cached_data = None

    def exists(self) -> bool:
        """Check if the BM25 index file exists on disk."""
        return os.path.exists(self._index_path)
