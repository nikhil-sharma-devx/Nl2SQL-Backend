"""BM25 indexer — tokenises chunks and builds a BM25 sparse index.

The index is persisted via the BM25Store in infrastructure/ so it can be
loaded by the retrieval pipeline at query time.
"""
import structlog

from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + lowercase tokeniser for BM25.

    Strips punctuation and converts to lowercase for better matching
    of table/column names.
    """
    import re
    # Split on non-alphanumeric characters, keep underscores (common in SQL names)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    return tokens


class BM25Indexer:
    """Builds a BM25 sparse index from schema chunks.

    SOLID:
      S — Only builds the BM25 index; does not perform retrieval.
    """

    def __init__(self, store: "BM25Store") -> None:  # noqa: F821
        from nl_to_sql.infrastructure.bm25_store import BM25Store
        self._store: BM25Store = store

    def build_index(self, chunks: list[SchemaChunk]) -> None:
        """Build the BM25 index from schema chunks and persist it.

        Args:
            chunks: Schema chunks to index.
        """
        if not chunks:
            logger.warning("No chunks to index for BM25")
            return

        log = logger.bind(chunk_count=len(chunks))
        log.info("Building BM25 index")

        # Tokenise all chunks
        corpus = [_tokenise(chunk.content) for chunk in chunks]

        # Build the BM25 index
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi(corpus)
        except ImportError:
            logger.warning(
                "rank-bm25 not installed — BM25 index will not be built. "
                "Install with: pip install rank-bm25"
            )
            return

        # Persist the index along with chunk metadata
        chunk_metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "table_name": chunk.table_name,
                "schema_name": chunk.schema_name,
                "content": chunk.content,
            }
            for chunk in chunks
        ]

        self._store.save(bm25, corpus, chunk_metadata)
        log.info("BM25 index built and persisted", chunk_count=len(chunks))
