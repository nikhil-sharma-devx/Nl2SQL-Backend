"""Embedder — calls the embedding model to convert chunks into dense vectors.

Thin wrapper around the IEmbedder interface that handles batching and
logging for the ingestion pipeline context.
"""
import structlog

from nl_to_sql.core.exceptions import SchemaIngestionError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


class IngestionEmbedder:
    """Embeds schema chunks during the ingestion pipeline.

    SOLID:
      S — Only computes embeddings for ingestion; does not store.
      D — Depends on IEmbedder abstraction.
    """

    def __init__(self, embedder: IEmbedder, batch_size: int = 32) -> None:
        self._embedder = embedder
        self._batch_size = batch_size

    async def embed_chunks(self, chunks: list[SchemaChunk]) -> list[SchemaChunk]:
        """Embed all chunks and attach the embedding vectors to each chunk.

        Args:
            chunks: Schema chunks with content but no embeddings.

        Returns:
            The same chunks with embedding field populated.

        Raises:
            SchemaIngestionError: If embedding fails.
        """
        if not chunks:
            return chunks

        log = logger.bind(chunk_count=len(chunks), batch_size=self._batch_size)
        log.info("Embedding schema chunks")

        texts = [c.content for c in chunks]

        try:
            # Process in batches to respect rate limits and memory
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                batch_embeddings = await self._embedder.embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                log.debug(
                    "Embedding batch complete",
                    batch_start=i,
                    batch_size=len(batch),
                )
        except Exception as exc:
            raise SchemaIngestionError(
                f"Failed to embed schema chunks: {exc}"
            ) from exc

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings, strict=True):
            chunk.embedding = embedding

        log.info("All chunks embedded successfully")
        return chunks
