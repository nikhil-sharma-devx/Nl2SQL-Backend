"""HuggingFace Sentence-Transformers embedding provider — implements IEmbedder."""
import structlog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from nl_to_sql.core.exceptions import EmbeddingError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder

logger = structlog.get_logger(__name__)


class HuggingFaceEmbedder(IEmbedder):
    """Embeds text using HuggingFace's sentence-transformers library.

    SOLID:
      S — Only computes embeddings; does not store or retrieve.
      I — Implements only the IEmbedder interface.

    The model is lazy-loaded on first use to avoid heavy initialization
    at import time.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dimensions: int = 384,
    ) -> None:
        self._model_name = model
        self._dimensions = dimensions
        self._model: "SentenceTransformer | None" = None

    def _get_model(self) -> "SentenceTransformer":
        """Lazy-load the sentence-transformer model."""
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            log = logger.bind(model=self._model_name)
            try:
                log.debug("Loading HuggingFace model")
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                log.error("Failed to load HuggingFace model", error=str(exc))
                raise EmbeddingError(
                    f"Failed to load HuggingFace model '{self._model_name}': {exc}",
                    detail=str(exc),
                ) from exc
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Strips newlines for better embedding quality.
        """
        log = logger.bind(model=self._model_name, count=len(texts))
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        try:
            log.debug("Computing embeddings")
            model = self._get_model()
            embeddings = model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as exc:
            log.error("Embedding error", error=str(exc))
            raise EmbeddingError(
                f"HuggingFace embedding failed: {exc}", detail=str(exc)
            ) from exc
