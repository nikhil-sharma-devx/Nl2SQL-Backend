"""Google Gemini embedding provider — implements IEmbedder using google-genai SDK."""
import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from google import genai

from nl_to_sql.core.exceptions import EmbeddingError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder

logger = structlog.get_logger(__name__)

_GEMINI_MODEL_DIMS: dict[str, int] = {
    "models/text-embedding-004": 768,
}


class GeminiEmbedder(IEmbedder):  # type: ignore[misc]
    """Embeds text via the Gemini text-embedding-004 API.

    No local model or PyTorch required — all compute runs on Google's servers.
    Free tier: 1 500 requests/day, up to 100 texts per request.

    SOLID:
      S — Only computes embeddings; does not store or retrieve.
      I — Implements only the IEmbedder interface.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004",
        dimensions: int = 768,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = _GEMINI_MODEL_DIMS.get(model, dimensions)
        self._client: "genai.Client | None" = None

    def _get_client(self) -> "genai.Client":
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        log = logger.bind(model=self._model, count=len(texts))
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        try:
            log.debug("Requesting Gemini embeddings")
            client = self._get_client()
            response = await asyncio.to_thread(
                client.models.embed_content,
                model=self._model,
                contents=cleaned,
            )
            return [list(e.values) for e in response.embeddings]
        except Exception as exc:
            log.error("Gemini embedding error", error=str(exc))
            raise EmbeddingError(
                f"Gemini embedding failed: {exc}", detail=str(exc)
            ) from exc
