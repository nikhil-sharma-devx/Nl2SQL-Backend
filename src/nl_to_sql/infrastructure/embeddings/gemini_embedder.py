"""Google Gemini embedding provider — implements IEmbedder using google-genai SDK."""
import asyncio
from typing import TYPE_CHECKING, Any, cast

import structlog

if TYPE_CHECKING:
    from google import genai

from nl_to_sql.core.exceptions import EmbeddingError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder

logger = structlog.get_logger(__name__)

_GEMINI_MODEL_DIMS: dict[str, int] = {
    "models/text-embedding-004": 768,
    "models/gemini-embedding-001": 3072,
}

# gemini-embedding-001 accepts only a single input per embedContent call
# (unlike the deprecated text-embedding-004, which allowed up to 100).
_MAX_CONCURRENT_REQUESTS = 10
_MAX_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_BASE_DELAY_SECONDS = 2.0
_RATE_LIMIT_MAX_DELAY_SECONDS = 60.0


def _is_rate_limited(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


class GeminiEmbedder(IEmbedder):
    """Embeds text via the Gemini gemini-embedding-001 API.

    No local model or PyTorch required — all compute runs on Google's servers.

    SOLID:
      S — Only computes embeddings; does not store or retrieve.
      I — Implements only the IEmbedder interface.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-embedding-001",
        dimensions: int = 3072,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = _GEMINI_MODEL_DIMS.get(model, dimensions)
        self._client: genai.Client | None = None
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

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
            return await asyncio.gather(*(self._embed_one(t) for t in cleaned))
        except Exception as exc:
            log.error("Gemini embedding error", error=str(exc))
            raise EmbeddingError(
                f"Gemini embedding failed: {exc}", detail=str(exc)
            ) from exc

    async def _embed_one(self, text: str) -> list[float]:
        async with self._semaphore:
            client = self._get_client()
            response = await self._embed_with_retry(client, text)
            if response.embeddings is None:
                raise EmbeddingError(
                    "Gemini returned no embeddings", detail="response.embeddings was None"
                )
            embedding = response.embeddings[0]
            if embedding.values is None:
                raise EmbeddingError(
                    "Gemini returned an empty embedding vector", detail=str(embedding)
                )
            return list(embedding.values)

    async def _embed_with_retry(self, client: "genai.Client", text: str) -> Any:
        delay = _RATE_LIMIT_BASE_DELAY_SECONDS
        for attempt in range(1, _MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return await asyncio.to_thread(
                    client.models.embed_content,
                    model=self._model,
                    contents=cast(Any, text),
                )
            except Exception as exc:
                if attempt == _MAX_RATE_LIMIT_RETRIES or not _is_rate_limited(exc):
                    raise
                logger.warning(
                    "Gemini embedding rate-limited, backing off",
                    model=self._model,
                    attempt=attempt,
                    delay=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RATE_LIMIT_MAX_DELAY_SECONDS)
        raise AssertionError("unreachable")  # loop always returns or raises
