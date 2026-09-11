"""Cohere embedding provider — implements IEmbedder using the REST /v2/embed endpoint.

Chosen as a free, lightweight alternative to Gemini: Cohere trial API keys are
free indefinitely (rate-limited rather than token-capped, unlike most "free
credits" embedding APIs), and this needs no SDK beyond the httpx dependency
this project already ships — no torch/transformers/sentence-transformers.
Embedding computation runs entirely on Cohere's servers.
"""
import asyncio
from typing import Any, cast

import httpx
import structlog

from nl_to_sql.core.exceptions import EmbeddingError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder

logger = structlog.get_logger(__name__)

_API_BASE_URL = "https://api.cohere.com/v2"

# Cohere's embed endpoint accepts many texts per call (unlike Gemini's
# gemini-embedding-001, which accepts only one input per request) — batch
# instead of one-request-per-text, which cuts both request count and the
# number of rate-limit hits for a full schema ingestion.
_MAX_TEXTS_PER_REQUEST = 96
_MAX_CONCURRENT_REQUESTS = 5
_MAX_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_BASE_DELAY_SECONDS = 2.0
_RATE_LIMIT_MAX_DELAY_SECONDS = 60.0

_COHERE_MODEL_DIMS: dict[str, int] = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "embed-v4.0": 1536,
}


def _is_retryable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


class CohereEmbedder(IEmbedder):
    """Embeds text via Cohere's hosted /v2/embed REST endpoint.

    No SDK, no local model, no PyTorch — a single httpx POST per batch of up
    to 96 texts; all compute runs on Cohere's servers.

    SOLID:
      S — Only computes embeddings; does not store or retrieve.
      I — Implements only the IEmbedder interface.
      L — Drop-in replacement for GeminiEmbedder.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        dimensions: int = 1024,
        # "search_document" suits both ingestion (indexing schema chunks) and
        # retrieval reasonably well; Cohere's query/document split is a
        # possible future refinement, not required for correctness here.
        input_type: str = "search_document",
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._dimensions = _COHERE_MODEL_DIMS.get(model, dimensions)
        self._input_type = input_type
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)
        self._client = httpx.AsyncClient(
            base_url=_API_BASE_URL,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        log = logger.bind(model=self._model, count=len(texts))
        if not texts:
            return []
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        chunks = [
            cleaned[i : i + _MAX_TEXTS_PER_REQUEST]
            for i in range(0, len(cleaned), _MAX_TEXTS_PER_REQUEST)
        ]
        try:
            log.debug("Requesting Cohere embeddings", batches=len(chunks))
            results = await asyncio.gather(*(self._embed_chunk(c) for c in chunks))
            return [vector for chunk_result in results for vector in chunk_result]
        except Exception as exc:
            log.error("Cohere embedding error", error=str(exc))
            raise EmbeddingError(
                f"Cohere embedding failed: {exc}", detail=str(exc)
            ) from exc

    async def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        async with self._semaphore:
            body = await self._request_with_retry(texts)
            embeddings = body.get("embeddings")
            vectors = embeddings.get("float") if isinstance(embeddings, dict) else embeddings
            if not vectors:
                raise EmbeddingError("Cohere returned no embeddings", detail=str(body))
            return cast(list[list[float]], vectors)

    async def _request_with_retry(self, texts: list[str]) -> dict[str, Any]:
        delay = _RATE_LIMIT_BASE_DELAY_SECONDS
        for attempt in range(1, _MAX_RATE_LIMIT_RETRIES + 1):
            response = await self._client.post(
                "/embed",
                json={
                    "model": self._model,
                    "texts": texts,
                    "input_type": self._input_type,
                    "embedding_types": ["float"],
                },
            )
            if response.status_code == 200:
                return dict(response.json())
            if attempt == _MAX_RATE_LIMIT_RETRIES or not _is_retryable(response.status_code):
                response.raise_for_status()
            logger.warning(
                "Cohere embedding rate-limited or unavailable, backing off",
                model=self._model,
                status_code=response.status_code,
                attempt=attempt,
                delay=delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RATE_LIMIT_MAX_DELAY_SECONDS)
        raise AssertionError("unreachable")  # loop always returns or raises
