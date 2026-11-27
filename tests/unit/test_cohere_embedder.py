"""Unit tests for CohereEmbedder — batching, retry, and error-mapping behavior."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nl_to_sql.core.exceptions import EmbeddingError
from nl_to_sql.infrastructure.embeddings.cohere_embedder import CohereEmbedder


def _response(status_code: int, payload: dict[str, Any] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.cohere.com/v2/embed")
    return httpx.Response(status_code, json=payload or {}, request=request)


class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_single_batch_returns_vectors_in_order(self) -> None:
        embedder = CohereEmbedder(api_key="k")
        payload = {"embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]}}
        with patch.object(embedder._client, "post", AsyncMock(return_value=_response(200, payload))) as post:
            result = await embedder.embed_batch(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_splits_into_chunks_of_96(self) -> None:
        embedder = CohereEmbedder(api_key="k")
        texts = [f"text-{i}" for i in range(150)]  # 2 chunks: 96 + 54

        async def fake_post(_url: str, json: dict[str, Any]) -> httpx.Response:
            n = len(json["texts"])
            return _response(200, {"embeddings": {"float": [[0.0] * 4 for _ in range(n)]}})

        with patch.object(embedder._client, "post", AsyncMock(side_effect=fake_post)) as post:
            result = await embedder.embed_batch(texts)
        assert len(result) == 150
        assert post.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_input_makes_no_request(self) -> None:
        embedder = CohereEmbedder(api_key="k")
        with patch.object(embedder._client, "post", AsyncMock()) as post:
            result = await embedder.embed_batch([])
        assert result == []
        post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_embed_delegates_to_batch(self) -> None:
        embedder = CohereEmbedder(api_key="k")
        payload = {"embeddings": {"float": [[0.5, 0.6]]}}
        with patch.object(embedder._client, "post", AsyncMock(return_value=_response(200, payload))):
            result = await embedder.embed("hello")
        assert result == [0.5, 0.6]


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self) -> None:
        embedder = CohereEmbedder(api_key="k")
        payload = {"embeddings": {"float": [[0.1]]}}
        responses = [_response(429), _response(200, payload)]
        with patch.object(embedder._client, "post", AsyncMock(side_effect=responses)) as post, \
                patch("nl_to_sql.infrastructure.embeddings.cohere_embedder.asyncio.sleep", AsyncMock()):
            result = await embedder.embed_batch(["hello"])
        assert result == [[0.1]]
        assert post.await_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries_and_wraps_error(self) -> None:
        embedder = CohereEmbedder(api_key="k")
        with patch.object(embedder._client, "post", AsyncMock(return_value=_response(429))), \
                patch("nl_to_sql.infrastructure.embeddings.cohere_embedder.asyncio.sleep", AsyncMock()):
            with pytest.raises(EmbeddingError, match="Cohere embedding failed"):
                await embedder.embed_batch(["hello"])

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self) -> None:
        embedder = CohereEmbedder(api_key="bad-key")
        with patch.object(embedder._client, "post", AsyncMock(return_value=_response(401))) as post:
            with pytest.raises(EmbeddingError, match="Cohere embedding failed"):
                await embedder.embed_batch(["hello"])
        post.assert_awaited_once()  # no retry on a non-retryable 4xx


class TestDimensions:
    def test_known_model_uses_mapped_dimensions(self) -> None:
        embedder = CohereEmbedder(api_key="k", model="embed-english-light-v3.0", dimensions=9999)
        assert embedder.dimensions == 384

    def test_unknown_model_falls_back_to_explicit_dimensions(self) -> None:
        embedder = CohereEmbedder(api_key="k", model="some-future-model", dimensions=42)
        assert embedder.dimensions == 42
