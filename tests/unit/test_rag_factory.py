"""Unit tests for rag_factory.build_embedder.

Regression test: build_embedder previously ignored settings.embedding_provider
entirely and always returned a GeminiEmbedder — a typo'd or newly-added
provider value would silently keep using Gemini instead of erroring.
"""
from __future__ import annotations

import pytest

from nl_to_sql.config.factories.rag_factory import build_embedder
from nl_to_sql.config.settings import Settings
from nl_to_sql.core.exceptions import ConfigurationError
from nl_to_sql.infrastructure.embeddings.cohere_embedder import CohereEmbedder
from nl_to_sql.infrastructure.embeddings.gemini_embedder import GeminiEmbedder


def _settings(**overrides: object) -> Settings:
    return Settings(
        secret_key="x" * 32,
        jwt_secret_key="y" * 32,
        gemini_api_key="k",
        cohere_api_key="k",
        **overrides,  # type: ignore[arg-type]
    )


class TestBuildEmbedder:
    def test_gemini_provider_returns_gemini_embedder(self) -> None:
        embedder = build_embedder(_settings(embedding_provider="gemini"))
        assert isinstance(embedder, GeminiEmbedder)

    def test_cohere_provider_returns_cohere_embedder(self) -> None:
        embedder = build_embedder(_settings(embedding_provider="cohere"))
        assert isinstance(embedder, CohereEmbedder)
        assert embedder.dimensions == 1024

    def test_unknown_provider_raises(self) -> None:
        # Settings' Literal type blocks constructing this directly — assign
        # after construction to simulate a value that reached the factory
        # through some other path (e.g. a future caller with its own string).
        settings = _settings(embedding_provider="gemini")
        settings.embedding_provider = "geminii"  # type: ignore[assignment]
        with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
            build_embedder(settings)
