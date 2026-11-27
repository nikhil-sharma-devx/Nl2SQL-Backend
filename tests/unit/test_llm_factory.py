"""Unit tests for llm_factory.create_llm_provider (Medium finding).

Regression test: any unrecognized provider string used to silently fall
through to ``GroqProvider`` — a typo'd provider reaching this factory
(bypassing a route's own validation) would silently activate the wrong LLM
while appearing to succeed.
"""
from __future__ import annotations

import pytest

from nl_to_sql.config.factories.llm_factory import create_llm_provider
from nl_to_sql.config.settings import Settings
from nl_to_sql.core.exceptions import ConfigurationError
from nl_to_sql.infrastructure.llm.groq_provider import GroqProvider


def _settings() -> Settings:
    return Settings(
        groq_api_key="k",
        openai_api_key="k",
        anthropic_api_key="k",
        gemini_api_key="k",
        secret_key="x" * 32,
        jwt_secret_key="y" * 32,
    )


class TestCreateLlmProviderUnknownProvider:
    def test_unknown_provider_raises_instead_of_defaulting_to_groq(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            create_llm_provider("groqq", "some-model", _settings())

    def test_empty_string_provider_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            create_llm_provider("", "some-model", _settings())

    def test_known_provider_still_works(self) -> None:
        provider = create_llm_provider("groq", "llama-3.3-70b-versatile", _settings())
        assert isinstance(provider, GroqProvider)
