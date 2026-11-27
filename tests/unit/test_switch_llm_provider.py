"""Unit tests for ApplicationContainer.switch_llm_provider (H9).

Regression test: switching the LLM provider/model at runtime only mutated
the runtime-override providers (``_current_llm_provider``/``_current_llm_model``,
read by ``get_current_llm_config``) — it never touched ``container.config()``
or the ``get_settings()`` lru_cache singleton. Any consumer reading
``settings.llm_provider``/``llm_model`` directly (e.g. GET /profile) kept
reporting the pre-switch value forever.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from nl_to_sql.config.container import ApplicationContainer


def _fake_container(container_settings: MagicMock) -> MagicMock:
    container = MagicMock()
    container.config.return_value = container_settings
    container.llm_provider = MagicMock()
    container._current_llm_provider = MagicMock()
    container._current_llm_model = MagicMock()
    return container


class TestSwitchLlmProviderDualSingleton:
    def test_mutates_container_config_settings(self) -> None:
        container_settings = MagicMock(llm_provider="groq", llm_model="llama-3.3-70b-versatile")
        container = _fake_container(container_settings)

        with (
            patch("nl_to_sql.config.container.create_llm_provider", return_value=MagicMock()),
            patch("nl_to_sql.config.container.get_settings", return_value=container_settings),
        ):
            ApplicationContainer.switch_llm_provider(container, "openai", "gpt-6-luna")

        assert container_settings.llm_provider == "openai"
        assert container_settings.llm_model == "gpt-6-luna"

    def test_mutates_live_get_settings_singleton_when_distinct_from_container_config(self) -> None:
        """The two Settings singletons (container.config() vs get_settings())
        can be distinct instances — both must be updated."""
        container_settings = MagicMock(llm_provider="groq", llm_model="llama-3.3-70b-versatile")
        live_settings = MagicMock(llm_provider="groq", llm_model="llama-3.3-70b-versatile")
        container = _fake_container(container_settings)

        with (
            patch("nl_to_sql.config.container.create_llm_provider", return_value=MagicMock()),
            patch("nl_to_sql.config.container.get_settings", return_value=live_settings),
        ):
            ApplicationContainer.switch_llm_provider(container, "anthropic", "claude-sonnet-5-5")

        assert container_settings.llm_provider == "anthropic"
        assert container_settings.llm_model == "claude-sonnet-5-5"
        assert live_settings.llm_provider == "anthropic"
        assert live_settings.llm_model == "claude-sonnet-5-5"

    def test_still_updates_runtime_override_tracking(self) -> None:
        """get_current_llm_config()'s override-aware read path must keep working."""
        container_settings = MagicMock(llm_provider="groq", llm_model="llama-3.3-70b-versatile")
        container = _fake_container(container_settings)

        with (
            patch("nl_to_sql.config.container.create_llm_provider", return_value=MagicMock()),
            patch("nl_to_sql.config.container.get_settings", return_value=container_settings),
        ):
            ApplicationContainer.switch_llm_provider(container, "openai", "gpt-6-luna")

        container._current_llm_provider.override.assert_called_once()
        container._current_llm_model.override.assert_called_once()
        container.llm_provider.override.assert_called_once()
