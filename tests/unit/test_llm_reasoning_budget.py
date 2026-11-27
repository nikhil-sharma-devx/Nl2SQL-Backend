"""Reasoning-model token headroom, Gemini JSON mode, retries and overload errors."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nl_to_sql.core.exceptions import LLMProviderError
from nl_to_sql.infrastructure.llm.gemini_provider import GeminiProvider
from nl_to_sql.infrastructure.llm.groq_provider import GroqProvider
from nl_to_sql.infrastructure.llm.reasoning import (
    REASONING_HEADROOM_TOKENS,
    is_groq_reasoning_model,
    with_reasoning_headroom,
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-oss-120b", True),
        ("openai/gpt-oss-20b", True),
        ("qwen/qwen3.8-27b", True),
        ("llama-3.3-70b-versatile", False),
    ],
)
def test_groq_reasoning_model_detection(model: str, expected: bool) -> None:
    assert is_groq_reasoning_model(model) is expected


def test_headroom_added_only_for_reasoning_models() -> None:
    assert with_reasoning_headroom(256, reasoning=True) == 256 + REASONING_HEADROOM_TOKENS
    assert with_reasoning_headroom(256, reasoning=False) == 256


def _groq_with_fake_client() -> tuple[GroqProvider, AsyncMock]:
    provider = GroqProvider(api_key="k", model="openai/gpt-oss-120b")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"tables": []}'))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    create = AsyncMock(return_value=response)
    provider._client = MagicMock()
    provider._client.chat.completions.create = create
    return provider, create


@pytest.mark.asyncio
async def test_groq_small_budget_gets_reasoning_headroom() -> None:
    provider, create = _groq_with_fake_client()

    await provider.complete(system_prompt="s", user_prompt="u", max_tokens=256)

    assert create.await_args.kwargs["max_tokens"] == 256 + REASONING_HEADROOM_TOKENS


@pytest.mark.asyncio
async def test_groq_model_override_decides_headroom() -> None:
    provider, create = _groq_with_fake_client()

    await provider.complete(
        system_prompt="s", user_prompt="u", max_tokens=256, model_override="llama-3.3-70b-versatile"
    )

    assert create.await_args.kwargs["max_tokens"] == 256


def _gemini_with_fake_client(*, error: Exception | None = None) -> tuple[GeminiProvider, AsyncMock]:
    provider = GeminiProvider(api_key="k", model="gemini-3.8-flash")
    if error is not None:
        generate = AsyncMock(side_effect=error)
    else:
        generate = AsyncMock(
            return_value=SimpleNamespace(
                text='{"sql": "SELECT 1"}',
                usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4),
            )
        )
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = generate
    return provider, generate


def test_gemini_client_retries_transient_server_errors() -> None:
    provider = GeminiProvider(api_key="k")
    retry = provider._client._api_client._http_options.retry_options

    assert retry.attempts == 3
    assert 503 in retry.http_status_codes
    assert 429 not in retry.http_status_codes  # rate limits surface with retry-after


@pytest.mark.asyncio
async def test_gemini_json_mode_and_headroom() -> None:
    provider, generate = _gemini_with_fake_client()

    await provider.complete(
        system_prompt="s", user_prompt="u", max_tokens=512, response_format={"type": "json_object"}
    )

    config = generate.await_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.max_output_tokens == 512 + REASONING_HEADROOM_TOKENS


@pytest.mark.asyncio
async def test_gemini_plain_text_when_no_response_format() -> None:
    provider, generate = _gemini_with_fake_client()

    await provider.complete(system_prompt="s", user_prompt="u", max_tokens=200)

    assert generate.await_args.kwargs["config"].response_mime_type is None


@pytest.mark.asyncio
async def test_gemini_overload_gives_actionable_message() -> None:
    provider, _ = _gemini_with_fake_client(
        error=RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.complete(system_prompt="s", user_prompt="u")

    assert "overloaded" in exc_info.value.message
    assert "gemini-3.8-flash" in exc_info.value.message
