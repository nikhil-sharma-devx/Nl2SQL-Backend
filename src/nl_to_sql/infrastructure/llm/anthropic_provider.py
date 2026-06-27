"""Anthropic LLM provider — implements ILLMProvider."""
import re
from collections.abc import AsyncIterator
from typing import Any

import structlog

from nl_to_sql.core.exceptions import LLMProviderError, RateLimitError
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.models.sql_result import LLMResponse

logger = structlog.get_logger(__name__)


class AnthropicProvider(ILLMProvider):  # type: ignore[misc]
    """Concrete LLM provider backed by Anthropic (Claude models).

    SOLID:
      L — Drop-in replacement for GroqProvider / OpenAIProvider.
      O — Adding new Claude models requires only config change, not code change.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None
        if api_key:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed — run: pip install anthropic")

    def _check_key(self) -> None:
        if not self._api_key:
            raise LLMProviderError(
                "No API key configured for Anthropic. "
                "Please add your Anthropic API key in Profile → API Keys.",
                detail="anthropic_api_key is empty",
            )
        if self._client is None:
            raise LLMProviderError(
                "Anthropic client not initialised. Install the package: pip install anthropic",
                detail="anthropic package missing",
            )

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Send a messages request to Anthropic."""
        self._check_key()
        assert self._client is not None
        import anthropic

        model = model_override or self._model
        log = logger.bind(model=model, provider="anthropic")
        try:
            log.debug("Sending completion request")
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = response.content[0].text if response.content else ""
            usage = response.usage
            log.debug("Completion received", tokens=usage.input_tokens + usage.output_tokens)
            return LLMResponse(
                content=content,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )
        except anthropic.RateLimitError as exc:
            log.warning("Anthropic rate limit hit", error=str(exc))
            retry_after = None
            match = re.search(r"try again in\s+([\d.]+)", str(exc), re.IGNORECASE)
            if match:
                retry_after = int(float(match.group(1))) + 5
            raise RateLimitError(
                message=f"Anthropic rate limit exceeded. Please try again in {retry_after or 60} seconds.",
                detail=str(exc),
                retry_after=retry_after,
            ) from exc
        except anthropic.APIError as exc:
            log.error("Anthropic API error", error=str(exc))
            raise LLMProviderError(
                f"Anthropic request failed: {exc}", detail=str(exc)
            ) from exc

    async def health_check(self) -> bool:
        """Ping Anthropic with a minimal message."""
        if not self._api_key or self._client is None:
            return False
        try:
            import anthropic
            await self._client.messages.create(
                model=self._model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except anthropic.APIError:
            return False
        except Exception:
            return False

    async def stream_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream a chat-completion response from Anthropic token by token."""
        self._check_key()
        assert self._client is not None
        import anthropic

        log = logger.bind(model=self._model, provider="anthropic", streaming=True)
        try:
            log.debug("Starting streaming completion request")
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
            log.debug("Streaming completion finished")
        except anthropic.RateLimitError as exc:
            log.warning("Anthropic rate limit hit in stream", error=str(exc))
            retry_after = None
            match = re.search(r"try again in\s+([\d.]+)", str(exc), re.IGNORECASE)
            if match:
                retry_after = int(float(match.group(1))) + 5
            raise RateLimitError(
                message=f"Anthropic rate limit exceeded. Please try again in {retry_after or 60} seconds.",
                detail=str(exc),
                retry_after=retry_after,
            ) from exc
        except anthropic.APIError as exc:
            log.error("Anthropic streaming API error", error=str(exc))
            raise LLMProviderError(
                f"Anthropic streaming request failed: {exc}", detail=str(exc)
            ) from exc
