"""Groq LLM provider — implements ILLMProvider."""
import re
from collections.abc import AsyncIterator
from typing import Any

import structlog
from groq import APIError as GroqAPIError
from groq import AsyncGroq

from nl_to_sql.core.exceptions import LLMProviderError, RateLimitError
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.models.sql_result import LLMResponse

logger = structlog.get_logger(__name__)


class GroqProvider(ILLMProvider):  # type: ignore[misc]
    """Concrete LLM provider backed by Groq (ultra-fast inference).

    Groq supports Llama 3 and Mixtral models, ideal for low-latency SQL
    generation without OpenAI costs.

    SOLID:
      L — Drop-in replacement for OpenAIProvider.
      O — Adding new Groq models requires only config change, not code change.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama3-70b-8192",
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Send a chat-completion request to Groq."""
        model = model_override or self._model
        log = logger.bind(model=model, provider="groq")
        try:
            log.debug("Sending completion request")
            kwargs = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = await self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
            content = response.choices[0].message.content or ""
            usage = response.usage
            log.debug("Completion received", tokens=usage.total_tokens if usage else 0)
            return LLMResponse(
                content=content,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except GroqAPIError as exc:
            log.error("Groq API error", error=str(exc))

            # Detect rate limit errors
            error_str = str(exc)
            if "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                # Extract retry_after time from error message if available
                retry_after = None
                match = re.search(r'try again in\s+([\d.]+)', error_str)
                if match:
                    retry_after = int(float(match.group(1))) + 5  # Add 5 second buffer

                raise RateLimitError(
                    message=f"Rate limit exceeded. Please try again in {retry_after or 30} seconds.",
                    detail=error_str,
                    retry_after=retry_after,
                ) from exc

            raise LLMProviderError(
                f"Groq request failed: {exc}", detail=str(exc)
            ) from exc

    async def health_check(self) -> bool:
        """Ping Groq by listing models."""
        try:
            await self._client.models.list()
            return True
        except GroqAPIError:
            return False

    async def stream_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream a chat-completion response from Groq token by token."""
        log = logger.bind(model=self._model, provider="groq", streaming=True)
        try:
            log.debug("Starting streaming completion request")
            stream = await self._client.chat.completions.create(
                model=self._model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content

            log.debug("Streaming completion finished")

        except GroqAPIError as exc:
            log.error("Groq streaming API error", error=str(exc))

            error_str = str(exc)
            if "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                retry_after = None
                match = re.search(r'try again in\s+([\d.]+)', error_str)
                if match:
                    retry_after = int(float(match.group(1))) + 5

                raise RateLimitError(
                    message=f"Rate limit exceeded. Please try again in {retry_after or 30} seconds.",
                    detail=error_str,
                    retry_after=retry_after,
                ) from exc

            raise LLMProviderError(
                f"Groq streaming request failed: {exc}", detail=str(exc)
            ) from exc
