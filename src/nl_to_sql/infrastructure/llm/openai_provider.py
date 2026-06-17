"""OpenAI LLM provider — implements ILLMProvider."""
import re
from collections.abc import AsyncIterator

import structlog
from openai import AsyncOpenAI
from openai import APIError as OpenAIAPIError
from openai import RateLimitError as OpenAIRateLimitError

from nl_to_sql.core.exceptions import LLMProviderError, RateLimitError
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.models.sql_result import LLMResponse

logger = structlog.get_logger(__name__)


class OpenAIProvider(ILLMProvider):
    """Concrete LLM provider backed by OpenAI (GPT models).

    SOLID:
      L — Drop-in replacement for GroqProvider.
      O — Adding new OpenAI models requires only config change, not code change.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key
        self._client = AsyncOpenAI(api_key=api_key) if api_key else None
        self._model = model

    def _check_key(self) -> None:
        """Raise LLMProviderError if no API key is configured."""
        if not self._api_key:
            raise LLMProviderError(
                "No API key configured for OpenAI. "
                "Please add your OpenAI API key in Profile → API Keys.",
                detail="openai_api_key is empty",
            )

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Send a chat-completion request to OpenAI."""
        self._check_key()
        model = model_override or self._model
        log = logger.bind(model=model, provider="openai")
        try:
            log.debug("Sending completion request")
            kwargs: dict = {
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

            response = await self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            usage = response.usage
            log.debug("Completion received", tokens=usage.total_tokens if usage else 0, model=model)
            return LLMResponse(
                content=content,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except OpenAIRateLimitError as exc:
            log.warning("OpenAI rate limit hit", error=str(exc))
            retry_after = None
            match = re.search(r"try again in\s+([\d.]+)", str(exc), re.IGNORECASE)
            if match:
                retry_after = int(float(match.group(1))) + 5
            raise RateLimitError(
                message=f"OpenAI rate limit exceeded. Please try again in {retry_after or 60} seconds.",
                detail=str(exc),
                retry_after=retry_after,
            ) from exc
        except OpenAIAPIError as exc:
            log.error("OpenAI API error", error=str(exc))
            raise LLMProviderError(
                f"OpenAI request failed: {exc}", detail=str(exc)
            ) from exc

    async def health_check(self) -> bool:
        """Ping OpenAI by listing models."""
        if not self._api_key:
            return False
        try:
            await self._client.models.list()
            return True
        except OpenAIAPIError:
            return False

    async def stream_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream a chat-completion response from OpenAI token by token."""
        self._check_key()
        log = logger.bind(model=self._model, provider="openai", streaming=True)
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
                    yield chunk.choices[0].delta.content

            log.debug("Streaming completion finished")

        except OpenAIRateLimitError as exc:
            log.warning("OpenAI rate limit hit in stream", error=str(exc))
            retry_after = None
            match = re.search(r"try again in\s+([\d.]+)", str(exc), re.IGNORECASE)
            if match:
                retry_after = int(float(match.group(1))) + 5
            raise RateLimitError(
                message=f"OpenAI rate limit exceeded. Please try again in {retry_after or 60} seconds.",
                detail=str(exc),
                retry_after=retry_after,
            ) from exc
        except OpenAIAPIError as exc:
            log.error("OpenAI streaming API error", error=str(exc))
            raise LLMProviderError(
                f"OpenAI streaming request failed: {exc}", detail=str(exc)
            ) from exc
