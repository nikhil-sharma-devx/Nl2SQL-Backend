"""Google Gemini LLM provider — implements ILLMProvider using google-genai SDK."""

import re
from collections.abc import AsyncIterator
from typing import Any

import structlog

from nl_to_sql.core.exceptions import LLMProviderError, RateLimitError
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.models.sql_result import LLMResponse
from nl_to_sql.infrastructure.llm.reasoning import with_reasoning_headroom

logger = structlog.get_logger(__name__)

# Original request + 2 retries for transient server-side failures.
_RETRY_ATTEMPTS = 3
_RETRYABLE_STATUS_CODES = (500, 502, 503, 504)


def _is_overloaded(error_str: str) -> bool:
    """True for Gemini's server-side capacity errors (still failing after retries)."""
    return any(code in error_str for code in ("503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED"))


def _overloaded_message(model: str) -> str:
    return (
        f"Gemini model '{model}' is overloaded or timing out on Google's side right now. "
        "Try again in a minute, or switch to another model in Settings."
    )


class GeminiProvider(ILLMProvider):
    """Concrete LLM provider backed by Google Gemini (google-genai SDK).

    SOLID:
      L — Drop-in replacement for GroqProvider / OpenAIProvider.
      O — Adding new Gemini models requires only config change, not code change.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        timeout_ms: int = 30_000,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None
        if api_key:
            try:
                from google import genai
                from google.genai import types as genai_types

                # Timeout so a hung upstream can't tie up request capacity, plus
                # retries with backoff for Gemini's transient overload errors
                # (503 "model is experiencing high demand", 500, 504). 429 is
                # left out on purpose: it surfaces as RateLimitError with a
                # retry-after the UI shows, rather than being retried blindly.
                self._client = genai.Client(
                    api_key=api_key,
                    http_options=genai_types.HttpOptions(
                        timeout=timeout_ms,
                        retry_options=genai_types.HttpRetryOptions(
                            attempts=_RETRY_ATTEMPTS,
                            initial_delay=1.0,
                            max_delay=8.0,
                            http_status_codes=list(_RETRYABLE_STATUS_CODES),
                        ),
                    ),
                )
            except ImportError:
                logger.warning("google-genai package not installed — run: pip install google-genai")

    def _check_key(self) -> None:
        if not self._api_key:
            raise LLMProviderError(
                "No API key configured for Gemini. "
                "Please add your Gemini API key in Profile → API Keys.",
                detail="gemini_api_key is empty",
            )
        if self._client is None:
            raise LLMProviderError(
                "Gemini client not initialised. Install the package: pip install google-genai",
                detail="google-genai package missing",
            )

    def _build_contents(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        """Build the contents list, prepending system prompt as a user turn if needed."""
        if system_prompt:
            return [
                {"role": "user", "parts": [{"text": f"System: {system_prompt}\n\n{user_prompt}"}]},
            ]
        return [{"role": "user", "parts": [{"text": user_prompt}]}]

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Send a generate_content request to Gemini."""
        self._check_key()
        assert self._client is not None
        from google.genai import types as genai_types

        model_name = model_override or self._model
        log = logger.bind(model=model_name, provider="gemini")
        try:
            log.debug("Sending completion request")
            wants_json = bool(response_format) and response_format.get("type") in (
                "json_object",
                "json_schema",
            )
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                # Gemini's thinking tokens count against max_output_tokens.
                max_output_tokens=with_reasoning_headroom(max_tokens, reasoning=True),
                system_instruction=system_prompt if system_prompt else None,
                # Honour JSON mode like the other providers; it was silently
                # dropped, so callers expecting JSON got prose or fenced JSON.
                response_mime_type="application/json" if wants_json else None,
            )
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config,
            )
            content = response.text or ""
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            log.debug("Completion received", tokens=prompt_tokens + completion_tokens)
            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            error_str = str(exc)
            log.error("Gemini API error", error=error_str)
            if "rate" in error_str.lower() or "quota" in error_str.lower() or "429" in error_str:
                retry_after = None
                match = re.search(r"retry.?after\s+([\d.]+)", error_str, re.IGNORECASE)
                if match:
                    retry_after = int(float(match.group(1))) + 5
                raise RateLimitError(
                    message=f"Gemini rate limit exceeded. Please try again in {retry_after or 60} seconds.",
                    detail=error_str,
                    retry_after=retry_after,
                ) from exc
            if _is_overloaded(error_str):
                raise LLMProviderError(_overloaded_message(model_name), detail=error_str) from exc
            raise LLMProviderError(f"Gemini request failed: {exc}", detail=error_str) from exc

    async def health_check(self) -> bool:
        """Ping Gemini with a minimal request."""
        if not self._api_key or self._client is None:
            return False
        try:
            from google.genai import types as genai_types

            await self._client.aio.models.generate_content(
                model=self._model,
                contents="hi",
                config=genai_types.GenerateContentConfig(max_output_tokens=1),
            )
            return True
        except Exception:
            return False

    async def stream_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream a response from Gemini token by token."""
        self._check_key()
        assert self._client is not None
        from google.genai import types as genai_types

        log = logger.bind(model=self._model, provider="gemini", streaming=True)
        try:
            log.debug("Starting streaming completion request")
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=with_reasoning_headroom(max_tokens, reasoning=True),
                system_instruction=system_prompt if system_prompt else None,
            )
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=user_prompt,
                config=config,
            ):
                text = getattr(chunk, "text", None)
                if text:
                    yield text
            log.debug("Streaming completion finished")
        except Exception as exc:
            error_str = str(exc)
            log.error("Gemini streaming API error", error=error_str)
            if "rate" in error_str.lower() or "quota" in error_str.lower() or "429" in error_str:
                raise RateLimitError(
                    message="Gemini rate limit exceeded. Please try again later.",
                    detail=error_str,
                    retry_after=None,
                ) from exc
            if _is_overloaded(error_str):
                raise LLMProviderError(_overloaded_message(self._model), detail=error_str) from exc
            raise LLMProviderError(
                f"Gemini streaming request failed: {exc}", detail=error_str
            ) from exc
