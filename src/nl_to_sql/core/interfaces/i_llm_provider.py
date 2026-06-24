"""ILLMProvider — Abstract interface for LLM backends (OpenAI, Groq, etc.)."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from nl_to_sql.core.models.sql_result import LLMResponse


class ILLMProvider(ABC):
    """Contract for any LLM provider.

    SOLID: Open/Closed — closed for modification, open for extension.
           Liskov       — any concrete provider is substitutable here.
    """

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Send a chat-completion request and return a structured response.

        Args:
            system_prompt: Instruction context for the LLM.
            user_prompt: The user's NL query with retrieved schema context.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the completion.

        Returns:
            LLMResponse containing raw text and token usage.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the provider is reachable (used by /ready endpoint)."""
        ...

    @abstractmethod
    async def stream_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream a chat-completion response token by token.

        Args:
            system_prompt: Instruction context for the LLM.
            user_prompt: The user's NL query with retrieved schema context.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the completion.

        Yields:
            Chunks of generated text as they become available.
        """
        ...
