"""LLM provider factory — selects a concrete ILLMProvider from settings."""
from nl_to_sql.config.settings import Settings
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.infrastructure.llm.groq_provider import GroqProvider


def build_llm_provider(settings: Settings) -> ILLMProvider:
    """Factory: choose the LLM provider from settings."""
    return create_llm_provider(settings.llm_provider, settings.llm_model, settings)


def create_llm_provider(provider: str, model: str, settings: Settings) -> ILLMProvider:
    """Factory: create a specific LLM provider instance.

    Args:
        provider: The provider name ("groq", "openai", "anthropic", or "gemini").
        model: The model name to use.
        settings: Application settings for API keys.

    Returns:
        Configured ILLMProvider instance.
    """
    if provider == "openai":
        from nl_to_sql.infrastructure.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=settings.openai_api_key, model=model)
    if provider == "anthropic":
        from nl_to_sql.infrastructure.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=model)
    if provider == "gemini":
        from nl_to_sql.infrastructure.llm.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=settings.gemini_api_key, model=model)
    return GroqProvider(api_key=settings.groq_api_key, model=model)
