"""Domain-specific exception hierarchy for the NL-to-SQL pipeline."""


class NLToSQLBaseError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ── LLM / Generation Errors ──────────────────────────────────────────────────


class LLMProviderError(NLToSQLBaseError):
    """Raised when the LLM provider returns an error or is unreachable."""


class RateLimitError(NLToSQLBaseError):
    """Raised when the LLM provider rate limit is exceeded."""

    def __init__(self, message: str, detail: str | None = None, retry_after: int | None = None) -> None:
        super().__init__(message, detail)
        self.retry_after = retry_after  # seconds to wait before retrying


class SQLGenerationError(NLToSQLBaseError):
    """Raised when the LLM fails to produce valid SQL after all retries."""


# ── Schema / Retrieval Errors ─────────────────────────────────────────────────


class SchemaIngestionError(NLToSQLBaseError):
    """Raised during schema parsing or embedding errors at ingestion time."""


class SchemaRetrievalError(NLToSQLBaseError):
    """Raised when schema chunks cannot be retrieved from the vector store."""


class EmptySchemaError(NLToSQLBaseError):
    """Raised when the vector store has no schema chunks for querying."""


# ── Validation Errors ─────────────────────────────────────────────────────────


class SQLValidationError(NLToSQLBaseError):
    """Raised when the generated SQL fails structural/syntactic validation."""


# ── Embedding Errors ──────────────────────────────────────────────────────────


class EmbeddingError(NLToSQLBaseError):
    """Raised when the embedding provider fails."""


# ── Vector Store Errors ───────────────────────────────────────────────────────


class VectorStoreError(NLToSQLBaseError):
    """Raised when the vector store operation fails."""


# ── Cache Errors ──────────────────────────────────────────────────────────────


class CacheError(NLToSQLBaseError):
    """Raised when a cache read/write operation fails."""


# ── Configuration Errors ──────────────────────────────────────────────────────


class ConfigurationError(NLToSQLBaseError):
    """Raised when the application is misconfigured."""


# ── Database Errors ───────────────────────────────────────────────────────────


class DatabaseExecutionError(NLToSQLBaseError):
    """Raised when executing a query against the target database fails."""
