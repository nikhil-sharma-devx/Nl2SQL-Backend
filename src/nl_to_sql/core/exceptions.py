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

    def __init__(
        self, message: str, detail: str | None = None, retry_after: int | None = None
    ) -> None:
        super().__init__(message, detail)
        self.retry_after = retry_after  # seconds to wait before retrying


class DailyQuotaExceededError(RateLimitError):
    """Raised when a user has used up their daily AI-query allowance.

    Subclasses RateLimitError so the envelope carries ``retry_after`` (seconds
    until the allowance resets at midnight UTC).
    """


class AuthRateLimitError(RateLimitError):
    """Raised when login / OTP / password-reset attempts exceed their throttle.

    Subclasses RateLimitError so the envelope carries ``retry_after``.
    """


class SQLGenerationError(NLToSQLBaseError):
    """Raised when the LLM fails to produce valid SQL after all retries."""


# ── Schema / Retrieval Errors ─────────────────────────────────────────────────


class SchemaIngestionError(NLToSQLBaseError):
    """Raised during schema parsing or embedding errors at ingestion time."""


class SchemaRetrievalError(NLToSQLBaseError):
    """Raised when schema chunks cannot be retrieved from the vector store."""


class EmptySchemaError(NLToSQLBaseError):
    """Raised when the vector store has no schema chunks for querying."""


class TableNotFoundError(NLToSQLBaseError):
    """Raised when a requested table/column does not exist for the user.

    Scoped per user: a table owned by another user is reported as "not found"
    (never confirmed to exist) so cross-user existence cannot be probed.
    """


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


# ── Authentication Errors ──────────────────────────────────────────────────────


class AuthenticationError(NLToSQLBaseError):
    """Raised when a credential (e.g. an external OAuth token) fails verification."""


# ── Chat Session Errors ─────────────────────────────────────────────────────────


class SessionNotFoundError(NLToSQLBaseError):
    """Raised when a chat session does not exist (e.g. deleted mid-request)."""


# ── Configuration Errors ──────────────────────────────────────────────────────


class ConfigurationError(NLToSQLBaseError):
    """Raised when the application is misconfigured."""


# ── Database Errors ───────────────────────────────────────────────────────────


class DatabaseExecutionError(NLToSQLBaseError):
    """Raised when executing a query against the target database fails."""


class AccessPolicyViolationError(DatabaseExecutionError):
    """Raised when SQL references a table or column the connection's access
    policy forbids. Subclasses ``DatabaseExecutionError`` on purpose: the query
    orchestrator's self-correction loop already feeds execution errors back to
    the LLM, so a violation becomes "rewrite without that column" for free."""


# ── Connection (multi-DB) Errors ──────────────────────────────────────────────


class ConnectionNotFoundError(NLToSQLBaseError):
    """Raised when a database connection does not exist for the current user.

    Scoped per user: a connection owned by another user is reported as "not
    found" (never confirmed to exist) so cross-user existence cannot be probed.
    """


class ConnectionValidationError(NLToSQLBaseError):
    """Raised when a connection's inputs are invalid (bad DSN, duplicate name)."""


# ── Workspace (team) Errors ───────────────────────────────────────────────────


class WorkspaceNotFoundError(NLToSQLBaseError):
    """Raised for an unknown workspace OR one the caller isn't a member of
    (never confirms that another team's workspace exists)."""


class WorkspacePermissionError(NLToSQLBaseError):
    """Raised when a member's role doesn't allow the requested action."""


class WorkspaceValidationError(NLToSQLBaseError):
    """Raised for invalid workspace input, e.g. adding an email with no account
    or removing the last admin."""


class WorkspaceConflictError(NLToSQLBaseError):
    """Raised when the user is already a member of the workspace."""


# ── Platform admin Errors ────────────────────────────────────────────────────


class AdminPermissionError(NLToSQLBaseError):
    """Raised when an admin action needs more than the caller has: removing an
    admin is owner-only, and the one-time setup needs the right setup token."""


class AccountDeactivatedError(NLToSQLBaseError):
    """Raised when a deactivated account signs in with valid credentials.

    ``detail`` carries a short-lived reactivation token the sign-in page uses
    to send a reactivation request to the admins.
    """


class ReactivationPendingError(AccountDeactivatedError):
    """A deactivated account signed in, but its reactivation request is
    already waiting for an admin — nothing more for the sign-in page to send."""


class AdminConflictError(NLToSQLBaseError):
    """Raised when the action no longer applies: setup already done, or the
    target is already (or no longer) an admin in a way the action can't change."""


class ConnectionTestError(NLToSQLBaseError):
    """Raised when a database connection cannot be reached during a test."""


# ── Scheduled Query Errors ────────────────────────────────────────────────────


class ScheduleNotFoundError(NLToSQLBaseError):
    """Raised when a scheduled query does not exist for the current user.

    Scoped per user: a schedule owned by another user is reported as "not
    found" (never confirmed to exist) so cross-user existence cannot be probed.
    """


class ScheduleValidationError(NLToSQLBaseError):
    """Raised when a schedule's inputs are invalid (bad cron/NL phrase, unknown connection)."""


class ScheduleExecutionError(NLToSQLBaseError):
    """Raised when a scheduled query fails to execute (orchestrator/DB error)."""


# ── Metrics Catalog Errors ────────────────────────────────────────────────────


class MetricNotFoundError(NLToSQLBaseError):
    """Raised when a metric does not exist for the current user/connection.

    Scoped per user+connection: a metric outside the caller's access is
    reported as "not found" (never confirmed to exist).
    """


class MetricValidationError(NLToSQLBaseError):
    """Raised when a metric's inputs are invalid (duplicate name, bad SQL definition)."""
