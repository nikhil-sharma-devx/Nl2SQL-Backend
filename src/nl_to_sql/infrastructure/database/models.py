"""SQLAlchemy ORM models for application data storage."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class User(Base):
    """ORM model for application users.

    Stores both email/password and Google OAuth users.
    auth_provider: 'email' or 'google'
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(Text, nullable=True)  # Null for Google-only users
    auth_provider: Mapped[str] = mapped_column(String(20), nullable=False, default="email")  # 'email' or 'google'
    google_sub: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)  # Google subject ID
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    otp_code: Mapped[str | None] = mapped_column(String(6), nullable=True)
    otp_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # DB-backed (not in-memory) lockout counter — shared across every worker
    # process, unlike a process-local dict that a multi-worker deployment
    # would let an attacker multiply.
    otp_failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    password_history = relationship(
        "PasswordHistory",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(PasswordHistory.created_at)",
    )
    api_keys = relationship("UserAPIKey", back_populates="user", cascade="all, delete-orphan")
    database_connections = relationship(
        "UserDatabaseConnection", back_populates="user", cascade="all, delete-orphan"
    )


class PasswordHistory(Base):
    """Tracks a user's password history to prevent recent reuse."""

    __tablename__ = "password_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="password_history")


class UserAPIKey(Base):
    """Per-user encrypted API keys for BYOK (Bring Your Own Key) LLM providers."""

    __tablename__ = "user_api_keys"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_user_api_key_provider"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(20), nullable=False)
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    # 1 = legacy sha256(secret_key) derivation, 2 = HKDF-SHA256. New/updated
    # rows always write the current version; older rows decrypt via v1 until
    # next write ("migrate opportunistically" — see api_key_service.py).
    kdf_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user = relationship("User", back_populates="api_keys")


class UserDatabaseConnection(Base):
    """A single database connection owned by a user (multi-connection BYOD).

    A user may own many connections and switch the *active* one at any time.

    - ``connection_id``: stable, non-enumerable UUID used as the external
      identifier (API + Qdrant payload key + schema-catalog scope). Never expose
      the integer ``id`` externally.
    - ``encrypted_url``: Fernet-encrypted DSN. ``NULL`` denotes the built-in
      "Server Default" connection, which resolves to the platform's shared
      database — this preserves the pre-multi-connection behaviour for users who
      never configured a personal database.
    - ``is_default``: exactly one row per user is the active/default connection;
      the service enforces the single-default invariant on create/select/delete.
    """

    __tablename__ = "user_database_connections"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_user_connection_name"),
        Index("ix_user_database_connections_user", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    connection_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, index=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="Default")
    db_type: Mapped[str] = mapped_column(String(20), nullable=False, default="postgresql")
    # NULL = the built-in "Server Default" connection (uses the platform database).
    encrypted_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # See UserAPIKey.kdf_version — same versioned-KDF scheme, same module.
    kdf_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user = relationship("User", back_populates="database_connections")


class ChatSession(Base):
    """ORM model for chat sessions.

    Each session contains multiple messages and represents a single conversation.
    """

    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_chat_sessions_user_updated", "user_id", "updated_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )  # Nullable for legacy sessions
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True
    )

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.timestamp",
    )


class ChatMessage(Base):
    """ORM model for individual messages within a chat session.

    Stores both user questions and AI responses.
    """

    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_messages_session_ts", "session_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # User question
    question: Mapped[str] = mapped_column(Text, nullable=False)

    # AI response fields (from QueryResponse)
    sql: Mapped[str] = mapped_column(Text, nullable=False, default="")
    dialect: Mapped[str] = mapped_column(String(50), nullable=False)
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    validation_errors: Mapped[Any] = mapped_column(JSON, nullable=True, default=list)
    retrieved_tables: Mapped[Any] = mapped_column(JSON, nullable=True, default=list)
    execution_result: Mapped[Any] = mapped_column(JSON, nullable=True)
    execution_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    cached: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)  # For greeting/off-topic messages

    # Premium response fields
    used_tables: Mapped[Any] = mapped_column(JSON, nullable=True, default=list)
    suggested_chart: Mapped[Any] = mapped_column(JSON, nullable=True)
    follow_up_questions: Mapped[Any] = mapped_column(JSON, nullable=True, default=list)

    # Analytics and intelligence fields
    intent_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    query_complexity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    retrieval_method: Mapped[str | None] = mapped_column(String(20), nullable=True)
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Soft-delete support (F5 Clear History)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationship back to session
    session = relationship("ChatSession", back_populates="messages")


class QueryHistoryRecord(Base):
    """ORM model for storing query history.

    Maps to the query_history table with all fields from QueryResponse
    plus metadata for retrieval and analytics.

    DEPRECATED: Use ChatSession and ChatMessage instead.
    Kept for backward compatibility and analytics.
    """

    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    sql: Mapped[str] = mapped_column(Text, nullable=False, default="")
    dialect: Mapped[str] = mapped_column(String(50), nullable=False)
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    validation_errors: Mapped[Any] = mapped_column(JSON, nullable=True, default=list)
    retrieved_tables: Mapped[Any] = mapped_column(JSON, nullable=True, default=list)
    execution_result: Mapped[Any] = mapped_column(JSON, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    cached: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    execution_error: Mapped[str | None] = mapped_column(Text, nullable=True)  # Error message if SQL execution failed
    message: Mapped[str | None] = mapped_column(Text, nullable=True)  # For greeting/off-topic messages

    # Analytics and intelligence fields
    intent_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # aggregation, filtering, join, etc.
    query_complexity: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-10 scale
    prompt_version: Mapped[str | None] = mapped_column(String(20), nullable=True)  # For A/B testing
    retrieval_method: Mapped[str | None] = mapped_column(String(20), nullable=True)  # vector, hybrid, semantic_cache
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Latency tracking
    user_feedback: Mapped[str | None] = mapped_column(String(20), nullable=True)  # correct, incorrect, no_feedback
    corrected_sql: Mapped[str | None] = mapped_column(Text, nullable=True)  # User-provided correction


class FeedbackRecord(Base):
    """ORM model for storing user feedback on query results."""

    __tablename__ = "feedback"
    __table_args__ = (Index("ix_feedback_ts_type", "timestamp", "feedback_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("query_history.id"), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    feedback_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # correct_tables, incorrect_tables, sql_correction, general
    feedback_data: Mapped[Any] = mapped_column(JSON, nullable=True)  # Flexible feedback data
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class SqlVersion(Base):
    """User-edited SQL versions for a specific chat message.

    Tracks the history of SQL edits made by the user after initial generation.
    """

    __tablename__ = "sql_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_messages.id"), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    sql: Mapped[str] = mapped_column(Text, nullable=False)
    results: Mapped[Any] = mapped_column(JSON, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    is_original: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    message = relationship("ChatMessage", backref="sql_versions")


class TrainingDataRecord(Base):
    """ORM model for storing successful queries for fine-tuning.

    ``user_id``/``connection_id`` (C8) scope few-shot retrieval
    (``get_recent_examples``) to the requesting tenant — nullable because
    rows collected before this column existed, and rows collected outside an
    authenticated request, have no owner and are treated as shared.
    """

    __tablename__ = "training_data"
    __table_args__ = (
        Index("ix_training_data_used_created", "used_for_training", "created_at"),
        Index("ix_training_data_connection", "connection_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    sql: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_tables: Mapped[Any] = mapped_column(JSON, nullable=True)
    schema_context: Mapped[str | None] = mapped_column(Text, nullable=True)  # Schema used for generation
    success_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # Quality score 0-1
    intent_type: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    used_for_training: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, default=False, index=True
    )  # Track if used in fine-tuning
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    connection_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


# ── Phase 1 Feature Models ─────────────────────────────────────────────────────


class UserInstructions(Base):
    """F1 - Custom Instructions persisted per user."""

    __tablename__ = "user_instructions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class UserInstructionsHistory(Base):
    """F1 - History of previous custom instructions (for recoverability)."""

    __tablename__ = "user_instructions_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    replaced_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class UserSettings(Base):
    """F2 + F4 - Per-user SQL style and app behavior settings."""

    __tablename__ = "user_settings"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    sql_keyword_case: Mapped[str] = mapped_column(String(10), nullable=False, default="upper")
    sql_cte_pref: Mapped[str] = mapped_column(String(10), nullable=False, default="cte")
    sql_alias_style: Mapped[str] = mapped_column(String(10), nullable=False, default="as")
    sql_indent: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    default_dialect: Mapped[str | None] = mapped_column(String(50), nullable=True)
    max_result_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=1000)
    auto_execute: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    default_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    data_retention: Mapped[str] = mapped_column(String(10), nullable=False, default="forever")
    font_size: Mapped[str] = mapped_column(String(10), nullable=False, default="medium")
    ui_density: Mapped[str] = mapped_column(String(15), nullable=False, default="comfortable")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class SavedQuery(Base):
    """F3 - Saved/starred queries for reuse."""

    __tablename__ = "saved_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    nl_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    generated_sql: Mapped[str] = mapped_column(Text, nullable=False)
    dialect: Mapped[str | None] = mapped_column(String(50), nullable=True)
    starred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    run_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class QueryMetrics(Base):
    """F6 - Per-query token and cost metrics for usage reporting."""

    __tablename__ = "query_metrics"
    __table_args__ = (Index("ix_query_metrics_user_created", "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    query_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class DataExportJob(Base):
    """F7 - Async data export job tracking (Download My Data)."""

    __tablename__ = "data_export_jobs"
    __table_args__ = (Index("ix_data_export_jobs_user_status", "user_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AccountDeletion(Base):
    """F7 - Account deletion grace period tracking."""

    __tablename__ = "account_deletions"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    purge_after: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="scheduled")


class UserLoginSession(Base):
    """F8 - Login session registry for session management."""

    __tablename__ = "user_login_sessions"
    __table_args__ = (
        Index("ix_login_sessions_user_revoked", "user_id", "revoked_at"),
        Index("ix_login_sessions_user_active", "user_id", "last_active_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    device: Mapped[str | None] = mapped_column(String(200), nullable=True)
    browser: Mapped[str | None] = mapped_column(String(200), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(50), nullable=True)
    last_active_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LoginEvent(Base):
    """F8 - Login audit log."""

    __tablename__ = "login_events"
    __table_args__ = (
        Index("ix_login_events_user_created", "user_id", "created_at"),
        Index("ix_login_events_outcome_created", "outcome", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    ip: Mapped[str | None] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False, default="success")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class RefreshToken(Base):
    """Opaque refresh token (hashed at rest) bound to a login session.

    Enables short-lived access tokens with rotation: the raw token is returned
    to the client only once; the server stores only its SHA-256 hash. On refresh
    the presented row is revoked (rotated) and a fresh one issued. Revoking the
    parent login session (``user_login_sessions.revoked_at``) also invalidates
    its refresh tokens.
    """

    __tablename__ = "refresh_tokens"
    __table_args__ = (Index("ix_refresh_tokens_user", "user_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("user_login_sessions.id"), nullable=True, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


# ── Phase 2 Feature Models ─────────────────────────────────────────────────────


class QueryTemplate(Base):
    """P2 - Parameterized query templates (saved patterns with {{placeholder}} vars)."""

    __tablename__ = "query_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    template_nl: Mapped[str] = mapped_column(Text, nullable=False)
    template_sql: Mapped[str] = mapped_column(Text, nullable=False)
    parameters: Mapped[Any] = mapped_column(
        JSON, nullable=False, default=list
    )  # [{name, type, description, default}]
    tags: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)
    is_builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # seeded starter example
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class FavoritedTable(Base):
    """P2 - User-pinned tables that get retrieval priority during query generation."""

    __tablename__ = "favorited_tables"
    __table_args__ = (UniqueConstraint("user_id", "table_name", name="uq_user_favorited_table"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    table_name: Mapped[str] = mapped_column(String(200), nullable=False)
    schema_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class GlossaryEntry(Base):
    """P2 - Business dictionary entries injected into the generation prompt."""

    __tablename__ = "glossary_entries"
    __table_args__ = (UniqueConstraint("user_id", "term", name="uq_user_glossary_term"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    term: Mapped[str] = mapped_column(String(200), nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Metric(Base):
    """Semantic layer — a governed business metric, scoped to a DB connection.

    Only ``certified=True`` metrics are injected into the SQL-generation
    prompt (see ``api/routes/query.py::_load_certified_metrics_context``) —
    uncertified/draft rows are catalog-only. ``connection_id`` is a loose
    reference (not an FK object), matching ``UserSchemaTable.connection_id``'s
    convention: scoping is validated by the service layer, not a DB constraint,
    since a metric's SQL definition is only meaningful against one specific
    connection's schema (never injected cross-connection).
    """

    __tablename__ = "metrics"
    __table_args__ = (
        UniqueConstraint("connection_id", "name", name="uq_connection_metric_name"),
        Index("ix_metrics_connection", "connection_id"),
        Index("ix_metrics_user", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    connection_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    sql_definition: Mapped[str] = mapped_column(Text, nullable=False)
    dimensions: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)  # list[str]
    tags: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)  # list[str]
    owner: Mapped[str | None] = mapped_column(String(200), nullable=True)  # free-text owner/team — not an auth role
    certified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # seeded starter example
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TutorialProgress(Base):
    """P2 - Tracks which tutorial/walkthrough steps a user has completed."""

    __tablename__ = "tutorial_progress"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    completed_steps: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class OnboardingState(Base):
    """P2 - Tracks which onboarding checklist items a user has completed."""

    __tablename__ = "onboarding_state"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    completed_items: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class NotificationPreferences(Base):
    """P2 - Per-user notification opt-in/out settings."""

    __tablename__ = "notification_preferences"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    email_digest: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    in_app_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    marketing_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Timestamp of the last activity digest sent to this user (cadence guard).
    last_digest_sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class SharedQuery(Base):
    """Export & Share — a shareable snapshot of a query and its result.

    Created by an owner (``user_id``) and accessed publicly via a signed share
    token (see ``services/export_service.py``). The snapshot stores only the
    question, generated SQL, and a bounded copy of the result rows — never any
    DSN, secret, or other user's data. Expiry and revocation are enforced from
    this row (the token authenticates the ``id``; the DB is authoritative for
    the 410 expired/revoked semantics).
    """

    __tablename__ = "shared_queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Natural-language prompt behind the query (stored under both the friendly
    # and the schema-consistent name via the route mapping).
    question: Mapped[str] = mapped_column(Text, nullable=False, default="")
    nl_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_sql: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Bounded copy of the result rows (capped in the service before persist).
    result_snapshot: Mapped[Any] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ── Schema Management (Schema Catalog) Models ───────────────────────────────────


class UserSchema(Base):
    """Per-user schema catalog header — source of truth for the Schema page.

    One row per user. Tracks where the catalog came from (`reflected` /
    `uploaded` / `merged`), the last-synced hash for change detection, and the
    raw uploaded JSON for audit/restore.
    """

    __tablename__ = "user_schemas"
    __table_args__ = (
        UniqueConstraint("connection_id", name="uq_user_schema_connection"),
        Index("ix_user_schemas_user", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    # Owning connection (multi-DB). Nullable only for legacy rows pre-backfill;
    # the catalog service always sets it. One catalog header per connection.
    connection_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    database_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    dialect: Mapped[str] = mapped_column(String(50), nullable=False, default="postgresql")
    # 'reflected' | 'uploaded' | 'merged'
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="reflected")
    schema_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # SchemaIngestionService.compute_schema_hash
    raw_upload_json: Mapped[Any] = mapped_column(
        JSON, nullable=True
    )  # exact payload the user uploaded (audit/restore)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class UserSchemaTable(Base):
    """One row per table in a user's catalog. Columns stored as JSON for simplicity.

    ``user_description`` is sticky — it survives re-reflection. The effective
    description used at embed time is ``user_description or reflected_description``.
    """

    __tablename__ = "user_schema_tables"
    __table_args__ = (
        UniqueConstraint("connection_id", "schema_name", "table_name", name="uq_conn_schema_table"),
        Index("ix_user_schema_tables_user", "user_id"),
        Index("ix_user_schema_tables_connection", "connection_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    # Owning connection (multi-DB). Nullable only for legacy rows pre-backfill.
    connection_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    schema_name: Mapped[str] = mapped_column(String(100), nullable=False, default="public")
    table_name: Mapped[str] = mapped_column(String(200), nullable=False)
    # 'reflected' | 'uploaded'
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="reflected")
    # list[ColumnInfo] as JSON: [{name, data_type, nullable, primary_key, foreign_key, description}]
    columns: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)
    reflected_description: Mapped[str | None] = mapped_column(Text, nullable=True)  # from DB comment, if any
    user_description: Mapped[str | None] = mapped_column(Text, nullable=True)  # user override, survives re-reflection
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )  # updated each sync
    is_new: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)  # cleared once the user has viewed it


# ── Auto Charting & Dashboards Models ───────────────────────────────────────────


class Dashboard(Base):
    """A user-owned dashboard: a named, ordered collection of chart widgets.

    Per-user (``user_id`` FK). Widgets cascade-delete with the dashboard. The
    ``updated_at`` column powers recency ordering on the list view.
    """

    __tablename__ = "dashboards"
    __table_args__ = (Index("ix_dashboards_user_updated", "user_id", "updated_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="Untitled Dashboard")
    is_builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # seeded starter example
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    widgets = relationship(
        "DashboardWidget",
        back_populates="dashboard",
        cascade="all, delete-orphan",
        order_by="DashboardWidget.position",
    )


class DashboardWidget(Base):
    """A single chart tile on a dashboard.

    Holds the SQL that produces the tile's data, the recommended/overridden
    ``chart_type``, a ``chart_config`` JSON compatible with the frontend
    ``DataChart`` (``{type, x_axis, y_axis}``), a responsive ``layout`` JSON
    (``{x, y, w, h}``), and an integer ``position`` for deterministic ordering.
    """

    __tablename__ = "dashboard_widgets"
    __table_args__ = (Index("ix_dashboard_widgets_dashboard", "dashboard_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    dashboard_id: Mapped[str] = mapped_column(String(36), ForeignKey("dashboards.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="Untitled")
    nl_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    sql: Mapped[str] = mapped_column(Text, nullable=False, default="")
    chart_type: Mapped[str] = mapped_column(String(20), nullable=False, default="table")
    chart_config: Mapped[Any] = mapped_column(JSON, nullable=True)  # {type, x_axis, y_axis}
    layout: Mapped[Any] = mapped_column(JSON, nullable=True)  # {x, y, w, h}
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    dashboard = relationship("Dashboard", back_populates="widgets")


# ── Scheduled Queries & Alerts Models ────────────────────────────────────────


class ScheduledQuery(Base):
    """A user-owned recurring natural-language query with email alerting.

    ``connection_id`` is a loose reference (not an FK object) to a
    ``UserDatabaseConnection.connection_id`` — same convention already used by
    ``UserSchemaTable.connection_id`` — since connection scoping is validated by
    the service layer, not enforced at the DB level. ``cron_expr`` is always the
    canonical, resolved 5-field cron string; ``raw_schedule_text`` retains the
    original NL phrase ("every morning") for display/edit. ``notify_in_app`` is
    reserved and not yet delivered (no in-app notification feed exists yet) —
    kept False by default and not exposed as a working toggle in the UI.
    """

    __tablename__ = "scheduled_queries"
    __table_args__ = (
        Index("ix_scheduled_queries_user", "user_id"),
        Index("ix_scheduled_queries_connection", "connection_id"),
        Index("ix_scheduled_queries_next_run", "next_run_at", "is_paused"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    connection_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    nl_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    cron_expr: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_schedule_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), nullable=False, default="UTC")
    is_paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notify_email: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notify_in_app: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # 'always' | 'on_results' | 'on_change'
    notify_condition: Mapped[str] = mapped_column(String(20), nullable=False, default="always")
    last_row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(20), nullable=True)  # 'success' | 'failed' | 'running'
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # seeded starter example
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    runs = relationship(
        "ScheduledQueryRun",
        back_populates="schedule",
        cascade="all, delete-orphan",
        order_by="desc(ScheduledQueryRun.started_at)",
    )


class ScheduledQueryRun(Base):
    """One execution history record for a ``ScheduledQuery`` (feeds "View history").

    Stores only row_count/generated_sql/error — never the result rows themselves,
    to avoid unbounded PII retention. The service caps retained rows per schedule.
    """

    __tablename__ = "scheduled_query_runs"
    __table_args__ = (
        Index("ix_scheduled_query_runs_schedule_started", "schedule_id", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    schedule_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("scheduled_queries.id"), nullable=False, index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # 'success' | 'failed' | 'timeout'
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generated_sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    notified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    schedule = relationship("ScheduledQuery", back_populates="runs")
