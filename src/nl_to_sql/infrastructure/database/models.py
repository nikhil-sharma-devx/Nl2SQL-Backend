"""SQLAlchemy ORM models for application data storage."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class User(Base):
    """ORM model for application users.

    Stores both email/password and Google OAuth users.
    auth_provider: 'email' or 'google'
    """

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(200), nullable=True)
    hashed_password = Column(Text, nullable=True)  # Null for Google-only users
    auth_provider = Column(String(20), nullable=False, default="email")  # 'email' or 'google'
    google_sub = Column(String(255), nullable=True, unique=True)  # Google subject ID
    is_active = Column(Boolean, nullable=False, default=True)
    is_verified = Column(Boolean, nullable=False, default=False)
    otp_code = Column(String(6), nullable=True)
    otp_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    password_history = relationship("PasswordHistory", back_populates="user", cascade="all, delete-orphan", order_by="desc(PasswordHistory.created_at)")
    api_keys = relationship("UserAPIKey", back_populates="user", cascade="all, delete-orphan")
    database_connections = relationship("UserDatabaseConnection", back_populates="user", cascade="all, delete-orphan")


class PasswordHistory(Base):
    """Tracks a user's password history to prevent recent reuse."""
    __tablename__ = "password_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="password_history")


class UserAPIKey(Base):
    """Per-user encrypted API keys for BYOK (Bring Your Own Key) LLM providers."""

    __tablename__ = "user_api_keys"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_user_api_key_provider"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String(20), nullable=False)
    encrypted_key = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

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

    id = Column(Integer, primary_key=True, autoincrement=True)
    connection_id = Column(
        String(36), nullable=False, unique=True, index=True, default=lambda: str(uuid4())
    )
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(200), nullable=False, default="Default")
    db_type = Column(String(20), nullable=False, default="postgresql")
    # NULL = the built-in "Server Default" connection (uses the platform database).
    encrypted_url = Column(Text, nullable=True)
    is_default = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="database_connections")


class ChatSession(Base):
    """ORM model for chat sessions.

    Each session contains multiple messages and represents a single conversation.
    """

    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("ix_chat_sessions_user_updated", "user_id", "updated_at"),
    )

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)  # Nullable for legacy sessions
    title = Column(String(200), nullable=False, default="New Chat")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")


class ChatMessage(Base):
    """ORM model for individual messages within a chat session.

    Stores both user questions and AI responses.
    """

    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_ts", "session_id", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # User question
    question = Column(Text, nullable=False)

    # AI response fields (from QueryResponse)
    sql = Column(Text, nullable=False, default="")
    dialect = Column(String(50), nullable=False)
    is_valid = Column(Boolean, nullable=False)
    validation_errors = Column(JSON, default=list)
    retrieved_tables = Column(JSON, default=list)
    execution_result = Column(JSON, nullable=True)
    execution_error = Column(Text, nullable=True)
    tokens_used = Column(Integer, default=0)
    cached = Column(Boolean, default=False)
    message = Column(Text, nullable=True)  # For greeting/off-topic messages

    # Premium response fields
    used_tables = Column(JSON, default=list)
    suggested_chart = Column(JSON, nullable=True)
    follow_up_questions = Column(JSON, default=list)

    # Analytics and intelligence fields
    intent_type = Column(String(50), nullable=True)
    query_complexity = Column(Integer, nullable=True)
    prompt_version = Column(String(20), nullable=True)
    retrieval_method = Column(String(20), nullable=True)
    response_time_ms = Column(Integer, nullable=True)

    # Soft-delete support (F5 Clear History)
    deleted_at = Column(DateTime, nullable=True)

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

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    question = Column(Text, nullable=False)
    sql = Column(Text, nullable=False, default="")
    dialect = Column(String(50), nullable=False)
    is_valid = Column(Boolean, nullable=False)
    validation_errors = Column(JSON, default=list)
    retrieved_tables = Column(JSON, default=list)
    execution_result = Column(JSON, nullable=True)
    tokens_used = Column(Integer, default=0)
    cached = Column(Boolean, default=False)
    execution_error = Column(Text, nullable=True)  # Error message if SQL execution failed
    message = Column(Text, nullable=True)  # For greeting/off-topic messages

    # Analytics and intelligence fields
    intent_type = Column(String(50), nullable=True)  # aggregation, filtering, join, etc.
    query_complexity = Column(Integer, nullable=True)  # 1-10 scale
    prompt_version = Column(String(20), nullable=True)  # For A/B testing
    retrieval_method = Column(String(20), nullable=True)  # vector, hybrid, semantic_cache
    response_time_ms = Column(Integer, nullable=True)  # Latency tracking
    user_feedback = Column(String(20), nullable=True)  # correct, incorrect, no_feedback
    corrected_sql = Column(Text, nullable=True)  # User-provided correction


class FeedbackRecord(Base):
    """ORM model for storing user feedback on query results."""

    __tablename__ = "feedback"
    __table_args__ = (
        Index("ix_feedback_ts_type", "timestamp", "feedback_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("query_history.id"), nullable=True, index=True)
    session_id = Column(String(36), nullable=True, index=True)
    feedback_type = Column(String(20), nullable=False)  # correct_tables, incorrect_tables, sql_correction, general
    feedback_data = Column(JSON, nullable=True)  # Flexible feedback data
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class SqlVersion(Base):
    """User-edited SQL versions for a specific chat message.

    Tracks the history of SQL edits made by the user after initial generation.
    """

    __tablename__ = "sql_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    sql = Column(Text, nullable=False)
    results = Column(JSON, nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_original = Column(Boolean, nullable=False, default=False)

    message = relationship("ChatMessage", backref="sql_versions")


class TrainingDataRecord(Base):
    """ORM model for storing successful queries for fine-tuning."""

    __tablename__ = "training_data"
    __table_args__ = (
        Index("ix_training_data_used_created", "used_for_training", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    sql = Column(Text, nullable=False)
    retrieved_tables = Column(JSON, nullable=True)
    schema_context = Column(Text, nullable=True)  # Schema used for generation
    success_score = Column(Float, nullable=True)  # Quality score 0-1
    intent_type = Column(String(50), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    used_for_training = Column(Boolean, default=False, index=True)  # Track if used in fine-tuning


# ── Phase 1 Feature Models ─────────────────────────────────────────────────────


class UserInstructions(Base):
    """F1 - Custom Instructions persisted per user."""

    __tablename__ = "user_instructions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    content = Column(Text, nullable=False, default="")
    enabled = Column(Boolean, nullable=False, default=True)
    char_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class UserInstructionsHistory(Base):
    """F1 - History of previous custom instructions (for recoverability)."""

    __tablename__ = "user_instructions_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    content = Column(Text, nullable=False, default="")
    replaced_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class UserSettings(Base):
    """F2 + F4 - Per-user SQL style and app behavior settings."""

    __tablename__ = "user_settings"

    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    sql_keyword_case = Column(String(10), nullable=False, default="upper")
    sql_cte_pref = Column(String(10), nullable=False, default="cte")
    sql_alias_style = Column(String(10), nullable=False, default="as")
    sql_indent = Column(Integer, nullable=False, default=2)
    default_dialect = Column(String(50), nullable=True)
    max_result_rows = Column(Integer, nullable=False, default=1000)
    auto_execute = Column(Boolean, nullable=False, default=False)
    default_model = Column(String(100), nullable=True)
    data_retention = Column(String(10), nullable=False, default="forever")
    font_size = Column(String(10), nullable=False, default="medium")
    ui_density = Column(String(15), nullable=False, default="comfortable")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class SavedQuery(Base):
    """F3 - Saved/starred queries for reuse."""

    __tablename__ = "saved_queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=True)
    nl_prompt = Column(Text, nullable=False)
    generated_sql = Column(Text, nullable=False)
    dialect = Column(String(50), nullable=True)
    starred = Column(Boolean, nullable=False, default=False)
    last_run_at = Column(DateTime, nullable=True)
    run_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class QueryMetrics(Base):
    """F6 - Per-query token and cost metrics for usage reporting."""

    __tablename__ = "query_metrics"
    __table_args__ = (
        Index("ix_query_metrics_user_created", "user_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    query_id = Column(String(36), nullable=True)
    tokens_in = Column(Integer, nullable=False, default=0)
    tokens_out = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=True)
    model = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class DataExportJob(Base):
    """F7 - Async data export job tracking (Download My Data)."""

    __tablename__ = "data_export_jobs"
    __table_args__ = (
        Index("ix_data_export_jobs_user_status", "user_id", "status"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="queued")
    artifact_path = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class AccountDeletion(Base):
    """F7 - Account deletion grace period tracking."""

    __tablename__ = "account_deletions"

    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    requested_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    purge_after = Column(DateTime, nullable=False)
    status = Column(String(20), nullable=False, default="scheduled")


class UserLoginSession(Base):
    """F8 - Login session registry for session management."""

    __tablename__ = "user_login_sessions"
    __table_args__ = (
        Index("ix_login_sessions_user_revoked", "user_id", "revoked_at"),
        Index("ix_login_sessions_user_active", "user_id", "last_active_at"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    device = Column(String(200), nullable=True)
    browser = Column(String(200), nullable=True)
    ip = Column(String(50), nullable=True)
    last_active_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    revoked_at = Column(DateTime, nullable=True)


class LoginEvent(Base):
    """F8 - Login audit log."""

    __tablename__ = "login_events"
    __table_args__ = (
        Index("ix_login_events_user_created", "user_id", "created_at"),
        Index("ix_login_events_outcome_created", "outcome", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    ip = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    outcome = Column(String(20), nullable=False, default="success")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class RefreshToken(Base):
    """Opaque refresh token (hashed at rest) bound to a login session.

    Enables short-lived access tokens with rotation: the raw token is returned
    to the client only once; the server stores only its SHA-256 hash. On refresh
    the presented row is revoked (rotated) and a fresh one issued. Revoking the
    parent login session (``user_login_sessions.revoked_at``) also invalidates
    its refresh tokens.
    """

    __tablename__ = "refresh_tokens"
    __table_args__ = (
        Index("ix_refresh_tokens_user", "user_id"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(
        String(36), ForeignKey("user_login_sessions.id"), nullable=True, index=True
    )
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Phase 2 Feature Models ─────────────────────────────────────────────────────


class QueryTemplate(Base):
    """P2 - Parameterized query templates (saved patterns with {{placeholder}} vars)."""

    __tablename__ = "query_templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    template_nl = Column(Text, nullable=False)
    template_sql = Column(Text, nullable=False)
    parameters = Column(JSON, nullable=False, default=list)  # [{name, type, description, default}]
    tags = Column(JSON, nullable=False, default=list)
    is_builtin = Column(Boolean, nullable=False, default=False)  # seeded starter example
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class FavoritedTable(Base):
    """P2 - User-pinned tables that get retrieval priority during query generation."""

    __tablename__ = "favorited_tables"
    __table_args__ = (UniqueConstraint("user_id", "table_name", name="uq_user_favorited_table"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    table_name = Column(String(200), nullable=False)
    schema_name = Column(String(100), nullable=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class GlossaryEntry(Base):
    """P2 - Business dictionary entries injected into the generation prompt."""

    __tablename__ = "glossary_entries"
    __table_args__ = (UniqueConstraint("user_id", "term", name="uq_user_glossary_term"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    term = Column(String(200), nullable=False)
    definition = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


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

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    connection_id = Column(String(36), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    sql_definition = Column(Text, nullable=False)
    dimensions = Column(JSON, nullable=False, default=list)  # list[str]
    tags = Column(JSON, nullable=False, default=list)  # list[str]
    owner = Column(String(200), nullable=True)  # free-text owner/team — not an auth role
    certified = Column(Boolean, nullable=False, default=False)
    is_builtin = Column(Boolean, nullable=False, default=False)  # seeded starter example
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class TutorialProgress(Base):
    """P2 - Tracks which tutorial/walkthrough steps a user has completed."""

    __tablename__ = "tutorial_progress"

    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    completed_steps = Column(JSON, nullable=False, default=list)
    dismissed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class OnboardingState(Base):
    """P2 - Tracks which onboarding checklist items a user has completed."""

    __tablename__ = "onboarding_state"

    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    completed_items = Column(JSON, nullable=False, default=list)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class NotificationPreferences(Base):
    """P2 - Per-user notification opt-in/out settings."""

    __tablename__ = "notification_preferences"

    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    email_digest = Column(Boolean, nullable=False, default=False)
    in_app_enabled = Column(Boolean, nullable=False, default=True)
    marketing_enabled = Column(Boolean, nullable=False, default=False)
    # Timestamp of the last activity digest sent to this user (cadence guard).
    last_digest_sent_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


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

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=True)
    # Natural-language prompt behind the query (stored under both the friendly
    # and the schema-consistent name via the route mapping).
    question = Column(Text, nullable=False, default="")
    nl_prompt = Column(Text, nullable=True)
    generated_sql = Column(Text, nullable=False, default="")
    # Bounded copy of the result rows (capped in the service before persist).
    result_snapshot = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)


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

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    # Owning connection (multi-DB). Nullable only for legacy rows pre-backfill;
    # the catalog service always sets it. One catalog header per connection.
    connection_id = Column(String(36), nullable=True, index=True)
    database_name = Column(String(200), nullable=True)
    dialect = Column(String(50), nullable=False, default="postgresql")
    # 'reflected' | 'uploaded' | 'merged'
    source = Column(String(20), nullable=False, default="reflected")
    schema_hash = Column(String(64), nullable=True)  # SchemaIngestionService.compute_schema_hash
    raw_upload_json = Column(JSON, nullable=True)  # exact payload the user uploaded (audit/restore)
    last_synced_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserSchemaTable(Base):
    """One row per table in a user's catalog. Columns stored as JSON for simplicity.

    ``user_description`` is sticky — it survives re-reflection. The effective
    description used at embed time is ``user_description or reflected_description``.
    """

    __tablename__ = "user_schema_tables"
    __table_args__ = (
        UniqueConstraint(
            "connection_id", "schema_name", "table_name", name="uq_conn_schema_table"
        ),
        Index("ix_user_schema_tables_user", "user_id"),
        Index("ix_user_schema_tables_connection", "connection_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    # Owning connection (multi-DB). Nullable only for legacy rows pre-backfill.
    connection_id = Column(String(36), nullable=True, index=True)
    schema_name = Column(String(100), nullable=False, default="public")
    table_name = Column(String(200), nullable=False)
    # 'reflected' | 'uploaded'
    source = Column(String(20), nullable=False, default="reflected")
    # list[ColumnInfo] as JSON: [{name, data_type, nullable, primary_key, foreign_key, description}]
    columns = Column(JSON, nullable=False, default=list)
    reflected_description = Column(Text, nullable=True)  # from DB comment, if any
    user_description = Column(Text, nullable=True)  # user override, survives re-reflection
    first_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)  # updated each sync
    is_new = Column(Boolean, nullable=False, default=True)  # cleared once the user has viewed it


# ── Auto Charting & Dashboards Models ───────────────────────────────────────────


class Dashboard(Base):
    """A user-owned dashboard: a named, ordered collection of chart widgets.

    Per-user (``user_id`` FK). Widgets cascade-delete with the dashboard. The
    ``updated_at`` column powers recency ordering on the list view.
    """

    __tablename__ = "dashboards"
    __table_args__ = (
        Index("ix_dashboards_user_updated", "user_id", "updated_at"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(200), nullable=False, default="Untitled Dashboard")
    is_builtin = Column(Boolean, nullable=False, default=False)  # seeded starter example
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

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
    __table_args__ = (
        Index("ix_dashboard_widgets_dashboard", "dashboard_id"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    dashboard_id = Column(String(36), ForeignKey("dashboards.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False, default="Untitled")
    nl_prompt = Column(Text, nullable=True)
    sql = Column(Text, nullable=False, default="")
    chart_type = Column(String(20), nullable=False, default="table")
    chart_config = Column(JSON, nullable=True)  # {type, x_axis, y_axis}
    layout = Column(JSON, nullable=True)  # {x, y, w, h}
    position = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

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

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    connection_id = Column(String(36), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    nl_prompt = Column(Text, nullable=False)
    cron_expr = Column(String(100), nullable=False)
    raw_schedule_text = Column(Text, nullable=True)
    timezone = Column(String(50), nullable=False, default="UTC")
    is_paused = Column(Boolean, nullable=False, default=False)
    notify_email = Column(Boolean, nullable=False, default=True)
    notify_in_app = Column(Boolean, nullable=False, default=False)
    # 'always' | 'on_results' | 'on_change'
    notify_condition = Column(String(20), nullable=False, default="always")
    last_row_count = Column(Integer, nullable=True)
    last_result_hash = Column(String(64), nullable=True)
    next_run_at = Column(DateTime, nullable=True, index=True)
    last_run_at = Column(DateTime, nullable=True)
    last_status = Column(String(20), nullable=True)  # 'success' | 'failed' | 'running'
    consecutive_failures = Column(Integer, nullable=False, default=0)
    is_builtin = Column(Boolean, nullable=False, default=False)  # seeded starter example
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

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

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    schedule_id = Column(String(36), ForeignKey("scheduled_queries.id"), nullable=False, index=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False)  # 'success' | 'failed' | 'timeout'
    row_count = Column(Integer, nullable=True)
    generated_sql = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    notified = Column(Boolean, nullable=False, default=False)
    duration_ms = Column(Integer, nullable=True)

    schedule = relationship("ScheduledQuery", back_populates="runs")
