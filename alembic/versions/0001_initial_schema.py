"""Initial schema — all Phase 0 + Phase 1 tables.

Revision ID: 0001
Revises:
Create Date: 2026-06-17

Run on a fresh database: alembic upgrade head
Existing databases (using create_all at startup): alembic stamp head
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("full_name", sa.String(200), nullable=True),
        sa.Column("hashed_password", sa.Text, nullable=True),
        sa.Column("auth_provider", sa.String(20), nullable=False, server_default="email"),
        sa.Column("google_sub", sa.String(255), nullable=True, unique=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("otp_code", sa.String(6), nullable=True),
        sa.Column("otp_expires_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "password_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "user_api_keys",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("encrypted_key", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "provider", name="uq_user_api_key_provider"),
    )

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True, index=True),
        sa.Column("title", sa.String(200), nullable=False, server_default="New Chat"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(36), sa.ForeignKey("chat_sessions.id"), nullable=False),
        sa.Column("timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("sql", sa.Text, nullable=False, server_default=""),
        sa.Column("dialect", sa.String(50), nullable=False),
        sa.Column("is_valid", sa.Boolean, nullable=False),
        sa.Column("validation_errors", sa.JSON, nullable=True),
        sa.Column("retrieved_tables", sa.JSON, nullable=True),
        sa.Column("execution_result", sa.JSON, nullable=True),
        sa.Column("execution_error", sa.Text, nullable=True),
        sa.Column("tokens_used", sa.Integer, server_default="0"),
        sa.Column("cached", sa.Boolean, server_default="false"),
        sa.Column("message", sa.Text, nullable=True),
        sa.Column("used_tables", sa.JSON, nullable=True),
        sa.Column("suggested_chart", sa.JSON, nullable=True),
        sa.Column("follow_up_questions", sa.JSON, nullable=True),
        sa.Column("intent_type", sa.String(50), nullable=True),
        sa.Column("query_complexity", sa.Integer, nullable=True),
        sa.Column("prompt_version", sa.String(20), nullable=True),
        sa.Column("retrieval_method", sa.String(20), nullable=True),
        sa.Column("response_time_ms", sa.Integer, nullable=True),
        sa.Column("deleted_at", sa.DateTime, nullable=True),
    )

    op.create_table(
        "query_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("sql", sa.Text, nullable=False, server_default=""),
        sa.Column("dialect", sa.String(50), nullable=False),
        sa.Column("is_valid", sa.Boolean, nullable=False),
        sa.Column("validation_errors", sa.JSON, nullable=True),
        sa.Column("retrieved_tables", sa.JSON, nullable=True),
        sa.Column("execution_result", sa.JSON, nullable=True),
        sa.Column("tokens_used", sa.Integer, server_default="0"),
        sa.Column("cached", sa.Boolean, server_default="false"),
        sa.Column("execution_error", sa.Text, nullable=True),
        sa.Column("message", sa.Text, nullable=True),
        sa.Column("intent_type", sa.String(50), nullable=True),
        sa.Column("query_complexity", sa.Integer, nullable=True),
        sa.Column("prompt_version", sa.String(20), nullable=True),
        sa.Column("retrieval_method", sa.String(20), nullable=True),
        sa.Column("response_time_ms", sa.Integer, nullable=True),
        sa.Column("user_feedback", sa.String(20), nullable=True),
        sa.Column("corrected_sql", sa.Text, nullable=True),
    )

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("query_id", sa.Integer, sa.ForeignKey("query_history.id"), nullable=True),
        sa.Column("session_id", sa.String(36), nullable=True),
        sa.Column("feedback_type", sa.String(20), nullable=False),
        sa.Column("feedback_data", sa.JSON, nullable=True),
        sa.Column("timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "sql_versions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("message_id", sa.Integer, sa.ForeignKey("chat_messages.id"), nullable=False, index=True),
        sa.Column("version_number", sa.Integer, nullable=False),
        sa.Column("sql", sa.Text, nullable=False),
        sa.Column("results", sa.JSON, nullable=True),
        sa.Column("success", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("is_original", sa.Boolean, nullable=False, server_default="false"),
    )

    op.create_table(
        "training_data",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("sql", sa.Text, nullable=False),
        sa.Column("retrieved_tables", sa.JSON, nullable=True),
        sa.Column("schema_context", sa.Text, nullable=True),
        sa.Column("success_score", sa.Float, nullable=True),
        sa.Column("intent_type", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("used_for_training", sa.Boolean, server_default="false"),
    )

    # ── Phase 1 tables ─────────────────────────────────────────────────────────

    op.create_table(
        "user_instructions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("char_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "user_instructions_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column("replaced_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "user_settings",
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("sql_keyword_case", sa.String(10), nullable=False, server_default="upper"),
        sa.Column("sql_cte_pref", sa.String(10), nullable=False, server_default="cte"),
        sa.Column("sql_alias_style", sa.String(10), nullable=False, server_default="as"),
        sa.Column("sql_indent", sa.Integer, nullable=False, server_default="2"),
        sa.Column("default_dialect", sa.String(50), nullable=True),
        sa.Column("max_result_rows", sa.Integer, nullable=False, server_default="1000"),
        sa.Column("auto_execute", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("default_model", sa.String(100), nullable=True),
        sa.Column("data_retention", sa.String(10), nullable=False, server_default="forever"),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "saved_queries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("nl_prompt", sa.Text, nullable=False),
        sa.Column("generated_sql", sa.Text, nullable=False),
        sa.Column("dialect", sa.String(50), nullable=True),
        sa.Column("starred", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("last_run_at", sa.DateTime, nullable=True),
        sa.Column("run_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "query_metrics",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True, index=True),
        sa.Column("query_id", sa.String(36), nullable=True),
        sa.Column("tokens_in", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=True),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "data_export_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("artifact_path", sa.Text, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )

    op.create_table(
        "account_deletions",
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("requested_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("purge_after", sa.DateTime, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="scheduled"),
    )

    op.create_table(
        "user_login_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("device", sa.String(200), nullable=True),
        sa.Column("browser", sa.String(200), nullable=True),
        sa.Column("ip", sa.String(50), nullable=True),
        sa.Column("last_active_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("revoked_at", sa.DateTime, nullable=True),
    )

    op.create_table(
        "login_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("ip", sa.String(50), nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("outcome", sa.String(20), nullable=False, server_default="success"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    # Drop in reverse FK order
    op.drop_table("login_events")
    op.drop_table("user_login_sessions")
    op.drop_table("account_deletions")
    op.drop_table("data_export_jobs")
    op.drop_table("query_metrics")
    op.drop_table("saved_queries")
    op.drop_table("user_settings")
    op.drop_table("user_instructions_history")
    op.drop_table("user_instructions")
    op.drop_table("training_data")
    op.drop_table("sql_versions")
    op.drop_table("feedback")
    op.drop_table("query_history")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("user_api_keys")
    op.drop_table("password_history")
    op.drop_table("users")
