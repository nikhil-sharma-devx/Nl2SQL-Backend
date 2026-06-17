"""Performance indexes — add missing indexes across all high-traffic tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-17

Adds the indexes defined in models.py __table_args__ and index=True columns
that were missing from the initial schema migration.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # if_not_exists=True makes every CREATE INDEX idempotent — safe to run even
    # when create_all() already created some of these indexes on an existing DB.

    # ── chat_messages ──────────────────────────────────────────────────────────
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"], if_not_exists=True)
    op.create_index("ix_chat_messages_timestamp", "chat_messages", ["timestamp"], if_not_exists=True)
    op.create_index("ix_chat_messages_session_ts", "chat_messages", ["session_id", "timestamp"], if_not_exists=True)

    # ── chat_sessions ──────────────────────────────────────────────────────────
    op.create_index("ix_chat_sessions_updated_at", "chat_sessions", ["updated_at"], if_not_exists=True)
    op.create_index("ix_chat_sessions_user_updated", "chat_sessions", ["user_id", "updated_at"], if_not_exists=True)

    # ── query_history ──────────────────────────────────────────────────────────
    op.create_index("ix_query_history_timestamp", "query_history", ["timestamp"], if_not_exists=True)

    # ── feedback ───────────────────────────────────────────────────────────────
    op.create_index("ix_feedback_query_id", "feedback", ["query_id"], if_not_exists=True)
    op.create_index("ix_feedback_session_id", "feedback", ["session_id"], if_not_exists=True)
    op.create_index("ix_feedback_timestamp", "feedback", ["timestamp"], if_not_exists=True)
    op.create_index("ix_feedback_ts_type", "feedback", ["timestamp", "feedback_type"], if_not_exists=True)

    # ── training_data ──────────────────────────────────────────────────────────
    op.create_index("ix_training_data_used_for_training", "training_data", ["used_for_training"], if_not_exists=True)
    op.create_index("ix_training_data_intent_type", "training_data", ["intent_type"], if_not_exists=True)
    op.create_index("ix_training_data_used_created", "training_data", ["used_for_training", "created_at"], if_not_exists=True)

    # ── query_metrics ──────────────────────────────────────────────────────────
    op.create_index("ix_query_metrics_user_created", "query_metrics", ["user_id", "created_at"], if_not_exists=True)

    # ── data_export_jobs ───────────────────────────────────────────────────────
    op.create_index("ix_data_export_jobs_user_status", "data_export_jobs", ["user_id", "status"], if_not_exists=True)

    # ── user_login_sessions ────────────────────────────────────────────────────
    op.create_index("ix_login_sessions_user_revoked", "user_login_sessions", ["user_id", "revoked_at"], if_not_exists=True)
    op.create_index("ix_login_sessions_user_active", "user_login_sessions", ["user_id", "last_active_at"], if_not_exists=True)

    # ── login_events ───────────────────────────────────────────────────────────
    op.create_index("ix_login_events_user_created", "login_events", ["user_id", "created_at"], if_not_exists=True)
    op.create_index("ix_login_events_outcome_created", "login_events", ["outcome", "created_at"], if_not_exists=True)


def downgrade() -> None:
    # ── login_events ───────────────────────────────────────────────────────────
    op.drop_index("ix_login_events_outcome_created", table_name="login_events")
    op.drop_index("ix_login_events_user_created", table_name="login_events")

    # ── user_login_sessions ────────────────────────────────────────────────────
    op.drop_index("ix_login_sessions_user_active", table_name="user_login_sessions")
    op.drop_index("ix_login_sessions_user_revoked", table_name="user_login_sessions")

    # ── data_export_jobs ───────────────────────────────────────────────────────
    op.drop_index("ix_data_export_jobs_user_status", table_name="data_export_jobs")

    # ── query_metrics ──────────────────────────────────────────────────────────
    op.drop_index("ix_query_metrics_user_created", table_name="query_metrics")

    # ── training_data ──────────────────────────────────────────────────────────
    op.drop_index("ix_training_data_used_created", table_name="training_data")
    op.drop_index("ix_training_data_intent_type", table_name="training_data")
    op.drop_index("ix_training_data_used_for_training", table_name="training_data")

    # ── feedback ───────────────────────────────────────────────────────────────
    op.drop_index("ix_feedback_ts_type", table_name="feedback")
    op.drop_index("ix_feedback_timestamp", table_name="feedback")
    op.drop_index("ix_feedback_session_id", table_name="feedback")
    op.drop_index("ix_feedback_query_id", table_name="feedback")

    # ── query_history ──────────────────────────────────────────────────────────
    op.drop_index("ix_query_history_timestamp", table_name="query_history")

    # ── chat_sessions ──────────────────────────────────────────────────────────
    op.drop_index("ix_chat_sessions_user_updated", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_updated_at", table_name="chat_sessions")

    # ── chat_messages ──────────────────────────────────────────────────────────
    op.drop_index("ix_chat_messages_session_ts", table_name="chat_messages")
    op.drop_index("ix_chat_messages_timestamp", table_name="chat_messages")
    op.drop_index("ix_chat_messages_session_id", table_name="chat_messages")
