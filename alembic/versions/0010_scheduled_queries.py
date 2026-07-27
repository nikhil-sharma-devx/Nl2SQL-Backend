"""Scheduled Queries & Alerts — scheduled_queries + scheduled_query_runs tables.

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-27

Adds the ``scheduled_queries`` and ``scheduled_query_runs`` tables backing the
"Scheduled Queries & Alerts" feature: users schedule a natural-language query
on a cron-like cadence against a specific connection, with email alerting and
bounded execution history. Also auto-created on startup by ensure_schema(),
but per the migration policy every schema change ships an Alembic migration so
``alembic upgrade head`` provisions it in prod.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "scheduled_queries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("connection_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("nl_prompt", sa.Text(), nullable=False),
        sa.Column("cron_expr", sa.String(length=100), nullable=False),
        sa.Column("raw_schedule_text", sa.Text(), nullable=True),
        sa.Column("timezone", sa.String(length=50), nullable=False, server_default="UTC"),
        sa.Column("is_paused", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notify_email", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("notify_in_app", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notify_condition", sa.String(length=20), nullable=False, server_default="always"),
        sa.Column("last_row_count", sa.Integer(), nullable=True),
        sa.Column("last_result_hash", sa.String(length=64), nullable=True),
        sa.Column("next_run_at", sa.DateTime(), nullable=True),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column("last_status", sa.String(length=20), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_scheduled_queries_user", "scheduled_queries", ["user_id"])
    op.create_index("ix_scheduled_queries_connection", "scheduled_queries", ["connection_id"])
    op.create_index(
        "ix_scheduled_queries_next_run", "scheduled_queries", ["next_run_at", "is_paused"]
    )

    op.create_table(
        "scheduled_query_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("schedule_id", sa.String(length=36), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("generated_sql", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("notified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["schedule_id"], ["scheduled_queries.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_scheduled_query_runs_schedule_id", "scheduled_query_runs", ["schedule_id"]
    )
    op.create_index(
        "ix_scheduled_query_runs_schedule_started",
        "scheduled_query_runs",
        ["schedule_id", "started_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_scheduled_query_runs_schedule_started", table_name="scheduled_query_runs"
    )
    op.drop_index(
        "ix_scheduled_query_runs_schedule_id", table_name="scheduled_query_runs"
    )
    op.drop_table("scheduled_query_runs")

    op.drop_index("ix_scheduled_queries_next_run", table_name="scheduled_queries")
    op.drop_index("ix_scheduled_queries_connection", table_name="scheduled_queries")
    op.drop_index("ix_scheduled_queries_user", table_name="scheduled_queries")
    op.drop_table("scheduled_queries")
