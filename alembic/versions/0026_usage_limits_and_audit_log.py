"""Per-user daily query limits and the admin audit log.

Revision ID: 0026
Revises: 0025
Create Date: 2026-11-27

Adds:
  - ``users.daily_query_limit`` (int, nullable) — an admin-set daily AI-query
    cap; NULL falls back to DEFAULT_DAILY_QUERY_LIMIT (0 = unlimited).
  - ``user_daily_usage`` — (user_id, day) → query_count, the counter the limit
    is enforced against. Kept apart from chat history so deleting chats does
    not give the allowance back.
  - ``admin_audit_events`` — append-only log of admin actions.

Idempotent: the startup schema backstop may already have created these.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())

    if "daily_query_limit" not in {c["name"] for c in insp.get_columns("users")}:
        op.add_column("users", sa.Column("daily_query_limit", sa.Integer(), nullable=True))

    if not insp.has_table("user_daily_usage"):
        op.create_table(
            "user_daily_usage",
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("day", sa.Date(), primary_key=True),
            sa.Column("query_count", sa.Integer(), nullable=False, server_default="0"),
        )

    if not insp.has_table("admin_audit_events"):
        op.create_table(
            "admin_audit_events",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "actor_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("actor_email", sa.String(255), nullable=False),
            sa.Column("action", sa.String(40), nullable=False),
            sa.Column("target_type", sa.String(20), nullable=False),
            sa.Column("target_id", sa.String(64), nullable=True),
            sa.Column("target_label", sa.String(255), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
    insp = sa.inspect(op.get_bind())
    existing = {i["name"] for i in insp.get_indexes("admin_audit_events")}
    for name, cols in (
        ("ix_admin_audit_events_action", ["action"]),
        ("ix_admin_audit_events_created_at", ["created_at"]),
    ):
        if name not in existing:
            op.create_index(name, "admin_audit_events", cols)


def downgrade() -> None:
    op.drop_table("admin_audit_events")
    op.drop_table("user_daily_usage")
    op.drop_column("users", "daily_query_limit")
