"""Notification digest — add last_digest_sent_at cadence guard.

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-18

Adds the ``last_digest_sent_at`` column to ``notification_preferences`` so the
email-digest worker can enforce a per-user send cadence. Also auto-created on
startup by ensure_schema(), but per the migration policy every schema change
ships an Alembic migration so `alembic upgrade head` provisions it in prod.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "notification_preferences",
        sa.Column("last_digest_sent_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("notification_preferences", "last_digest_sent_at")
