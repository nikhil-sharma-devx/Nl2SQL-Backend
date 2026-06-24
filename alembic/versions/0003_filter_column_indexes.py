"""Add indexes on high-frequency filter columns.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-23

chat_messages.deleted_at — filtered on every clear_history, export, and list query.
account_deletions.status — polled on every purge worker run.
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_chat_messages_deleted_at",
        "chat_messages",
        ["deleted_at"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_account_deletions_status",
        "account_deletions",
        ["status"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_account_deletions_status", table_name="account_deletions")
    op.drop_index("ix_chat_messages_deleted_at", table_name="chat_messages")
