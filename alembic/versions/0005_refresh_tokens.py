"""Refresh tokens — rotating opaque tokens for short-lived access JWTs.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-13

Adds the ``refresh_tokens`` table backing refresh-token issuance + rotation.
Only the SHA-256 hash of each token is stored; rows are bound to a login
session so revoking the session also invalidates its refresh tokens.

Also auto-created on startup by ensure_schema(), but per the migration policy
(Documentation/07_SCHEMA_MIGRATIONS.md) every new model ships an Alembic
migration so `alembic upgrade head` provisions it in production.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=True),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["user_login_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_refresh_tokens_user", "refresh_tokens", ["user_id"], if_not_exists=True
    )
    op.create_index(
        "ix_refresh_tokens_session_id",
        "refresh_tokens",
        ["session_id"],
        if_not_exists=True,
    )
    # Unique index (matches the model's `unique=True, index=True` on token_hash,
    # which ensure_schema/create_all emits as a unique index of this name).
    op.create_index(
        "ix_refresh_tokens_token_hash",
        "refresh_tokens",
        ["token_hash"],
        unique=True,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_refresh_tokens_token_hash", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_session_id", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_user", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
