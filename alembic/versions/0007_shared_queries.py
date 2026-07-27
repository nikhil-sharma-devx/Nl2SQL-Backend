"""Export & Share — shared_queries table.

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-19

Adds the ``shared_queries`` table backing the "Export and Share" feature: a
public, token-authed snapshot of a query + its (bounded) result set. Also
auto-created on startup by ensure_schema(), but per the migration policy every
schema change ships an Alembic migration so `alembic upgrade head` provisions
it in prod.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shared_queries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column("question", sa.Text(), nullable=False, server_default=""),
        sa.Column("nl_prompt", sa.Text(), nullable=True),
        sa.Column("generated_sql", sa.Text(), nullable=False, server_default=""),
        sa.Column("result_snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_shared_queries_user_id"), "shared_queries", ["user_id"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_shared_queries_user_id"), table_name="shared_queries")
    op.drop_table("shared_queries")
