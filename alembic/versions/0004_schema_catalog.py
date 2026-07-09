"""Schema catalog — per-user schema management tables.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-09

Adds the Schema Management catalog (source of truth for the Schema page):
  - user_schemas        — per-user catalog header (source, hash, raw upload).
  - user_schema_tables  — one row per table, columns stored as JSON.

These are also auto-created on startup by ensure_schema(), but per the migration
policy (Documentation/07_SCHEMA_MIGRATIONS.md) every new model ships an Alembic
migration so `alembic upgrade head` provisions them in production.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── user_schemas ───────────────────────────────────────────────────────────
    op.create_table(
        "user_schemas",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("database_name", sa.String(length=200), nullable=True),
        sa.Column("dialect", sa.String(length=50), nullable=False, server_default="postgresql"),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="reflected"),
        sa.Column("schema_hash", sa.String(length=64), nullable=True),
        sa.Column("raw_upload_json", sa.JSON(), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index("ix_user_schemas_user_id", "user_schemas", ["user_id"], if_not_exists=True)

    # ── user_schema_tables ──────────────────────────────────────────────────────
    op.create_table(
        "user_schema_tables",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("schema_name", sa.String(length=100), nullable=False, server_default="public"),
        sa.Column("table_name", sa.String(length=200), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="reflected"),
        sa.Column("columns", sa.JSON(), nullable=False),
        sa.Column("reflected_description", sa.Text(), nullable=True),
        sa.Column("user_description", sa.Text(), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("is_new", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "schema_name", "table_name", name="uq_user_schema_table"),
    )
    op.create_index(
        "ix_user_schema_tables_user", "user_schema_tables", ["user_id"], if_not_exists=True
    )


def downgrade() -> None:
    op.drop_index("ix_user_schema_tables_user", table_name="user_schema_tables")
    op.drop_table("user_schema_tables")
    op.drop_index("ix_user_schemas_user_id", table_name="user_schemas")
    op.drop_table("user_schemas")
