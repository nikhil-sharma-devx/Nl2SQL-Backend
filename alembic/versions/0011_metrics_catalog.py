"""Semantic Layer / Metrics Catalog — metrics table.

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-27

Adds the ``metrics`` table backing the "Semantic Layer / Metrics Catalog"
feature: users define governed business metrics (name, SQL definition,
dimensions, tags, certified flag) scoped to a DB connection. Only
``certified=True`` rows are injected into the SQL-generation prompt. Also
auto-created on startup by ensure_schema(), but per the migration policy every
schema change ships an Alembic migration so ``alembic upgrade head``
provisions it in prod.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("metric_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("connection_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("sql_definition", sa.Text(), nullable=False),
        sa.Column("dimensions", sa.JSON(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("owner", sa.String(length=200), nullable=True),
        sa.Column("certified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_metrics_metric_id"), "metrics", ["metric_id"], unique=True)
    op.create_index("ix_metrics_connection", "metrics", ["connection_id"])
    op.create_index("ix_metrics_user", "metrics", ["user_id"])
    op.create_unique_constraint("uq_connection_metric_name", "metrics", ["connection_id", "name"])


def downgrade() -> None:
    op.drop_constraint("uq_connection_metric_name", "metrics", type_="unique")
    op.drop_index("ix_metrics_user", table_name="metrics")
    op.drop_index("ix_metrics_connection", table_name="metrics")
    op.drop_index(op.f("ix_metrics_metric_id"), table_name="metrics")
    op.drop_table("metrics")
