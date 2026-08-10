"""Scope training_data to a tenant (C8).

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-11

``get_recent_examples()`` (few-shot context injected into the SQL-generation
prompt when ``rag_few_shot_retrieval_enabled``) had no ``user_id``/
``connection_id`` filter — it pulled the globally most-recent high-score
examples across every tenant and every connection, leaking one tenant's
Q/SQL pairs into another tenant's prompt (and biasing generation toward
tables that don't exist in the current connection).

Adds ``user_id`` (indexed) and ``connection_id`` to ``training_data``, both
nullable — existing rows and rows collected outside an authenticated request
have no owner and are treated as shared (visible to every connection),
mirroring the same own-or-shared model already used by the Qdrant schema
store and the few-shot example store. No backfill is possible (or
necessary): historical rows were never tied to a tenant, so they simply stay
in the shared pool.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_column(insp: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def _has_index(insp: sa.Inspector, table: str, index: str) -> bool:
    return index in {ix["name"] for ix in insp.get_indexes(table)}


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if not _has_column(insp, "training_data", "user_id"):
        op.add_column("training_data", sa.Column("user_id", sa.String(length=36), nullable=True))
    if not _has_index(insp, "training_data", "ix_training_data_user_id"):
        op.create_index("ix_training_data_user_id", "training_data", ["user_id"])

    if not _has_column(insp, "training_data", "connection_id"):
        op.add_column(
            "training_data", sa.Column("connection_id", sa.String(length=36), nullable=True)
        )
    if not _has_index(insp, "training_data", "ix_training_data_connection"):
        op.create_index("ix_training_data_connection", "training_data", ["connection_id"])


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if _has_index(insp, "training_data", "ix_training_data_connection"):
        op.drop_index("ix_training_data_connection", table_name="training_data")
    if _has_column(insp, "training_data", "connection_id"):
        op.drop_column("training_data", "connection_id")

    if _has_index(insp, "training_data", "ix_training_data_user_id"):
        op.drop_index("ix_training_data_user_id", table_name="training_data")
    if _has_column(insp, "training_data", "user_id"):
        op.drop_column("training_data", "user_id")
