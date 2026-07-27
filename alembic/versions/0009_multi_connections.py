"""Multiple database connections per user.

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-20

Evolves the single-connection BYOD model into per-user *multiple* connections
and makes the schema catalog connection-scoped:

  - ``user_database_connections``: add ``connection_id`` (UUID, unique),
    ``name``, ``db_type``, ``is_default``; make ``encrypted_url`` nullable
    (NULL = the built-in "Server Default" connection); drop the ``UNIQUE(user_id)``
    single-row constraint and add ``UNIQUE(user_id, name)``.
  - ``user_schemas`` / ``user_schema_tables``: add ``connection_id`` and move
    uniqueness onto it (one catalog per connection).

A data backfill (``backfill_multi_connections``) then gives every legacy row a
``connection_id``, guarantees one default per user, creates a Server Default
connection for catalog-only users, and links catalog rows to their connection.

These tables are also provisioned by the ``ensure_schema`` backstop, but per the
migration policy every schema change ships a numbered migration so
``alembic upgrade head`` reproduces production exactly.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from nl_to_sql.infrastructure.database.migration_helpers import (
    backfill_multi_connections,
)

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_column(insp: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def _drop_constraint_if_exists(insp: sa.Inspector, table: str, name: str) -> None:
    uniques = {uc["name"] for uc in insp.get_unique_constraints(table)}
    if name in uniques:
        op.drop_constraint(name, table, type_="unique")


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    # ── user_database_connections ───────────────────────────────────────────────
    if "user_database_connections" not in tables:
        op.create_table(
            "user_database_connections",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("connection_id", sa.String(length=36), nullable=True),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False, server_default="Default"),
            sa.Column("db_type", sa.String(length=20), nullable=False, server_default="postgresql"),
            sa.Column("encrypted_url", sa.Text(), nullable=True),
            sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_user_database_connections_user", "user_database_connections", ["user_id"]
        )
    else:
        if not _has_column(insp, "user_database_connections", "connection_id"):
            op.add_column(
                "user_database_connections",
                sa.Column("connection_id", sa.String(length=36), nullable=True),
            )
        if not _has_column(insp, "user_database_connections", "name"):
            op.add_column(
                "user_database_connections",
                sa.Column("name", sa.String(length=200), nullable=False, server_default="Default"),
            )
        if not _has_column(insp, "user_database_connections", "db_type"):
            op.add_column(
                "user_database_connections",
                sa.Column("db_type", sa.String(length=20), nullable=False, server_default="postgresql"),
            )
        if not _has_column(insp, "user_database_connections", "is_default"):
            op.add_column(
                "user_database_connections",
                sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
            )
        # encrypted_url must allow NULL (Server Default connection).
        op.alter_column("user_database_connections", "encrypted_url", nullable=True)
        # Drop the legacy single-connection UNIQUE(user_id).
        _drop_constraint_if_exists(
            insp, "user_database_connections", "user_database_connections_user_id_key"
        )

    # ── user_schemas: add connection_id ─────────────────────────────────────────
    if "user_schemas" in tables and not _has_column(insp, "user_schemas", "connection_id"):
        op.add_column(
            "user_schemas", sa.Column("connection_id", sa.String(length=36), nullable=True)
        )

    # ── user_schema_tables: add connection_id ───────────────────────────────────
    if "user_schema_tables" in tables and not _has_column(
        insp, "user_schema_tables", "connection_id"
    ):
        op.add_column(
            "user_schema_tables", sa.Column("connection_id", sa.String(length=36), nullable=True)
        )

    # ── Data backfill (dialect-agnostic) ────────────────────────────────────────
    backfill_multi_connections(bind)

    # ── Constraints + indexes that require populated data ───────────────────────
    op.create_index(
        "ix_user_database_connections_connection_id",
        "user_database_connections",
        ["connection_id"],
        unique=True,
    )
    op.create_unique_constraint(
        "uq_user_connection_name", "user_database_connections", ["user_id", "name"]
    )

    if "user_schemas" in tables:
        _drop_constraint_if_exists(insp, "user_schemas", "user_schemas_user_id_key")
        op.create_unique_constraint(
            "uq_user_schema_connection", "user_schemas", ["connection_id"]
        )
        op.create_index(
            "ix_user_schemas_connection", "user_schemas", ["connection_id"]
        )

    if "user_schema_tables" in tables:
        _drop_constraint_if_exists(insp, "user_schema_tables", "uq_user_schema_table")
        op.create_unique_constraint(
            "uq_conn_schema_table",
            "user_schema_tables",
            ["connection_id", "schema_name", "table_name"],
        )
        op.create_index(
            "ix_user_schema_tables_connection", "user_schema_tables", ["connection_id"]
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "user_schema_tables" in tables:
        op.drop_index("ix_user_schema_tables_connection", table_name="user_schema_tables")
        _drop_constraint_if_exists(insp, "user_schema_tables", "uq_conn_schema_table")
        op.create_unique_constraint(
            "uq_user_schema_table",
            "user_schema_tables",
            ["user_id", "schema_name", "table_name"],
        )
        op.drop_column("user_schema_tables", "connection_id")

    if "user_schemas" in tables:
        op.drop_index("ix_user_schemas_connection", table_name="user_schemas")
        _drop_constraint_if_exists(insp, "user_schemas", "uq_user_schema_connection")
        op.create_unique_constraint("user_schemas_user_id_key", "user_schemas", ["user_id"])
        op.drop_column("user_schemas", "connection_id")

    op.drop_constraint(
        "uq_user_connection_name", "user_database_connections", type_="unique"
    )
    op.drop_index(
        "ix_user_database_connections_connection_id",
        table_name="user_database_connections",
    )
    # Revert to single-connection: keep only each user's default row.
    bind.execute(
        sa.text(
            "DELETE FROM user_database_connections WHERE is_default = :f"
        ),
        {"f": False},
    )
    op.create_unique_constraint(
        "user_database_connections_user_id_key",
        "user_database_connections",
        ["user_id"],
    )
    op.drop_column("user_database_connections", "is_default")
    op.drop_column("user_database_connections", "db_type")
    op.drop_column("user_database_connections", "name")
    op.drop_column("user_database_connections", "connection_id")
