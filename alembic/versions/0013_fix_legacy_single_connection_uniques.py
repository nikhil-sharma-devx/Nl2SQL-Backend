"""Fix legacy single-connection UNIQUE indexes left over from before 0009.

Revision ID: 0013
Revises: 0012
Create Date: 2026-07-28

``0009`` intended to move ``user_schemas``/``user_schema_tables``/
``user_database_connections`` from "one row per user" to "one row per
connection", dropping the old single-column uniques via
``_drop_constraint_if_exists``. That helper only inspects
``get_unique_constraints()`` (pg_constraint entries). On databases where the
old uniqueness was implemented as a bare ``UNIQUE INDEX`` (e.g.
``Column(unique=True, index=True)`` from the pre-multi-connection model,
provisioned by the ``ensure_schema()`` dev backstop before Alembic caught up),
that index has no pg_constraint entry and was silently left in place — so
``0009``'s replacement constraints were also never created (each guarded by a
plain ``if "table" in tables`` with no per-object existence check, so a
partial/no-op run still advanced cleanly).

Net effect on affected databases: syncing a second connection's schema catalog
(or creating a second connection at all) 500s with
``duplicate key value violates unique constraint "ix_..._user_id"`` even
though the app-layer code is already connection-scoped.

This migration is fully idempotent: it inspects live indexes/constraints by
column signature (not by assumed name) before dropping or creating anything,
so it is a no-op on databases where 0009 already completed correctly.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from nl_to_sql.infrastructure.database.migration_helpers import (
    backfill_multi_connections,
)

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _drop_unique_on(insp: sa.Inspector, table: str, columns: list[str]) -> None:
    """Drop whatever unique constraint or unique index sits on exactly ``columns``."""
    constraint_names = {
        uc["name"]
        for uc in insp.get_unique_constraints(table)
        if uc["column_names"] == columns
    }
    for name in constraint_names:
        op.drop_constraint(name, table, type_="unique")

    index_names = {
        ix["name"]
        for ix in insp.get_indexes(table)
        if ix.get("unique") and ix["column_names"] == columns and ix["name"] not in constraint_names
    }
    for name in index_names:
        op.drop_index(name, table_name=table)


def _ensure_unique(insp: sa.Inspector, table: str, columns: list[str], name: str) -> None:
    if name not in {uc["name"] for uc in insp.get_unique_constraints(table)}:
        op.create_unique_constraint(name, table, columns)


def _ensure_index(
    insp: sa.Inspector, table: str, columns: list[str], name: str, unique: bool = False
) -> None:
    if name not in {ix["name"] for ix in insp.get_indexes(table)}:
        op.create_index(name, table, columns, unique=unique)


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "user_database_connections" in tables:
        _drop_unique_on(insp, "user_database_connections", ["user_id"])

    if "user_schemas" in tables:
        _drop_unique_on(insp, "user_schemas", ["user_id"])

    if "user_schema_tables" in tables:
        _drop_unique_on(
            insp, "user_schema_tables", ["user_id", "schema_name", "table_name"]
        )

    # Re-run the (idempotent) backfill in case connection_id was never
    # stamped because 0009's own DDL silently no-op'd on this database.
    backfill_multi_connections(bind)

    # Re-inspect: the backfill/DDL above may have changed what's reflected.
    insp = sa.inspect(bind)

    if "user_database_connections" in tables:
        _ensure_unique(
            insp, "user_database_connections", ["user_id", "name"], "uq_user_connection_name"
        )
        _ensure_index(
            insp,
            "user_database_connections",
            ["connection_id"],
            "ix_user_database_connections_connection_id",
            unique=True,
        )

    if "user_schemas" in tables:
        _ensure_unique(insp, "user_schemas", ["connection_id"], "uq_user_schema_connection")
        _ensure_index(insp, "user_schemas", ["connection_id"], "ix_user_schemas_connection")

    if "user_schema_tables" in tables:
        _ensure_unique(
            insp,
            "user_schema_tables",
            ["connection_id", "schema_name", "table_name"],
            "uq_conn_schema_table",
        )
        _ensure_index(
            insp,
            "user_schema_tables",
            ["connection_id"],
            "ix_user_schema_tables_connection",
        )


def downgrade() -> None:
    """Best-effort reversal. Fails loudly (FK/unique violation) if any user by
    now has more than one connection or catalog row — that data genuinely
    cannot be represented under the old one-row-per-user shape.
    """
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "user_schema_tables" in tables:
        _drop_unique_on(
            insp, "user_schema_tables", ["connection_id", "schema_name", "table_name"]
        )
        _ensure_unique(
            insp,
            "user_schema_tables",
            ["user_id", "schema_name", "table_name"],
            "uq_user_schema_table",
        )

    if "user_schemas" in tables:
        _drop_unique_on(insp, "user_schemas", ["connection_id"])
        _ensure_unique(insp, "user_schemas", ["user_id"], "uq_user_schema_user")

    if "user_database_connections" in tables:
        _drop_unique_on(insp, "user_database_connections", ["user_id", "name"])
        _ensure_unique(
            insp, "user_database_connections", ["user_id"], "uq_user_database_connections_user"
        )
