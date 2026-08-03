"""Add organization_id to every existing user-owned resource table.

Revision ID: 0015
Revises: 0014
Create Date: 2026-07-28

Adds a nullable ``organization_id`` (real FK to ``organizations.id``,
``ondelete=SET NULL``) plus a composite index to the 11 tables that were
previously scoped by ``user_id`` alone: ``user_database_connections``,
``metrics``, ``query_templates``, ``scheduled_queries``, ``dashboards``,
``saved_queries``, ``user_schemas``, ``user_schema_tables``,
``glossary_entries``, ``favorited_tables``, ``shared_queries``. ``user_id``
columns are left untouched — this is additive, not a replacement.

Unlike the loose ``connection_id`` convention used elsewhere in this schema,
``organization_id`` is a real FK: it is the tenancy boundary that later
row/column-security, audit-log, and billing phases will trust wholesale, so a
silently-stale value would defeat those systems with no DB-level guardrail.

Must run after ``0014`` (which backfills a personal organization + OWNER
membership for every existing user) — ``backfill_resource_organization_ids``
looks up each row's owner's default membership to fill the new column.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from nl_to_sql.infrastructure.database.migration_helpers import (
    backfill_resource_organization_ids,
)

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# table -> (fk constraint name, index spec). Index spec is either a single
# column name or a tuple for a composite index; the index name always ends
# in "_org" (single-column) or "_org_<col>" (composite), matching models.py.
_SIMPLE_ORG_INDEX: dict[str, str] = {
    "user_database_connections": "ix_user_database_connections_org",
    "metrics": "ix_metrics_org",
    "scheduled_queries": "ix_scheduled_queries_org",
    "user_schemas": "ix_user_schemas_org",
    "user_schema_tables": "ix_user_schema_tables_org",
    "glossary_entries": "ix_glossary_entries_org",
    "favorited_tables": "ix_favorited_tables_org",
    "shared_queries": "ix_shared_queries_org",
}

_COMPOSITE_ORG_INDEX: dict[str, tuple[str, str]] = {
    "query_templates": ("ix_query_templates_org_created", "created_at"),
    "dashboards": ("ix_dashboards_org_updated", "updated_at"),
    "saved_queries": ("ix_saved_queries_org_created", "created_at"),
}

_ALL_TABLES: tuple[str, ...] = tuple(_SIMPLE_ORG_INDEX) + tuple(_COMPOSITE_ORG_INDEX)


def _has_column(insp: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    for tbl in _ALL_TABLES:
        if tbl not in tables:
            continue
        if not _has_column(insp, tbl, "organization_id"):
            op.add_column(tbl, sa.Column("organization_id", sa.String(length=36), nullable=True))
            op.create_foreign_key(
                f"fk_{tbl}_organization_id",
                tbl,
                "organizations",
                ["organization_id"],
                ["id"],
                ondelete="SET NULL",
            )

        if tbl in _SIMPLE_ORG_INDEX:
            index_name = _SIMPLE_ORG_INDEX[tbl]
            if index_name not in {ix["name"] for ix in insp.get_indexes(tbl)}:
                op.create_index(index_name, tbl, ["organization_id"])
        else:
            index_name, second_col = _COMPOSITE_ORG_INDEX[tbl]
            if index_name not in {ix["name"] for ix in insp.get_indexes(tbl)}:
                op.create_index(index_name, tbl, ["organization_id", second_col])

    backfill_resource_organization_ids(bind)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    for tbl in _ALL_TABLES:
        if tbl not in tables:
            continue
        index_name = (
            _SIMPLE_ORG_INDEX[tbl] if tbl in _SIMPLE_ORG_INDEX else _COMPOSITE_ORG_INDEX[tbl][0]
        )
        if index_name in {ix["name"] for ix in insp.get_indexes(tbl)}:
            op.drop_index(index_name, table_name=tbl)
        if _has_column(insp, tbl, "organization_id"):
            op.drop_constraint(f"fk_{tbl}_organization_id", tbl, type_="foreignkey")
            op.drop_column(tbl, "organization_id")
