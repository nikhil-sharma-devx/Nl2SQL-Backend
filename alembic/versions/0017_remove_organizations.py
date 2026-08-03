"""Remove Organizations/Teams/RBAC (Phases 6a+6b) entirely.

Revision ID: 0017
Revises: 0016
Create Date: 2026-08-04

The organizations feature (multi-tenant orgs, memberships, RBAC roles,
invitations, join requests, custom roles, audit log) has been removed from
the application. This migration reverses everything ``0014``, ``0015``, and
``0016`` added:

  - Drops ``organization_id`` (column + FK + index) from the 11 resource
    tables it was additively bolted onto: ``user_database_connections``,
    ``metrics``, ``query_templates``, ``scheduled_queries``, ``dashboards``,
    ``saved_queries``, ``user_schemas``, ``user_schema_tables``,
    ``glossary_entries``, ``favorited_tables``, ``shared_queries``.
    ``user_id`` — the real ownership column all along — is untouched.
  - Drops ``audit_logs`` and ``custom_roles`` tables, the
    ``organization_memberships.status``/``custom_role_id`` columns, and
    ``organizations.allow_join_requests`` (all added by ``0016``).
  - Drops ``organization_invitations``, ``organization_memberships``, and
    ``organizations`` tables (added by ``0014``).

``downgrade()`` recreates the full schema (mirrors ``0014``/``0015``/``0016``
``upgrade()`` bodies) for completeness, but does not restore any data — those
tables would come back empty since the application no longer writes to them.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

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

_ALL_RESOURCE_TABLES: tuple[str, ...] = tuple(_SIMPLE_ORG_INDEX) + tuple(_COMPOSITE_ORG_INDEX)


def _has_column(insp: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def _has_table(insp: sa.Inspector, table: str) -> bool:
    return table in set(insp.get_table_names())


def _has_index(insp: sa.Inspector, table: str, index: str) -> bool:
    return index in {ix["name"] for ix in insp.get_indexes(table)}


def _drop_fk_on_column(insp: sa.Inspector, table: str, column: str) -> None:
    """Drop whatever FK constraint (if any) is on ``column``, by its real name.

    Doesn't assume a naming convention — the column may have been added by
    the ``ensure_schema()`` dev-backstop (bare ``ALTER TABLE ADD COLUMN``,
    no named FK to match) rather than by the Alembic migration that first
    introduced it, so the constraint may not exist under the expected name
    or at all.
    """
    for fk in insp.get_foreign_keys(table):
        if column in fk.get("constrained_columns", []) and fk.get("name"):
            op.drop_constraint(fk["name"], table, type_="foreignkey")


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # 1. Drop organization_id (+FK+index) from the 11 resource tables.
    for tbl in _ALL_RESOURCE_TABLES:
        if not _has_table(insp, tbl):
            continue
        index_name = (
            _SIMPLE_ORG_INDEX[tbl] if tbl in _SIMPLE_ORG_INDEX else _COMPOSITE_ORG_INDEX[tbl][0]
        )
        if index_name in {ix["name"] for ix in insp.get_indexes(tbl)}:
            op.drop_index(index_name, table_name=tbl)
        if _has_column(insp, tbl, "organization_id"):
            _drop_fk_on_column(insp, tbl, "organization_id")
            op.drop_column(tbl, "organization_id")

    # 2. audit_logs (references organizations + users).
    if _has_table(insp, "audit_logs"):
        op.drop_table("audit_logs")

    # 3. organization_memberships.custom_role_id / status (added by 0016).
    if _has_column(insp, "organization_memberships", "custom_role_id"):
        _drop_fk_on_column(insp, "organization_memberships", "custom_role_id")
        op.drop_column("organization_memberships", "custom_role_id")
    if _has_column(insp, "organization_memberships", "status"):
        if _has_index(insp, "organization_memberships", "ix_org_memberships_org_status"):
            op.drop_index("ix_org_memberships_org_status", table_name="organization_memberships")
        op.drop_column("organization_memberships", "status")

    # 4. custom_roles (now nothing references it).
    if _has_table(insp, "custom_roles"):
        if _has_index(insp, "custom_roles", "ix_custom_roles_org"):
            op.drop_index("ix_custom_roles_org", table_name="custom_roles")
        op.drop_table("custom_roles")

    # 5. organizations.allow_join_requests (added by 0016).
    if _has_column(insp, "organizations", "allow_join_requests"):
        op.drop_column("organizations", "allow_join_requests")

    # 6. organization_invitations (references organizations + users).
    if _has_table(insp, "organization_invitations"):
        if _has_index(insp, "organization_invitations", "ix_org_invitations_org_status"):
            op.drop_index("ix_org_invitations_org_status", table_name="organization_invitations")
        if _has_index(insp, "organization_invitations", "ix_org_invitations_email"):
            op.drop_index("ix_org_invitations_email", table_name="organization_invitations")
        op.drop_table("organization_invitations")

    # 7. organization_memberships (references organizations + users).
    if _has_table(insp, "organization_memberships"):
        if _has_index(insp, "organization_memberships", "ix_org_memberships_org_role"):
            op.drop_index("ix_org_memberships_org_role", table_name="organization_memberships")
        if _has_index(insp, "organization_memberships", "ix_org_memberships_user"):
            op.drop_index("ix_org_memberships_user", table_name="organization_memberships")
        op.drop_table("organization_memberships")

    # 8. organizations (nothing references it anymore).
    if _has_table(insp, "organizations"):
        if _has_index(insp, "organizations", "ix_organizations_slug"):
            op.drop_index("ix_organizations_slug", table_name="organizations")
        op.drop_table("organizations")


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    # Recreate organizations, organization_memberships, organization_invitations (0014).
    if "organizations" not in tables:
        op.create_table(
            "organizations",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("slug", sa.String(length=200), nullable=False),
            sa.Column("is_personal", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("owner_user_id", sa.String(length=36), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("slug", name="uq_organizations_slug"),
        )
        op.create_index("ix_organizations_slug", "organizations", ["slug"])

    if "organization_memberships" not in tables:
        op.create_table(
            "organization_memberships",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("role", sa.String(length=20), nullable=True),
            sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("organization_id", "user_id", name="uq_org_membership"),
        )
        op.create_index("ix_org_memberships_user", "organization_memberships", ["user_id"])
        op.create_index(
            "ix_org_memberships_org_role", "organization_memberships", ["organization_id", "role"]
        )

    if "organization_invitations" not in tables:
        op.create_table(
            "organization_invitations",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("email", sa.String(length=255), nullable=False),
            sa.Column("role", sa.String(length=20), nullable=False),
            sa.Column("invited_by_user_id", sa.String(length=36), nullable=False),
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("accepted_at", sa.DateTime(), nullable=True),
            sa.Column("accepted_by_user_id", sa.String(length=36), nullable=True),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.ForeignKeyConstraint(["invited_by_user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["accepted_by_user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("token_hash", name="uq_org_invitations_token_hash"),
        )
        op.create_index("ix_org_invitations_email", "organization_invitations", ["email"])
        op.create_index(
            "ix_org_invitations_org_status", "organization_invitations", ["organization_id", "status"]
        )

    # Recreate organizations.allow_join_requests, custom_roles,
    # organization_memberships.status/custom_role_id, audit_logs (0016).
    insp = sa.inspect(bind)
    if not _has_column(insp, "organizations", "allow_join_requests"):
        op.add_column(
            "organizations",
            sa.Column(
                "allow_join_requests", sa.Boolean(), nullable=False, server_default=sa.false()
            ),
        )

    if not _has_table(insp, "custom_roles"):
        op.create_table(
            "custom_roles",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "organization_id",
                sa.String(length=36),
                sa.ForeignKey("organizations.id"),
                nullable=False,
            ),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("permissions", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("organization_id", "name", name="uq_custom_role_org_name"),
        )
        op.create_index("ix_custom_roles_org", "custom_roles", ["organization_id"])

    insp = sa.inspect(bind)

    if not _has_column(insp, "organization_memberships", "status"):
        op.add_column(
            "organization_memberships",
            sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        )
        op.create_index(
            "ix_org_memberships_org_status",
            "organization_memberships",
            ["organization_id", "status"],
        )

    if not _has_column(insp, "organization_memberships", "custom_role_id"):
        op.add_column(
            "organization_memberships",
            sa.Column("custom_role_id", sa.String(length=36), nullable=True),
        )
        op.create_foreign_key(
            "fk_org_memberships_custom_role_id",
            "organization_memberships",
            "custom_roles",
            ["custom_role_id"],
            ["id"],
        )

    if not _has_table(insp, "audit_logs"):
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "organization_id",
                sa.String(length=36),
                sa.ForeignKey("organizations.id"),
                nullable=False,
            ),
            sa.Column("actor_user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("action", sa.String(length=50), nullable=False),
            sa.Column("target_type", sa.String(length=50), nullable=True),
            sa.Column("target_id", sa.String(length=64), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_audit_logs_org", "audit_logs", ["organization_id"])
        op.create_index("ix_audit_logs_org_created", "audit_logs", ["organization_id", "created_at"])

    # Recreate organization_id on the 11 resource tables (0015).
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())
    for tbl in _ALL_RESOURCE_TABLES:
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
