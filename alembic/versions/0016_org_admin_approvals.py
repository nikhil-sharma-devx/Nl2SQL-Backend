"""Organization Administration & User Approval System (Phase 6b).

Revision ID: 0016
Revises: 0015
Create Date: 2026-07-28

Adds:
  - ``organizations.allow_join_requests`` (bool, default False) — opt-in gate
    for the new join-by-slug request flow.
  - ``custom_roles`` table — org-scoped, admin-configurable permission sets
    (``permissions`` stored as a JSON list, not a join table).
  - ``organization_memberships.status`` (active|pending|suspended, default
    "active" so every existing row is backfilled as active) and
    ``custom_role_id`` (nullable FK). ``role`` becomes nullable — a pending
    membership has none yet.
  - ``audit_logs`` table — append-only, indexed by ``(organization_id,
    created_at)`` for the new audit-log viewer.

All existing rows are unaffected: ``allow_join_requests`` defaults False and
``status`` defaults "active", so pre-migration organizations/memberships keep
behaving exactly as they do today.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_column(insp: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def _has_table(insp: sa.Inspector, table: str) -> bool:
    return table in set(insp.get_table_names())


def upgrade() -> None:
    bind = op.get_bind()
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

    insp = sa.inspect(bind)  # refresh after custom_roles create (FK target for the next column)

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

    _make_role_nullable(bind, insp)

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


def _make_role_nullable(bind: sa.engine.Connection, insp: sa.Inspector) -> None:
    """SQLite (test/dev) can't ALTER COLUMN in place; Postgres (prod) can."""
    role_col = next(c for c in insp.get_columns("organization_memberships") if c["name"] == "role")
    if role_col["nullable"]:
        return
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("organization_memberships") as batch:
            batch.alter_column("role", existing_type=sa.String(length=20), nullable=True)
    else:
        op.alter_column(
            "organization_memberships", "role", existing_type=sa.String(length=20), nullable=True
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if _has_table(insp, "audit_logs"):
        op.drop_table("audit_logs")

    if _has_column(insp, "organization_memberships", "custom_role_id"):
        op.drop_constraint(
            "fk_org_memberships_custom_role_id", "organization_memberships", type_="foreignkey"
        )
        op.drop_column("organization_memberships", "custom_role_id")

    if _has_column(insp, "organization_memberships", "status"):
        op.drop_index("ix_org_memberships_org_status", table_name="organization_memberships")
        op.drop_column("organization_memberships", "status")

    if _has_table(insp, "custom_roles"):
        op.drop_index("ix_custom_roles_org", table_name="custom_roles")
        op.drop_table("custom_roles")

    if _has_column(insp, "organizations", "allow_join_requests"):
        op.drop_column("organizations", "allow_join_requests")
