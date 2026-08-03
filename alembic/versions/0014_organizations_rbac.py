"""Organizations, memberships & invitations — Teams/RBAC foundation.

Revision ID: 0014
Revises: 0013
Create Date: 2026-07-28

Adds the ``organizations``, ``organization_memberships``, and
``organization_invitations`` tables backing Phase 6a (Teams, Organizations &
RBAC). Every existing user is backfilled a personal (``is_personal=True``)
organization with an OWNER membership, via ``backfill_personal_organizations``
— this must run before migration ``0015`` adds ``organization_id`` to the
existing resource tables, since that backfill looks up each user's default
membership. Also auto-created going forward at registration
(``OrganizationService.bootstrap_personal_org``), per the migration policy
that every schema change ships an Alembic migration so ``alembic upgrade
head`` reproduces production exactly.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from nl_to_sql.infrastructure.database.migration_helpers import (
    backfill_personal_organizations,
)

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

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
            sa.Column("role", sa.String(length=20), nullable=False),
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

    backfill_personal_organizations(bind)


def downgrade() -> None:
    op.drop_index("ix_org_invitations_org_status", table_name="organization_invitations")
    op.drop_index("ix_org_invitations_email", table_name="organization_invitations")
    op.drop_table("organization_invitations")

    op.drop_index("ix_org_memberships_org_role", table_name="organization_memberships")
    op.drop_index("ix_org_memberships_user", table_name="organization_memberships")
    op.drop_table("organization_memberships")

    op.drop_index("ix_organizations_slug", table_name="organizations")
    op.drop_table("organizations")
