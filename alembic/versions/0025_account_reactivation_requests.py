"""Reactivation requests from deactivated accounts.

Revision ID: 0025
Revises: 0024
Create Date: 2026-11-27

Adds ``account_reactivation_requests``: a deactivated user who signs in with
valid credentials can ask the admins to reactivate them; an admin approves
(reactivates the account) or rejects it from the Admin page. A partial unique
index allows at most one ``pending`` request per user.

Idempotent like 0024: the startup schema backstop may already have created
the table if the app ran before this migration.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "account_reactivation_requests"


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    if not insp.has_table(_TABLE):
        op.create_table(
            _TABLE,
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("status", sa.String(10), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("decided_at", sa.DateTime(), nullable=True),
            sa.Column(
                "decided_by",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
        insp = sa.inspect(op.get_bind())
    existing = {i["name"] for i in insp.get_indexes(_TABLE)}
    if "ix_account_reactivation_requests_user_id" not in existing:
        op.create_index("ix_account_reactivation_requests_user_id", _TABLE, ["user_id"])
    if "ix_reactivation_requests_status_created" not in existing:
        op.create_index("ix_reactivation_requests_status_created", _TABLE, ["status", "created_at"])
    if "uq_reactivation_requests_one_pending" not in existing:
        op.create_index(
            "uq_reactivation_requests_one_pending",
            _TABLE,
            ["user_id"],
            unique=True,
            postgresql_where=sa.text("status = 'pending'"),
            sqlite_where=sa.text("status = 'pending'"),
        )


def downgrade() -> None:
    op.drop_table(_TABLE)
