"""Workspace invitations (accept / decline).

Revision ID: 0022
Revises: 0021
Create Date: 2026-11-27

Adding someone to a workspace now creates a ``pending`` invitation instead of
a membership. The invitee sees it in the app (Team page + notification badge)
and accepts or declines; only acceptance creates the ``workspace_members``
row. Invitations live in their own table so existing membership-based access
checks keep their meaning unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    if "workspace_invitations" in insp.get_table_names():
        return
    op.create_table(
        "workspace_invitations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "invitee_user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("invited_by", sa.String(36), nullable=True),
        sa.Column("role", sa.String(10), nullable=False, server_default="member"),
        sa.Column("status", sa.String(10), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("responded_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_workspace_invitations_invitee_status",
        "workspace_invitations",
        ["invitee_user_id", "status"],
    )
    op.create_index(
        "ix_workspace_invitations_workspace_status",
        "workspace_invitations",
        ["workspace_id", "status"],
    )


def downgrade() -> None:
    insp = sa.inspect(op.get_bind())
    if "workspace_invitations" in insp.get_table_names():
        op.drop_table("workspace_invitations")
