"""Store platform admin status on ``users`` (replaces the ADMIN_EMAILS setting).

Revision ID: 0024
Revises: 0023
Create Date: 2026-11-27

Adds:
  - ``users.is_admin`` (bool, default False) — may open /admin and grant admin.
  - ``users.is_owner`` (bool, default False) — the single account that may
    also revoke admin. ``uq_users_single_owner`` (partial unique index on
    ``is_owner`` WHERE true) guarantees there is never more than one.

No rows are promoted here: every account starts as a normal user, and the
first owner is claimed once through the setup page (``POST
/api/v1/admin/setup`` with ``ADMIN_SETUP_TOKEN``). The old ``ADMIN_EMAILS``
setting is no longer read.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Idempotent: the startup schema backstop (ensure_schema) may already have
    # added the columns if the app ran before this migration did.
    insp = sa.inspect(op.get_bind())
    columns = {c["name"] for c in insp.get_columns("users")}
    for name in ("is_admin", "is_owner"):
        if name not in columns:
            op.add_column(
                "users",
                sa.Column(name, sa.Boolean(), nullable=False, server_default=sa.false()),
            )
    if "uq_users_single_owner" not in {i["name"] for i in insp.get_indexes("users")}:
        op.create_index(
            "uq_users_single_owner",
            "users",
            ["is_owner"],
            unique=True,
            postgresql_where=sa.text("is_owner"),
            sqlite_where=sa.text("is_owner"),
        )


def downgrade() -> None:
    op.drop_index("uq_users_single_owner", table_name="users")
    op.drop_column("users", "is_owner")
    op.drop_column("users", "is_admin")
