"""Built-in starter content flag — query_templates, metrics, scheduled_queries, dashboards.

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-27

Adds ``is_builtin`` (Boolean, NOT NULL, default False) to the four
user-content tables. Marks rows seeded automatically as starter/example
content (see ``services/starter_content_service.py``) so the frontend can
label them — the flag never restricts CRUD; built-ins remain fully editable
and deletable like any user-created row.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("query_templates", "metrics", "scheduled_queries", "dashboards")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(
            table,
            sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    for table in _TABLES:
        op.drop_column(table, "is_builtin")
