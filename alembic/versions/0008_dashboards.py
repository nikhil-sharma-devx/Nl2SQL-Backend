"""Auto Charting & Dashboards — dashboards + dashboard_widgets tables.

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-20

Adds the ``dashboards`` and ``dashboard_widgets`` tables backing the "Auto
Charting and Dashboards" feature. Also auto-created on startup by
ensure_schema(), but per the migration policy every schema change ships an
Alembic migration so ``alembic upgrade head`` provisions it in prod.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dashboards",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_dashboards_user_id"), "dashboards", ["user_id"])
    op.create_index(
        "ix_dashboards_user_updated", "dashboards", ["user_id", "updated_at"]
    )

    op.create_table(
        "dashboard_widgets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dashboard_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("nl_prompt", sa.Text(), nullable=True),
        sa.Column("sql", sa.Text(), nullable=False),
        sa.Column("chart_type", sa.String(length=20), nullable=False),
        sa.Column("chart_config", sa.JSON(), nullable=True),
        sa.Column("layout", sa.JSON(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dashboard_id"], ["dashboards.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_dashboard_widgets_dashboard_id"), "dashboard_widgets", ["dashboard_id"]
    )
    op.create_index(
        "ix_dashboard_widgets_dashboard", "dashboard_widgets", ["dashboard_id"]
    )


def downgrade() -> None:
    op.drop_index(
        "ix_dashboard_widgets_dashboard", table_name="dashboard_widgets"
    )
    op.drop_index(
        op.f("ix_dashboard_widgets_dashboard_id"), table_name="dashboard_widgets"
    )
    op.drop_table("dashboard_widgets")
    op.drop_index("ix_dashboards_user_updated", table_name="dashboards")
    op.drop_index(op.f("ix_dashboards_user_id"), table_name="dashboards")
    op.drop_table("dashboards")
