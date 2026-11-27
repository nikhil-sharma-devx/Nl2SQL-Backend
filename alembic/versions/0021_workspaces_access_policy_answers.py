"""Workspaces (rebuilt), per-connection access policy, answer-summary opt-out.

Revision ID: 0021
Revises: 0020
Create Date: 2026-11-26

Adds, all additively (no existing column changes meaning):

  - ``workspaces`` + ``workspace_members`` — a deliberately smaller team model
    than the one ``0017`` removed: membership by existing user id only (no
    invitation tokens, no join requests, no custom roles), three fixed roles
    (admin / editor / member), and only connection-level resources are shared.
  - ``user_database_connections.workspace_id`` — the workspace a connection is
    shared into (NULL = private).
  - ``user_database_connections.allowed_tables`` (JSON, NULL = all) and
    ``restricted_columns`` (JSON list of "table.column") — the access policy.
  - ``glossary_entries.workspace_id`` — workspace-scoped glossary terms.
  - ``users.active_shared_connection_id`` — the active connection when it is a
    shared one (a member can't set ``is_default`` on another user's row).
  - ``user_settings.answer_summary_enabled`` — opt-out for sending result rows
    to the LLM to write a plain-English answer (default on).
  - ``chat_messages.trust`` (JSON) — the answer / assumptions / confidence of
    each assistant message, so reopened chats match what was shown live.

Idempotent guards (``_has_table`` / ``_has_column``) match the existing
migrations, because ``ensure_schema()``'s create_all backstop may have
created some of these in dev databases first.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONN = "user_database_connections"


def _has_table(insp: sa.Inspector, table: str) -> bool:
    return table in insp.get_table_names()


def _has_column(insp: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def _has_index(insp: sa.Inspector, table: str, index: str) -> bool:
    return any(i["name"] == index for i in insp.get_indexes(table))


def _add_fk(bind: sa.Connection, name: str, table: str, ondelete: str) -> None:
    # SQLite can't ALTER-add a foreign key; the ORM model still declares it.
    if bind.dialect.name != "sqlite":
        op.create_foreign_key(
            name, table, "workspaces", ["workspace_id"], ["id"], ondelete=ondelete
        )


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if not _has_table(insp, "workspaces"):
        op.create_table(
            "workspaces",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column(
                "created_by",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_workspaces_created_by", "workspaces", ["created_by"])

    if not _has_table(insp, "workspace_members"):
        op.create_table(
            "workspace_members",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "workspace_id",
                sa.String(36),
                sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("role", sa.String(10), nullable=False, server_default="member"),
            sa.Column("added_by", sa.String(36), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member"),
        )
        op.create_index("ix_workspace_members_workspace_id", "workspace_members", ["workspace_id"])
        op.create_index("ix_workspace_members_user", "workspace_members", ["user_id"])

    insp = sa.inspect(bind)

    if not _has_column(insp, _CONN, "workspace_id"):
        op.add_column(_CONN, sa.Column("workspace_id", sa.String(36), nullable=True))
        _add_fk(bind, "fk_user_database_connections_workspace", _CONN, "SET NULL")
    if not _has_index(insp, _CONN, "ix_user_database_connections_workspace_id"):
        op.create_index("ix_user_database_connections_workspace_id", _CONN, ["workspace_id"])
    if not _has_column(insp, _CONN, "allowed_tables"):
        op.add_column(_CONN, sa.Column("allowed_tables", sa.JSON(), nullable=True))
    if not _has_column(insp, _CONN, "restricted_columns"):
        op.add_column(
            _CONN,
            sa.Column("restricted_columns", sa.JSON(), nullable=False, server_default="[]"),
        )

    if not _has_column(insp, "glossary_entries", "workspace_id"):
        op.add_column("glossary_entries", sa.Column("workspace_id", sa.String(36), nullable=True))
        _add_fk(bind, "fk_glossary_entries_workspace", "glossary_entries", "CASCADE")
    if not _has_index(insp, "glossary_entries", "ix_glossary_entries_workspace_id"):
        op.create_index("ix_glossary_entries_workspace_id", "glossary_entries", ["workspace_id"])

    if not _has_column(insp, "users", "active_shared_connection_id"):
        op.add_column(
            "users", sa.Column("active_shared_connection_id", sa.String(36), nullable=True)
        )

    if _has_table(insp, "chat_messages") and not _has_column(insp, "chat_messages", "trust"):
        op.add_column("chat_messages", sa.Column("trust", sa.JSON(), nullable=True))

    if _has_table(insp, "user_settings") and not _has_column(
        insp, "user_settings", "answer_summary_enabled"
    ):
        op.add_column(
            "user_settings",
            sa.Column(
                "answer_summary_enabled", sa.Boolean(), nullable=False, server_default=sa.true()
            ),
        )


def _drop_columns(table: str, columns: list[str], indexes: list[str]) -> None:
    """Drop columns via batch mode: a table rebuild on SQLite (which can't drop
    a column a foreign key is declared on), a plain ALTER on Postgres (which
    drops the dependent FK with the column, whatever it was named)."""
    insp = sa.inspect(op.get_bind())
    present = [c for c in columns if _has_column(insp, table, c)]
    for index in indexes:
        if _has_index(insp, table, index):
            op.drop_index(index, table_name=table)
    if not present:
        return
    with op.batch_alter_table(table) as batch:
        for col in present:
            batch.drop_column(col)


def downgrade() -> None:
    insp = sa.inspect(op.get_bind())

    if _has_table(insp, "chat_messages"):
        _drop_columns("chat_messages", ["trust"], [])
    if _has_table(insp, "user_settings"):
        _drop_columns("user_settings", ["answer_summary_enabled"], [])
    _drop_columns("users", ["active_shared_connection_id"], [])
    _drop_columns("glossary_entries", ["workspace_id"], ["ix_glossary_entries_workspace_id"])
    _drop_columns(
        _CONN,
        ["restricted_columns", "allowed_tables", "workspace_id"],
        ["ix_user_database_connections_workspace_id"],
    )

    insp = sa.inspect(op.get_bind())
    if _has_table(insp, "workspace_members"):
        op.drop_table("workspace_members")
    if _has_table(insp, "workspaces"):
        op.drop_table("workspaces")
