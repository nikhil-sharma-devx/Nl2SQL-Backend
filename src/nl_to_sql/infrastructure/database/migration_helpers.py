"""Data-backfill helpers for schema migrations.

Kept out of the numbered Alembic module so the logic is importable and unit
testable (Alembic revision modules are awkward to import directly). The
``0009`` migration calls :func:`backfill_multi_connections` after adding the
new columns; a unit test exercises the same helper against a throwaway SQLite
database.

The backfill is written with dialect-agnostic SQLAlchemy Core so it runs
identically on PostgreSQL (production) and SQLite (tests). Boolean decisions are
made in Python rather than in SQL ``WHERE`` clauses to avoid the 0/1-vs-true
mismatch between the two dialects.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import Connection


def backfill_multi_connections(bind: Connection) -> None:
    """Backfill ``connection_id`` across connections and the schema catalog.

    Idempotent: safe to run more than once (rows that already carry a
    ``connection_id`` are skipped). Concretely it:

      1. gives every legacy ``user_database_connections`` row a ``connection_id``
         plus sensible ``name``/``db_type`` defaults;
      2. guarantees exactly one ``is_default`` connection per user;
      3. creates a NULL-DSN "Server Default" connection for users who have a
         schema catalog but no stored connection (they were on the platform DB);
      4. stamps ``user_schemas`` / ``user_schema_tables`` with their owning
         connection's ``connection_id``.
    """
    now = datetime.utcnow()

    # ── 1. Assign connection_id + defaults to existing connection rows ──────────
    rows = bind.execute(
        text(
            "SELECT id, user_id, connection_id, name, db_type, is_default "
            "FROM user_database_connections"
        )
    ).fetchall()
    for r in rows:
        if not r.connection_id:
            bind.execute(
                text(
                    "UPDATE user_database_connections "
                    "SET connection_id = :cid, "
                    "    name = COALESCE(name, 'Default'), "
                    "    db_type = COALESCE(db_type, 'postgresql') "
                    "WHERE id = :id"
                ),
                {"cid": str(uuid.uuid4()), "id": r.id},
            )

    # ── 2. Exactly one default per user ─────────────────────────────────────────
    user_ids = [
        row.user_id
        for row in bind.execute(
            text("SELECT DISTINCT user_id FROM user_database_connections")
        )
    ]
    for uid in user_ids:
        conns = bind.execute(
            text(
                "SELECT id, is_default FROM user_database_connections "
                "WHERE user_id = :u ORDER BY id"
            ),
            {"u": uid},
        ).fetchall()
        has_default = any(bool(c.is_default) for c in conns)
        if not has_default and conns:
            bind.execute(
                text("UPDATE user_database_connections SET is_default = :t WHERE id = :id"),
                {"t": True, "id": conns[0].id},
            )

    # ── 3. Server-default connection for catalog-only users ─────────────────────
    orphan_users = [
        row.user_id
        for row in bind.execute(
            text(
                "SELECT DISTINCT user_id FROM user_schemas "
                "WHERE user_id NOT IN (SELECT user_id FROM user_database_connections)"
            )
        )
    ]
    for uid in orphan_users:
        bind.execute(
            text(
                "INSERT INTO user_database_connections "
                "(connection_id, user_id, name, db_type, encrypted_url, is_default, "
                " created_at, updated_at) "
                "VALUES (:cid, :u, :name, :db_type, NULL, :t, :now, :now)"
            ),
            {
                "cid": str(uuid.uuid4()),
                "u": uid,
                "name": "Server Default",
                "db_type": "postgresql",
                "t": True,
                "now": now,
            },
        )

    # ── 4. Stamp catalog rows with their connection's connection_id ─────────────
    for tbl in ("user_schemas", "user_schema_tables"):
        bind.execute(
            text(
                f"UPDATE {tbl} SET connection_id = ("  # noqa: S608 - fixed table names
                "  SELECT c.connection_id FROM user_database_connections c "
                f"  WHERE c.user_id = {tbl}.user_id AND c.is_default = :t "
                "  LIMIT 1"
                ") WHERE connection_id IS NULL"
            ),
            {"t": True},
        )


def backfill_personal_organizations(bind: Connection) -> None:
    """Create a personal ``Organization`` + OWNER membership for every user lacking one.

    Idempotent: only processes users with no ``organization_memberships`` row at
    all, so re-running after new users have self-bootstrapped via the
    registration hook (``OrganizationService.bootstrap_personal_org``) is a
    no-op for them. The ``slug`` embeds the generated org id, which trivially
    guarantees uniqueness without a retry loop.
    """
    now = datetime.utcnow()
    users = bind.execute(
        text(
            "SELECT id, email, full_name FROM users "
            "WHERE id NOT IN (SELECT user_id FROM organization_memberships)"
        )
    ).fetchall()
    for u in users:
        org_id = str(uuid.uuid4())
        display_name = u.full_name or u.email
        bind.execute(
            text(
                "INSERT INTO organizations "
                "(id, name, slug, is_personal, owner_user_id, created_at, updated_at) "
                "VALUES (:id, :name, :slug, :personal, :owner, :now, :now)"
            ),
            {
                "id": org_id,
                "name": f"{display_name}'s Organization",
                "slug": f"personal-{org_id}",
                "personal": True,
                "owner": u.id,
                "now": now,
            },
        )
        bind.execute(
            text(
                "INSERT INTO organization_memberships "
                "(organization_id, user_id, role, is_default, created_at, updated_at) "
                "VALUES (:org, :user, :role, :is_default, :now, :now)"
            ),
            {"org": org_id, "user": u.id, "role": "OWNER", "is_default": True, "now": now},
        )


# Every table that gained a nullable organization_id column in migration 0015.
_ORG_SCOPED_TABLES: tuple[str, ...] = (
    "user_database_connections",
    "metrics",
    "query_templates",
    "scheduled_queries",
    "dashboards",
    "saved_queries",
    "user_schemas",
    "user_schema_tables",
    "glossary_entries",
    "favorited_tables",
    "shared_queries",
)


def backfill_resource_organization_ids(bind: Connection) -> None:
    """Stamp every legacy resource row with its owner's default organization.

    Idempotent: only rows with ``organization_id IS NULL`` are touched. Must
    run after :func:`backfill_personal_organizations` so every ``user_id`` has
    a default-membership row to look up.
    """
    for tbl in _ORG_SCOPED_TABLES:
        bind.execute(
            text(
                f"UPDATE {tbl} SET organization_id = ("  # noqa: S608 - fixed table names
                "  SELECT om.organization_id FROM organization_memberships om "
                f"  WHERE om.user_id = {tbl}.user_id AND om.is_default = :t "
                "  LIMIT 1"
                ") WHERE organization_id IS NULL"
            ),
            {"t": True},
        )
