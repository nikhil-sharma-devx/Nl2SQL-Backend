"""Migration backfill tests — 0009 multi-connection data migration.

Exercises ``backfill_multi_connections`` against a throwaway SQLite database in
the *post-DDL, pre-backfill* state (new columns present, ``connection_id`` NULL)
and asserts the invariants the migration must guarantee.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine, text

from nl_to_sql.infrastructure.database.migration_helpers import (
    backfill_multi_connections,
    backfill_personal_organizations,
    backfill_resource_organization_ids,
)

_NOW = datetime(2026, 7, 21, 12, 0, 0)


def _make_db():
    engine = create_engine("sqlite://")  # in-memory, single connection reused
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE user_database_connections ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " connection_id VARCHAR(36),"
                " user_id VARCHAR(36) NOT NULL,"
                " name VARCHAR(200),"
                " db_type VARCHAR(20),"
                " encrypted_url TEXT,"
                " is_default BOOLEAN,"
                " created_at DATETIME NOT NULL,"
                " updated_at DATETIME NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE user_schemas ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " user_id VARCHAR(36) NOT NULL,"
                " connection_id VARCHAR(36))"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE user_schema_tables ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " user_id VARCHAR(36) NOT NULL,"
                " connection_id VARCHAR(36),"
                " table_name VARCHAR(200))"
            )
        )
    return engine


def test_backfill_assigns_connection_id_and_default() -> None:
    engine = _make_db()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO user_database_connections "
                "(user_id, encrypted_url, created_at, updated_at) "
                "VALUES ('u1', 'enc', :now, :now)"
            ),
            {"now": _NOW},
        )

    with engine.begin() as conn:
        backfill_multi_connections(conn)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT connection_id, name, db_type, is_default "
                "FROM user_database_connections WHERE user_id='u1'"
            )
        ).one()
    assert row.connection_id  # a UUID was assigned
    assert row.name == "Default"
    assert row.db_type == "postgresql"
    assert bool(row.is_default) is True


def test_backfill_links_catalog_to_default_connection() -> None:
    engine = _make_db()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO user_database_connections "
                "(user_id, encrypted_url, created_at, updated_at) "
                "VALUES ('u1', 'enc', :now, :now)"
            ),
            {"now": _NOW},
        )
        conn.execute(text("INSERT INTO user_schemas (user_id) VALUES ('u1')"))
        conn.execute(
            text(
                "INSERT INTO user_schema_tables (user_id, table_name) "
                "VALUES ('u1', 'orders')"
            )
        )

    with engine.begin() as conn:
        backfill_multi_connections(conn)

    with engine.connect() as conn:
        cid = conn.execute(
            text("SELECT connection_id FROM user_database_connections WHERE user_id='u1'")
        ).scalar_one()
        schema_cid = conn.execute(
            text("SELECT connection_id FROM user_schemas WHERE user_id='u1'")
        ).scalar_one()
        table_cid = conn.execute(
            text("SELECT connection_id FROM user_schema_tables WHERE user_id='u1'")
        ).scalar_one()
    assert schema_cid == cid
    assert table_cid == cid


def test_backfill_creates_server_default_for_catalog_only_user() -> None:
    """A user with a catalog but no connection gets a NULL-DSN Server Default."""
    engine = _make_db()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO user_schemas (user_id) VALUES ('u2')"))
        conn.execute(
            text(
                "INSERT INTO user_schema_tables (user_id, table_name) "
                "VALUES ('u2', 'customers')"
            )
        )

    with engine.begin() as conn:
        backfill_multi_connections(conn)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT connection_id, name, encrypted_url, is_default "
                "FROM user_database_connections WHERE user_id='u2'"
            )
        ).one()
        schema_cid = conn.execute(
            text("SELECT connection_id FROM user_schemas WHERE user_id='u2'")
        ).scalar_one()
    assert row.name == "Server Default"
    assert row.encrypted_url is None
    assert bool(row.is_default) is True
    assert schema_cid == row.connection_id


def test_backfill_is_idempotent() -> None:
    engine = _make_db()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO user_database_connections "
                "(user_id, encrypted_url, created_at, updated_at) "
                "VALUES ('u1', 'enc', :now, :now)"
            ),
            {"now": _NOW},
        )
    with engine.begin() as conn:
        backfill_multi_connections(conn)
    with engine.connect() as conn:
        first = conn.execute(
            text("SELECT connection_id FROM user_database_connections WHERE user_id='u1'")
        ).scalar_one()
    # Running again must not change the assigned connection_id or duplicate rows.
    with engine.begin() as conn:
        backfill_multi_connections(conn)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT connection_id FROM user_database_connections WHERE user_id='u1'")
        ).scalars().all()
    assert rows == [first]


def test_backfill_preserves_existing_default() -> None:
    engine = _make_db()
    with engine.begin() as conn:
        for i, is_def in enumerate([False, True]):
            conn.execute(
                text(
                    "INSERT INTO user_database_connections "
                    "(user_id, name, connection_id, encrypted_url, is_default, "
                    "created_at, updated_at) "
                    "VALUES ('u1', :name, :cid, 'enc', :d, :now, :now)"
                ),
                {"name": f"c{i}", "cid": f"cid{i}", "d": is_def, "now": _NOW},
            )
    with engine.begin() as conn:
        backfill_multi_connections(conn)
    with engine.connect() as conn:
        defaults = conn.execute(
            text(
                "SELECT connection_id FROM user_database_connections "
                "WHERE user_id='u1' AND is_default=1"
            )
        ).scalars().all()
    assert defaults == ["cid1"]


def _make_org_db():
    """Post-DDL, pre-backfill state for migrations 0014/0015: users +
    organizations/memberships tables, plus one org-scoped resource table
    (``user_database_connections``) with a nullable ``organization_id``."""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE users ("
                " id VARCHAR(36) PRIMARY KEY,"
                " email VARCHAR(255) NOT NULL,"
                " full_name VARCHAR(200))"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE organizations ("
                " id VARCHAR(36) PRIMARY KEY,"
                " name VARCHAR(200) NOT NULL,"
                " slug VARCHAR(200) NOT NULL,"
                " is_personal BOOLEAN NOT NULL,"
                " owner_user_id VARCHAR(36) NOT NULL,"
                " created_at DATETIME NOT NULL,"
                " updated_at DATETIME NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE organization_memberships ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " organization_id VARCHAR(36) NOT NULL,"
                " user_id VARCHAR(36) NOT NULL,"
                " role VARCHAR(20) NOT NULL,"
                " is_default BOOLEAN NOT NULL,"
                " created_at DATETIME NOT NULL,"
                " updated_at DATETIME NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE user_database_connections ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " connection_id VARCHAR(36),"
                " user_id VARCHAR(36) NOT NULL,"
                " organization_id VARCHAR(36),"
                " name VARCHAR(200),"
                " created_at DATETIME NOT NULL,"
                " updated_at DATETIME NOT NULL)"
            )
        )
        # backfill_resource_organization_ids loops over every table in
        # migration 0015's _ORG_SCOPED_TABLES unconditionally (they all truly
        # exist by the time it runs in production) — stub the other ten here
        # so the loop doesn't hit "no such table" in this minimal test schema.
        for tbl in (
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
        ):
            conn.execute(
                text(
                    f"CREATE TABLE {tbl} ("
                    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    " user_id VARCHAR(36) NOT NULL,"
                    " organization_id VARCHAR(36))"
                )
            )
    return engine


def test_backfill_personal_organizations_creates_org_and_owner_membership() -> None:
    engine = _make_org_db()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO users (id, email, full_name) "
                "VALUES ('u1', 'u1@example.com', 'U One')"
            )
        )

    with engine.begin() as conn:
        backfill_personal_organizations(conn)

    with engine.connect() as conn:
        membership = conn.execute(
            text(
                "SELECT organization_id, role, is_default "
                "FROM organization_memberships WHERE user_id='u1'"
            )
        ).one()
        org = conn.execute(
            text("SELECT is_personal, owner_user_id FROM organizations WHERE id=:oid"),
            {"oid": membership.organization_id},
        ).one()
    assert membership.role == "OWNER"
    assert bool(membership.is_default) is True
    assert bool(org.is_personal) is True
    assert org.owner_user_id == "u1"


def test_backfill_personal_organizations_is_idempotent() -> None:
    engine = _make_org_db()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO users (id, email, full_name) VALUES ('u1', 'u1@example.com', NULL)")
        )
    with engine.begin() as conn:
        backfill_personal_organizations(conn)
    with engine.connect() as conn:
        first_org = conn.execute(
            text("SELECT organization_id FROM organization_memberships WHERE user_id='u1'")
        ).scalar_one()

    # Running again must not create a second organization/membership for the
    # same user (mirrors bootstrap_personal_org already having run at signup).
    with engine.begin() as conn:
        backfill_personal_organizations(conn)
    with engine.connect() as conn:
        org_ids = conn.execute(
            text("SELECT organization_id FROM organization_memberships WHERE user_id='u1'")
        ).scalars().all()
    assert org_ids == [first_org]


def test_backfill_resource_organization_ids_stamps_default_org() -> None:
    engine = _make_org_db()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO users (id, email, full_name) VALUES ('u1', 'u1@example.com', NULL)")
        )
        conn.execute(
            text(
                "INSERT INTO user_database_connections "
                "(connection_id, user_id, name, created_at, updated_at) "
                "VALUES ('c1', 'u1', 'Default', :now, :now)"
            ),
            {"now": _NOW},
        )
    with engine.begin() as conn:
        backfill_personal_organizations(conn)
        backfill_resource_organization_ids(conn)

    with engine.connect() as conn:
        expected_org = conn.execute(
            text(
                "SELECT organization_id FROM organization_memberships "
                "WHERE user_id='u1' AND is_default=1"
            )
        ).scalar_one()
        actual_org = conn.execute(
            text("SELECT organization_id FROM user_database_connections WHERE connection_id='c1'")
        ).scalar_one()
    assert actual_org == expected_org


def test_backfill_resource_organization_ids_never_overwrites_existing() -> None:
    engine = _make_org_db()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO users (id, email, full_name) VALUES ('u1', 'u1@example.com', NULL)")
        )
        conn.execute(
            text(
                "INSERT INTO user_database_connections "
                "(connection_id, user_id, organization_id, name, created_at, updated_at) "
                "VALUES ('c1', 'u1', 'already-set', 'Default', :now, :now)"
            ),
            {"now": _NOW},
        )
    with engine.begin() as conn:
        backfill_personal_organizations(conn)
        backfill_resource_organization_ids(conn)

    with engine.connect() as conn:
        actual_org = conn.execute(
            text("SELECT organization_id FROM user_database_connections WHERE connection_id='c1'")
        ).scalar_one()
    assert actual_org == "already-set"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
