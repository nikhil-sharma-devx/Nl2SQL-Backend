"""Integration tests for the schema controller — every endpoint in
``backend/src/nl_to_sql/api/routes/schema.py``.

Covers:
  - GET    /api/v1/schema/tables
  - GET    /api/v1/schema/tables/{table_id}
  - PATCH  /api/v1/schema/tables/{table_id}
  - POST   /api/v1/schema/tables/seen
  - POST   /api/v1/schema/sync
  - POST   /api/v1/schema/ingest
  - GET    /api/v1/schema/status
  - POST   /api/v1/schema/refresh
  - GET    /api/v1/schema/visualize

Auth is exercised via ``get_current_user`` override (missing override -> 401).
Service deps are stubbed via ``app.dependency_overrides``. The error envelope
uses ``"message"`` (see ``http_exception_handler``), not ``"detail"``.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from nl_to_sql.api.app import create_app
from nl_to_sql.api.dependencies import (
    get_active_connection_id,
    get_connection_service,
    get_current_user,
    get_db_client,
    get_schema_catalog,
    get_schema_doc_service,
    get_vector_store,
)
from nl_to_sql.core.exceptions import ConnectionValidationError, TableNotFoundError
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.schema_doc_service import SchemaExplanation

# ── Constants ─────────────────────────────────────────────────────────────────

USER_ID = "test-user-schema-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="schematest@example.com",
    full_name="Schema Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)

BLOCKED_SCHEMA = "pg_catalog"
CONNECTION_ID = "conn-schema-001"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _table(table_id: int = 1, **over) -> dict:
    """A dict satisfying the ``CatalogTable`` response model."""
    t = {
        "id": table_id,
        "schema_name": "public",
        "table_name": "orders",
        "source": "reflected",
        "columns": [],
        "description": None,
        "pinned": False,
        "is_new": False,
        "last_seen_at": None,
    }
    t.update(over)
    return t


def _catalog(tables: list | None = None) -> dict:
    """A dict satisfying the ``SchemaTablesResponse`` response model."""
    return {
        "database_name": "testdb",
        "dialect": "postgresql",
        "source": "reflected",
        "last_synced_at": None,
        "tables": tables if tables is not None else [],
    }


def _make_app(*, authed: bool = True, overrides: dict | None = None):
    app = create_app()

    if authed:
        async def _fake_user():
            return FAKE_USER

        async def _fake_connection_id():
            return CONNECTION_ID

        app.dependency_overrides[get_current_user] = _fake_user
        app.dependency_overrides[get_active_connection_id] = _fake_connection_id

    for dep, provider in (overrides or {}).items():
        app.dependency_overrides[dep] = provider

    return app


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/schema/tables
# ═══════════════════════════════════════════════════════════════════════════════


class TestListSchemaTables:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.get("/api/v1/schema/tables")
        assert r.status_code == 401

    async def test_returns_200_with_catalog(self) -> None:
        catalog = MagicMock()
        catalog.get_catalog = AsyncMock(return_value=_catalog([_table(1)]))
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/tables")

        assert r.status_code == 200
        data = r.json()
        assert data["database_name"] == "testdb"
        assert data["dialect"] == "postgresql"
        assert len(data["tables"]) == 1
        assert data["tables"][0]["id"] == 1

    async def test_blocked_schema_name_returns_400(self) -> None:
        catalog = MagicMock()
        catalog.get_catalog = AsyncMock(return_value=_catalog())
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.get(f"/api/v1/schema/tables?schema_name={BLOCKED_SCHEMA}")

        assert r.status_code == 400
        assert "not accessible" in r.json()["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/schema/tables/{table_id}
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetSchemaTable:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.get("/api/v1/schema/tables/1")
        assert r.status_code == 401

    async def test_found_returns_200(self) -> None:
        catalog = MagicMock()
        catalog.get_catalog = AsyncMock(return_value=_catalog([_table(7, table_name="users")]))
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/tables/7")

        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 7
        assert data["table_name"] == "users"

    async def test_not_found_returns_404(self) -> None:
        catalog = MagicMock()
        catalog.get_catalog = AsyncMock(return_value=_catalog([_table(1)]))
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/tables/999")

        assert r.status_code == 404
        assert r.json()["message"] == "Table not found."


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH /api/v1/schema/tables/{table_id}
# ═══════════════════════════════════════════════════════════════════════════════


class TestPatchSchemaTable:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.patch("/api/v1/schema/tables/1", json={"user_description": "x"})
        assert r.status_code == 401

    async def test_updates_returns_200(self) -> None:
        updated = _table(3, description="A nice table")
        catalog = MagicMock()
        catalog.set_table_description = AsyncMock(return_value=updated)
        catalog.get_catalog = AsyncMock(return_value=_catalog([updated]))
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.patch(
                "/api/v1/schema/tables/3",
                json={"user_description": "A nice table"},
            )

        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 3
        assert data["description"] == "A nice table"
        catalog.set_table_description.assert_awaited_once()

    async def test_not_found_returns_404(self) -> None:
        catalog = MagicMock()
        catalog.set_table_description = AsyncMock(return_value=None)
        catalog.get_catalog = AsyncMock(return_value=_catalog())
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.patch(
                "/api/v1/schema/tables/999",
                json={"user_description": "x"},
            )

        assert r.status_code == 404
        assert r.json()["message"] == "Table not found."


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/schema/tables/seen
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarkTablesSeen:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.post("/api/v1/schema/tables/seen", json={"table_ids": [1, 2]})
        assert r.status_code == 401

    async def test_marks_seen_returns_200(self) -> None:
        catalog = MagicMock()
        catalog.mark_seen = AsyncMock(return_value=2)
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/tables/seen", json={"table_ids": [1, 2]})

        assert r.status_code == 200
        assert r.json()["updated"] == 2
        catalog.mark_seen.assert_awaited_once_with(USER_ID, CONNECTION_ID, [1, 2])


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/schema/sync
# ═══════════════════════════════════════════════════════════════════════════════


CONNECTION_CLIENT = MagicMock(name="active-connection-client")
CONNECTION_CLIENT.list_schemas = AsyncMock(
    return_value=[
        {"name": "public", "table_count": 1},
        {"name": "sales", "table_count": 2},
        {"name": "empty", "table_count": 0},
    ]
)
DEFAULT_CLIENT = MagicMock(name="server-default-client")


def _conns(scoped: AsyncMock | None = None) -> MagicMock:
    conns = MagicMock()
    conns.get_scoped_client = scoped or AsyncMock(return_value=CONNECTION_CLIENT)
    return conns


def _sync_overrides(catalog, conns: MagicMock | None = None):
    conns = conns or _conns()
    if not isinstance(getattr(catalog, "synced_schemas", None), AsyncMock):
        catalog.synced_schemas = AsyncMock(return_value=[])
    return {
        get_schema_catalog: lambda: catalog,
        get_connection_service: lambda: conns,
        get_db_client: lambda: DEFAULT_CLIENT,
    }


class TestSyncSchema:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.post("/api/v1/schema/sync")
        assert r.status_code == 401

    async def test_sync_returns_200(self) -> None:
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(
            return_value=SimpleNamespace(changed=True, new_tables=["orders"], total_tables=5)
        )
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/sync")

        assert r.status_code == 200
        data = r.json()
        assert data["changed"] is True
        assert data["new_tables"] == ["orders"]
        assert data["total_tables"] == 5

    async def test_blocked_schema_name_returns_400(self) -> None:
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock()
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post(f"/api/v1/schema/sync?schema_name={BLOCKED_SCHEMA}")

        assert r.status_code == 400
        assert "not accessible" in r.json()["message"]

    async def test_sync_exception_returns_500(self) -> None:
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(side_effect=Exception("boom"))
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/sync")

        assert r.status_code == 500
        assert "Schema sync failed" in r.json()["message"]

    async def test_sync_reflects_the_active_connection_strictly(self) -> None:
        """Regression: the catalog is filled from the ACTIVE connection's client,
        resolved strictly from the same connection id it is written under."""
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(
            return_value=SimpleNamespace(changed=True, new_tables=[], total_tables=1)
        )
        conns = _conns()
        app = _make_app(overrides=_sync_overrides(catalog, conns))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/sync")

        assert r.status_code == 200
        conns.get_scoped_client.assert_awaited_once_with(
            FAKE_USER.id, CONNECTION_ID, DEFAULT_CLIENT, strict=True
        )
        args = catalog.sync_from_live_db.await_args.args
        assert args[1] == CONNECTION_ID and args[2] is CONNECTION_CLIENT

    async def test_first_sync_covers_every_schema_with_tables(self) -> None:
        """Regression: only `public` was reflected, so tables in other schemas
        synced as "0 tables"."""
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(
            return_value=SimpleNamespace(changed=True, new_tables=[], total_tables=3)
        )
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/sync")

        assert r.status_code == 200
        assert catalog.sync_from_live_db.await_args.kwargs["schemas"] == ["public", "sales"]

    async def test_resync_keeps_the_schemas_already_synced(self) -> None:
        catalog = MagicMock()
        catalog.synced_schemas = AsyncMock(return_value=["sales"])
        catalog.sync_from_live_db = AsyncMock(
            return_value=SimpleNamespace(changed=False, new_tables=[], total_tables=2)
        )
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/sync")

        assert r.status_code == 200
        assert catalog.sync_from_live_db.await_args.kwargs["schemas"] == ["sales"]

    async def test_explicit_schemas_are_synced_as_given(self) -> None:
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(
            return_value=SimpleNamespace(changed=True, new_tables=[], total_tables=2)
        )
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/sync?schemas=sales&schemas=hr&schemas=sales")
            blocked = await c.post(f"/api/v1/schema/sync?schemas=sales&schemas={BLOCKED_SCHEMA}")

        assert r.status_code == 200
        assert catalog.sync_from_live_db.await_args.kwargs["schemas"] == ["sales", "hr"]
        assert blocked.status_code == 400

    async def test_lists_schemas_with_sync_state(self) -> None:
        catalog = MagicMock()
        catalog.synced_schemas = AsyncMock(return_value=["public"])
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/schemas")

        assert r.status_code == 200
        assert r.json()["schemas"] == [
            {"name": "public", "table_count": 1, "synced": True},
            {"name": "sales", "table_count": 2, "synced": False},
            {"name": "empty", "table_count": 0, "synced": False},
        ]

    async def test_unreadable_connection_is_an_error_not_the_default_db(self) -> None:
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock()
        conns = _conns(AsyncMock(side_effect=ConnectionValidationError("can't be read")))
        app = _make_app(overrides=_sync_overrides(catalog, conns))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/sync")

        assert r.status_code == 400
        catalog.sync_from_live_db.assert_not_awaited()


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/schema/ingest
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_UPLOAD = b'{"tables":[{"name":"t","columns":[{"name":"c"}]}]}'


class TestIngestSchema:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.post(
                "/api/v1/schema/ingest",
                files={"file": ("schema.json", _VALID_UPLOAD, "application/json")},
            )
        assert r.status_code == 401

    async def test_valid_upload_returns_200(self) -> None:
        catalog = MagicMock()
        catalog.apply_upload = AsyncMock(return_value=SimpleNamespace(tables=1))
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.post(
                "/api/v1/schema/ingest",
                files={"file": ("schema.json", _VALID_UPLOAD, "application/json")},
            )

        assert r.status_code == 200
        data = r.json()
        assert data["chunks_ingested"] == 1
        assert "table(s)" in data["message"]

    async def test_invalid_json_returns_400(self) -> None:
        catalog = MagicMock()
        catalog.apply_upload = AsyncMock()
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        async with _client(app) as c:
            r = await c.post(
                "/api/v1/schema/ingest",
                files={"file": ("schema.json", b"{not valid json", "application/json")},
            )

        assert r.status_code == 400
        assert "Invalid JSON" in r.json()["message"]

    async def test_malformed_shape_returns_400(self) -> None:
        catalog = MagicMock()
        catalog.apply_upload = AsyncMock()
        app = _make_app(overrides={get_schema_catalog: lambda: catalog})

        # 'tables' is not a list -> rejected by _validate_upload_shape
        async with _client(app) as c:
            r = await c.post(
                "/api/v1/schema/ingest",
                files={"file": ("schema.json", b'{"tables": "nope"}', "application/json")},
            )

        assert r.status_code == 400
        assert "'tables' array" in r.json()["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/schema/explain
# ═══════════════════════════════════════════════════════════════════════════════


def _explanation(**over) -> SchemaExplanation:
    data = {
        "table": "orders",
        "column": None,
        "schema_name": "public",
        "description": "Customer orders.",
        "business_meaning": "Each row is a purchase.",
        "relationships": ["orders.customer_id → customers.customer_id"],
        "example_usage": "Count orders per customer.",
        "common_joins": ["orders JOIN customers ON orders.customer_id = customers.customer_id"],
        "example_sql": "SELECT * FROM orders LIMIT 10;",
        "cached": False,
        "llm_generated": True,
    }
    data.update(over)
    return SchemaExplanation.model_validate(data)


class TestExplainSchema:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.get("/api/v1/schema/explain?table=orders")
        assert r.status_code == 401

    async def test_returns_200_shape(self) -> None:
        svc = MagicMock()
        svc.explain = AsyncMock(return_value=_explanation())
        app = _make_app(overrides={get_schema_doc_service: lambda: svc})

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/explain?table=orders")

        assert r.status_code == 200
        data = r.json()
        assert data["table"] == "orders"
        assert data["description"] == "Customer orders."
        assert data["business_meaning"]
        assert data["relationships"]
        assert data["example_usage"]
        assert data["common_joins"]
        assert data["example_sql"]
        svc.explain.assert_awaited_once_with(
            user_id=USER_ID, connection_id=CONNECTION_ID, table="orders", column=None
        )

    async def test_column_param_forwarded(self) -> None:
        svc = MagicMock()
        svc.explain = AsyncMock(return_value=_explanation(column="total"))
        app = _make_app(overrides={get_schema_doc_service: lambda: svc})

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/explain?table=orders&column=total")

        assert r.status_code == 200
        assert r.json()["column"] == "total"
        svc.explain.assert_awaited_once_with(
            user_id=USER_ID, connection_id=CONNECTION_ID, table="orders", column="total"
        )

    async def test_missing_table_param_returns_422(self) -> None:
        svc = MagicMock()
        svc.explain = AsyncMock()
        app = _make_app(overrides={get_schema_doc_service: lambda: svc})

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/explain")

        assert r.status_code == 422

    async def test_unknown_table_returns_404(self) -> None:
        svc = MagicMock()
        svc.explain = AsyncMock(side_effect=TableNotFoundError("Table 'ghost' was not found."))
        app = _make_app(overrides={get_schema_doc_service: lambda: svc})

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/explain?table=ghost")

        assert r.status_code == 404
        assert "not found" in r.json()["message"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/schema/status
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaStatus:

    async def test_returns_200(self) -> None:
        vector_store = MagicMock()
        vector_store.count = AsyncMock(return_value=42)
        vector_store.health_check = AsyncMock(return_value=True)
        app = _make_app(overrides={get_vector_store: lambda: vector_store})

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/status")

        assert r.status_code == 200
        data = r.json()
        assert data["chunks_stored"] == 42
        assert data["vector_store_ready"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/schema/refresh
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaRefresh:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.post("/api/v1/schema/refresh")
        assert r.status_code == 401

    async def test_returns_200_from_the_active_connection(self) -> None:
        """Regression: /refresh used to re-ingest the server's default DB
        globally, so the default tables showed whatever connection was active."""
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(
            return_value=SimpleNamespace(changed=True, new_tables=["a"], total_tables=12)
        )
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/refresh")

        assert r.status_code == 200
        data = r.json()
        assert data["chunks_ingested"] == 12
        assert data["tables_found"] == 12
        args = catalog.sync_from_live_db.await_args.args
        assert args[1] == CONNECTION_ID and args[2] is CONNECTION_CLIENT

    async def test_run_exception_returns_500(self) -> None:
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(side_effect=Exception("reflection failed"))
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post("/api/v1/schema/refresh")

        assert r.status_code == 500
        assert "Schema sync failed" in r.json()["message"]

    async def test_blocked_schema_name_returns_400(self) -> None:
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock()
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.post(f"/api/v1/schema/refresh?schema_name={BLOCKED_SCHEMA}")

        assert r.status_code == 400
        assert "not accessible" in r.json()["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/schema/visualize
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualizeSchema:

    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False)
        async with _client(app) as c:
            r = await c.get("/api/v1/schema/visualize")
        assert r.status_code == 401

    @staticmethod
    def _catalog(tables: list[dict]) -> MagicMock:
        catalog = MagicMock()
        catalog.get_catalog = AsyncMock(
            return_value={
                "database_name": "db",
                "dialect": "postgresql",
                "source": "reflected",
                "last_synced_at": None,
                "tables": tables,
            }
        )
        return catalog

    async def test_serves_the_synced_catalog_with_sql_names(self) -> None:
        """Regression: the graph/builder reflected only `public` of the live DB,
        so tables in other schemas (and new connections) never showed."""
        col = {"name": "id", "data_type": "INTEGER", "primary_key": True}
        fk = {"name": "user_id", "data_type": "INTEGER", "foreign_key": "users.id"}
        catalog = self._catalog(
            [
                {"schema_name": "public", "table_name": "users", "description": None, "columns": [col]},
                {"schema_name": "sales", "table_name": "orders", "description": "Orders", "columns": [col, fk]},
            ]
        )
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/visualize")

        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "catalog"
        assert [t["name"] for t in data["tables"]] == ["users", "sales.orders"]
        assert data["tables"][1]["table_name"] == "orders"
        assert catalog.get_catalog.await_args.args[1] == CONNECTION_ID
        CONNECTION_CLIENT.reflect_schema.assert_not_called()

    async def test_before_first_sync_reads_every_live_schema(self) -> None:
        catalog = self._catalog([])
        CONNECTION_CLIENT.reflect_schema = AsyncMock(
            return_value={
                "dialect": "postgresql",
                "tables": [
                    {"name": "users", "schema_name": "public", "columns": []},
                    {"name": "orders", "schema_name": "sales", "columns": []},
                ],
            }
        )
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/visualize")

        assert r.status_code == 200
        assert [t["name"] for t in r.json()["tables"]] == ["users", "sales.orders"]
        assert CONNECTION_CLIENT.reflect_schema.await_args.kwargs["schemas"] == ["public", "sales"]

    async def test_reflect_exception_returns_500(self) -> None:
        catalog = self._catalog([])
        CONNECTION_CLIENT.reflect_schema = AsyncMock(side_effect=Exception("no connection"))
        app = _make_app(overrides=_sync_overrides(catalog))

        async with _client(app) as c:
            r = await c.get("/api/v1/schema/visualize")

        assert r.status_code == 500
        assert "Schema visualization failed" in r.json()["message"]

    async def test_blocked_schema_name_returns_400(self) -> None:
        app = _make_app(overrides=_sync_overrides(self._catalog([])))

        async with _client(app) as c:
            r = await c.get(f"/api/v1/schema/visualize?schema_name={BLOCKED_SCHEMA}")

        assert r.status_code == 400
        assert "not accessible" in r.json()["message"]
