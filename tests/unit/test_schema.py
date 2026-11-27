"""Unit tests for /api/v1/schema/refresh endpoint."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from nl_to_sql.api.app import create_app


@pytest.fixture
def app():
    """FastAPI test application."""
    from datetime import datetime

    from nl_to_sql.api.dependencies import get_current_user
    from nl_to_sql.core.models.auth import UserPublic

    app = create_app()

    async def _fake_user():
        return UserPublic(
            id="test-user-id",
            email="admin@example.com",
            full_name="Admin User",
            auth_provider="email",
            is_verified=True,
            created_at=datetime.utcnow()
        )

    app.dependency_overrides[get_current_user] = _fake_user
    return app


@pytest.fixture
def sync_client(app):
    return TestClient(app)


@pytest.fixture
async def async_client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


def _override_sync(app, catalog):
    """Wire /schema/refresh to a fake catalog and a fake active connection."""
    from nl_to_sql.api.dependencies import (
        get_active_connection_id,
        get_connection_service,
        get_db_client,
        get_schema_catalog,
    )

    client = MagicMock(name="connection-client")
    client.list_schemas = AsyncMock(return_value=[{"name": "public", "table_count": 12}])
    conns = MagicMock()
    conns.get_scoped_client = AsyncMock(return_value=client)
    catalog.synced_schemas = AsyncMock(return_value=[])

    async def _conn_id():
        return "conn-1"

    app.dependency_overrides[get_active_connection_id] = _conn_id
    app.dependency_overrides[get_schema_catalog] = lambda: catalog
    app.dependency_overrides[get_connection_service] = lambda: conns
    app.dependency_overrides[get_db_client] = lambda: MagicMock(name="default-client")


class TestSchemaRefresh:
    """Tests for POST /api/v1/schema/refresh (now an alias of /schema/sync)."""

    def test_refresh_success(self, app, sync_client):
        """Should sync the active connection and report its table count."""
        from types import SimpleNamespace

        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(
            return_value=SimpleNamespace(changed=True, new_tables=[], total_tables=12)
        )
        _override_sync(app, catalog)
        response = sync_client.post("/api/v1/schema/refresh")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Schema refreshed from the active connection."
        assert data["chunks_ingested"] == 12
        assert data["tables_found"] == 12
        assert catalog.sync_from_live_db.await_args.args[1] == "conn-1"

    def test_refresh_failure_returns_500(self, app, sync_client):
        """Should return 500 when reflecting the live DB fails."""
        catalog = MagicMock()
        catalog.sync_from_live_db = AsyncMock(side_effect=Exception("Database reflection failed"))
        _override_sync(app, catalog)
        response = sync_client.post("/api/v1/schema/refresh")
        assert response.status_code == 500
        assert "Schema sync failed" in response.json()["message"]


class TestSchemaStatus:
    """Tests for GET /api/v1/schema/status."""

    def test_schema_status_returns_data(self, app, sync_client):
        """Should return chunks_stored and vector_store_ready for the active connection."""
        from nl_to_sql.api.dependencies import get_active_connection_id, get_vector_store

        store = MagicMock()
        store.count = AsyncMock(return_value=7)
        store.health_check = AsyncMock(return_value=True)

        async def _conn_id():
            return "conn-1"

        app.dependency_overrides[get_active_connection_id] = _conn_id
        app.dependency_overrides[get_vector_store] = lambda: store
        response = sync_client.get("/api/v1/schema/status")
        assert response.status_code == 200
        data = response.json()
        assert "chunks_stored" in data
        assert "vector_store_ready" in data
        assert isinstance(data["chunks_stored"], int)
        assert isinstance(data["vector_store_ready"], bool)
        assert data["chunks_stored"] == 7
        store.count.assert_awaited_once_with(connection_id="conn-1")
