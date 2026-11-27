"""Integration tests for POST /api/v1/query/preview (query cost preview)."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

FAKE_USER = UserPublic(
    id="preview-user-001",
    email="preview@example.com",
    full_name="Preview User",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)

CANNED_PREVIEW = {
    "supported": True,
    "estimated_rows": 1234,
    "estimated_cost": 987.6,
    "plan": {"node_type": "Seq Scan", "relation": "customers", "total_cost": 987.6},
    "warnings": [{"type": "seq_scan", "message": "Sequential (full-table) scan on customers."}],
    "message": None,
}


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _make_app(explain_return=None, with_client=True, authed=True):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_current_user, get_request_orchestrator

        app = create_app()

        if authed:
            async def _fake_user():
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user

            orchestrator = MagicMock()
            if with_client:
                db_client = MagicMock()
                db_client.explain = AsyncMock(return_value=explain_return)
                orchestrator._db_client = db_client
            else:
                orchestrator._db_client = None

            async def _fake_orch():
                return orchestrator

            app.dependency_overrides[get_request_orchestrator] = _fake_orch

        return app


@pytest.fixture
async def _client_factory():
    clients = []

    async def _build(**kwargs):
        app = _make_app(**kwargs)
        c = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(c)
        return c

    yield _build
    for c in clients:
        await c.aclose()


async def test_preview_requires_auth(_client_factory) -> None:
    client = await _client_factory(authed=False)
    resp = await client.post("/api/v1/query/preview", json={"sql": "SELECT 1"})
    assert resp.status_code == 401


async def test_preview_returns_shape(_client_factory) -> None:
    client = await _client_factory(explain_return=CANNED_PREVIEW)
    resp = await client.post(
        "/api/v1/query/preview", json={"sql": "SELECT * FROM customers"}, headers=_auth()
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is True
    assert data["estimated_rows"] == 1234
    assert data["estimated_cost"] == 987.6
    assert data["sql"] == "SELECT * FROM customers"
    assert data["warnings"][0]["type"] == "seq_scan"


async def test_preview_unsupported_passthrough(_client_factory) -> None:
    unsupported = {
        "supported": False,
        "estimated_rows": None,
        "estimated_cost": None,
        "plan": None,
        "warnings": [],
        "message": "Query preview is only supported on PostgreSQL databases.",
    }
    client = await _client_factory(explain_return=unsupported)
    resp = await client.post("/api/v1/query/preview", json={"sql": "SELECT 1"}, headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is False
    assert "PostgreSQL" in data["message"]


async def test_preview_no_db_client(_client_factory) -> None:
    client = await _client_factory(with_client=False)
    resp = await client.post("/api/v1/query/preview", json={"sql": "SELECT 1"}, headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is False
    assert "not configured" in data["message"]
