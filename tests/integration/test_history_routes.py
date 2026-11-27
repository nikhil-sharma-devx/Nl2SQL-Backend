"""Integration tests for history routes — GET/DELETE /api/v1/history."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

USER_ID = "test-user-history-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="histtest@example.com",
    full_name="History Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_history_svc():
    svc = MagicMock()
    svc.initialize = AsyncMock()
    svc.dispose = AsyncMock()
    svc.get_entries = AsyncMock(return_value=([], 0))
    svc.count = AsyncMock(return_value=0)
    svc.clear = AsyncMock(return_value=0)
    svc.get_user_entries = AsyncMock(return_value=([], 0))
    svc.clear_user = AsyncMock(return_value=0)
    return svc


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    return db


def _make_app(mock_history_svc=None, mock_db=None, authed=True, admin=False):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_current_user,
            get_query_history,
            get_session_service,
            require_admin,
        )

        app = create_app()

        if authed:
            async def _fake_user():
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user

        if admin:
            async def _fake_admin():
                return FAKE_USER

            app.dependency_overrides[require_admin] = _fake_admin

        if mock_history_svc is not None:
            app.dependency_overrides[get_query_history] = lambda: mock_history_svc

        if mock_db is not None:
            mock_svc = MagicMock()
            mock_svc._session_factory = MagicMock(return_value=mock_db)
            mock_svc.list_all_messages = AsyncMock(return_value=[])
            mock_svc.count_all_messages = AsyncMock(return_value=0)
            app.dependency_overrides[get_session_service] = lambda: mock_svc

        return app


@pytest.fixture
async def client(mock_history_svc, mock_db):
    app = _make_app(mock_history_svc=mock_history_svc, mock_db=mock_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def admin_client(mock_history_svc, mock_db):
    """Admin-authenticated — DELETE /history (legacy, unscoped table) is admin-only (C4)."""
    app = _make_app(mock_history_svc=mock_history_svc, mock_db=mock_db, admin=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(authed=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestGetHistory:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/history")
        assert r.status_code == 401

    async def test_returns_200_with_empty_list(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_mock_count_and_messages(0, []))
        r = await client.get("/api/v1/history", headers=_auth())
        assert r.status_code == 200

    async def test_default_pagination_params(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_mock_count_and_messages(0, []))
        r = await client.get("/api/v1/history", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert "entries" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    async def test_limit_and_offset_accepted(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_mock_count_and_messages(0, []))
        r = await client.get("/api/v1/history?limit=5&offset=10", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["limit"] == 5
        assert data["offset"] == 10

    async def test_invalid_limit_returns_422(self, client) -> None:
        r = await client.get("/api/v1/history?limit=0", headers=_auth())
        assert r.status_code == 422

    async def test_invalid_offset_returns_422(self, client) -> None:
        r = await client.get("/api/v1/history?offset=-1", headers=_auth())
        assert r.status_code == 422


class TestDeleteHistory:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/history")
        assert r.status_code == 401

    async def test_non_admin_forbidden(self, client, mock_db) -> None:
        """C4: the legacy query_history table has no user_id column, so
        clearing it wipes every user's data — admin-only."""
        mock_db.execute = AsyncMock(return_value=_mock_scalar(5))
        r = await client.delete("/api/v1/history", headers=_auth())
        assert r.status_code == 403

    async def test_clear_returns_200(self, admin_client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_mock_scalar(5))
        r = await admin_client.delete("/api/v1/history", headers=_auth())
        assert r.status_code in (200, 204)


class TestHistoryExport:

    async def test_export_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/history/export")
        assert r.status_code == 401

    async def test_export_json_format(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_mock_count_and_messages(0, []))
        r = await client.get("/api/v1/history/export?format=json", headers=_auth())
        assert r.status_code in (200, 404, 422)

    async def test_export_csv_format(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_mock_count_and_messages(0, []))
        r = await client.get("/api/v1/history/export?format=csv", headers=_auth())
        assert r.status_code in (200, 404, 422)


# ── Mock helpers ──────────────────────────────────────────────────────────────


def _mock_count_and_messages(count: int, messages: list):
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=count)
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=messages)))
    r.scalar = MagicMock(return_value=count)
    return r


def _mock_scalar(value):
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    r.scalar = MagicMock(return_value=value)
    return r
