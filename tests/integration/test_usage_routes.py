"""Integration tests for Usage routes:
  - Usage  GET /api/v1/usage?period=today|7d|30d

All DB calls are mocked via FastAPI dependency_overrides.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

# ── Constants ─────────────────────────────────────────────────────────────────

USER_ID = "test-user-usage-001"
USER_EMAIL = "usagetest@example.com"

FAKE_USER = UserPublic(
    id=USER_ID,
    email=USER_EMAIL,
    full_name="Usage Test User",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)

# Rough cost estimate per token (Groq pricing, USD) — mirrors the controller.
_COST_PER_TOKEN = 0.000_000_59


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.flush = AsyncMock()
    db.delete = AsyncMock()
    return db


def _make_app(mock_db=None):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_current_user, get_session_service

        app = create_app()

        if mock_db is not None:
            async def _fake_user():
                return FAKE_USER

            mock_svc = MagicMock()
            mock_svc._session_factory = MagicMock(return_value=mock_db)

            app.dependency_overrides[get_current_user] = _fake_user
            app.dependency_overrides[get_session_service] = lambda: mock_svc

        return app


@pytest.fixture
def usage_app(mock_db):
    yield _make_app(mock_db)


@pytest.fixture
async def client(usage_app):
    async with AsyncClient(transport=ASGITransport(app=usage_app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(mock_db=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ===========================================================================
# Usage
# ===========================================================================


class TestUsageRoutes:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/usage")
        assert r.status_code == 401

    async def test_default_period_happy_path(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(
            return_value=MagicMock(one=MagicMock(return_value=(5, 1000)))
        )

        r = await client.get("/api/v1/usage", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["queries_used"] == 5
        assert data["tokens_in"] == 1000
        assert data["tokens_out"] == 0
        assert data["period"] == "7d"  # default
        assert data["est_cost_usd"] == round(1000 * _COST_PER_TOKEN, 6)

    @pytest.mark.parametrize("period", ["today", "7d", "30d"])
    async def test_each_valid_period(self, client, mock_db, period) -> None:
        mock_db.execute = AsyncMock(
            return_value=MagicMock(one=MagicMock(return_value=(5, 1000)))
        )

        r = await client.get(f"/api/v1/usage?period={period}", headers=_auth())
        assert r.status_code == 200
        assert r.json()["period"] == period

    async def test_invalid_period_returns_422(self, client) -> None:
        r = await client.get("/api/v1/usage?period=bogus", headers=_auth())
        assert r.status_code == 422
