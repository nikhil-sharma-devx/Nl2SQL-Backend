"""Integration tests for Onboarding routes:
  - GET   /api/v1/onboarding
  - PATCH /api/v1/onboarding

All DB calls are mocked via FastAPI dependency_overrides. The route's
``_get_or_create`` runs ``db.execute(...)`` then ``.scalar_one_or_none()``;
returning ``None`` makes the route construct a real ``OnboardingState``
ORM object, so the happy path needs no fabricated row.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

# ── Constants ─────────────────────────────────────────────────────────────────

USER_ID = "test-user-onboarding-001"
USER_EMAIL = "onboarding@example.com"

FAKE_USER = UserPublic(
    id=USER_ID,
    email=USER_EMAIL,
    full_name="Onboarding Test User",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _scalar_result(value):
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    return r


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
def onboarding_app(mock_db):
    yield _make_app(mock_db)


@pytest.fixture
async def client(onboarding_app):
    async with AsyncClient(
        transport=ASGITransport(app=onboarding_app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(mock_db=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ===========================================================================
# GET /api/v1/onboarding
# ===========================================================================


class TestGetOnboarding:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/onboarding")
        assert r.status_code == 401

    async def test_returns_state_for_new_user(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))

        r = await client.get("/api/v1/onboarding", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["completed_items"] == []
        assert "available_items" in data
        assert len(data["available_items"]) == 7
        assert data["progress_pct"] == 0


# ===========================================================================
# PATCH /api/v1/onboarding
# ===========================================================================


class TestPatchOnboarding:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.patch(
            "/api/v1/onboarding", json={"completed_items": ["run_first_query"]}
        )
        assert r.status_code == 401

    async def test_missing_body_field_returns_422(self, client) -> None:
        r = await client.patch("/api/v1/onboarding", json={}, headers=_auth())
        assert r.status_code == 422

    async def test_updates_completed_items_and_progress(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))

        r = await client.patch(
            "/api/v1/onboarding",
            json={"completed_items": ["run_first_query", "save_a_query"]},
            headers=_auth(),
        )
        assert r.status_code == 200
        data = r.json()
        assert "run_first_query" in data["completed_items"]
        assert "save_a_query" in data["completed_items"]
        # 2 of 7 available items completed -> ~28.6%. Tolerate round vs
        # truncation across implementations (int() -> 28, round() -> 29).
        assert data["progress_pct"] > 0
        assert abs(data["progress_pct"] - (2 / 7 * 100)) < 1

    async def test_unknown_items_do_not_inflate_progress(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))

        r = await client.patch(
            "/api/v1/onboarding",
            json={"completed_items": ["run_first_query", "totally_bogus_item"]},
            headers=_auth(),
        )
        assert r.status_code == 200
        data = r.json()
        # Only the 1 valid item counts toward progress, not the bogus one.
        assert abs(data["progress_pct"] - (1 / 7 * 100)) < 1
        assert data["progress_pct"] < 100
