"""Integration tests for training data routes — GET/POST /api/v1/training/*."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

# ── Constants ─────────────────────────────────────────────────────────────────

USER_ID = "test-user-training-001"

FAKE_USER = UserPublic(
    id=USER_ID,
    email="trainingtest@example.com",
    full_name="Training Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)

STATS = {"total": 10, "used": 2, "unused": 8}
EXPORT_JSON = '{"a":1}'
EXPORT_JSONL = '{"prompt":"p","completion":"c"}\n{"prompt":"p2","completion":"c2"}'
MARKED = 2


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.get_training_stats = AsyncMock(return_value=STATS)
    svc.export_training_data = AsyncMock(return_value=EXPORT_JSON)
    svc.mark_as_used = AsyncMock(return_value=MARKED)
    return svc


def _make_app(mock_service=None, authed: bool = True, admin: bool = False):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_current_user, require_admin
        from nl_to_sql.api.routes import training as training_module

        app = create_app()

        if authed:
            async def _fake_user():
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user

        if admin:
            async def _fake_admin():
                return FAKE_USER

            app.dependency_overrides[require_admin] = _fake_admin

        if mock_service is not None:
            app.dependency_overrides[
                training_module.get_training_data_service
            ] = lambda: mock_service

        return app


@pytest.fixture
async def client(mock_service):
    """Admin-authenticated — every /training/* route is admin-only (C3)."""
    app = _make_app(mock_service=mock_service, authed=True, admin=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def non_admin_client(mock_service):
    app = _make_app(mock_service=mock_service, authed=True, admin=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(mock_service=None, authed=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ===========================================================================
# GET /api/v1/training/stats
# ===========================================================================


class TestTrainingStats:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/training/stats")
        assert r.status_code == 401

    async def test_returns_stats(self, client) -> None:
        r = await client.get("/api/v1/training/stats", headers=_auth())
        assert r.status_code == 200
        assert r.json() == STATS


# ===========================================================================
# GET /api/v1/training/export
# ===========================================================================


class TestExportTrainingData:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/training/export")
        assert r.status_code == 401

    async def test_export_json_happy(self, client, mock_service) -> None:
        r = await client.get(
            "/api/v1/training/export?format=json&limit=100&include_used=false",
            headers=_auth(),
        )
        assert r.status_code == 200
        assert r.json() == EXPORT_JSON
        mock_service.export_training_data.assert_awaited_once_with(
            format="json", limit=100, include_used=False
        )

    async def test_export_jsonl_happy(self, client, mock_service) -> None:
        mock_service.export_training_data = AsyncMock(return_value=EXPORT_JSONL)
        r = await client.get(
            "/api/v1/training/export?format=jsonl&include_used=true",
            headers=_auth(),
        )
        assert r.status_code == 200
        assert r.json() == EXPORT_JSONL

    async def test_invalid_format_returns_422(self, client) -> None:
        r = await client.get("/api/v1/training/export?format=xml", headers=_auth())
        assert r.status_code == 422

    async def test_limit_zero_returns_422(self, client) -> None:
        r = await client.get("/api/v1/training/export?limit=0", headers=_auth())
        assert r.status_code == 422

    async def test_limit_too_large_returns_422(self, client) -> None:
        r = await client.get("/api/v1/training/export?limit=10001", headers=_auth())
        assert r.status_code == 422


# ===========================================================================
# GET /api/v1/training/download
# ===========================================================================


class TestDownloadTrainingData:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/training/download")
        assert r.status_code == 401

    async def test_download_happy(self, client, mock_service) -> None:
        mock_service.export_training_data = AsyncMock(return_value=EXPORT_JSONL)
        r = await client.get("/api/v1/training/download", headers=_auth())
        assert r.status_code == 200
        disposition = r.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert "filename" in disposition
        assert "training_data" in disposition
        assert r.text == EXPORT_JSONL


# ===========================================================================
# POST /api/v1/training/mark-used
# ===========================================================================


class TestMarkUsed:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post("/api/v1/training/mark-used?ids=1&ids=2")
        assert r.status_code == 401

    async def test_mark_used_happy(self, client, mock_service) -> None:
        r = await client.post("/api/v1/training/mark-used?ids=1&ids=2", headers=_auth())
        assert r.status_code == 200
        assert r.json() == {"marked_count": MARKED}
        mock_service.mark_as_used.assert_awaited_once_with([1, 2])

    async def test_missing_ids_returns_422(self, client) -> None:
        r = await client.post("/api/v1/training/mark-used", headers=_auth())
        assert r.status_code == 422


# ===========================================================================
# C3: the training corpus is global — non-admins must be rejected
# ===========================================================================


class TestAdminRequired:

    async def test_stats_forbidden_for_non_admin(self, non_admin_client) -> None:
        r = await non_admin_client.get("/api/v1/training/stats", headers=_auth())
        assert r.status_code == 403

    async def test_export_forbidden_for_non_admin(self, non_admin_client) -> None:
        r = await non_admin_client.get("/api/v1/training/export", headers=_auth())
        assert r.status_code == 403

    async def test_download_forbidden_for_non_admin(self, non_admin_client) -> None:
        r = await non_admin_client.get("/api/v1/training/download", headers=_auth())
        assert r.status_code == 403

    async def test_mark_used_forbidden_for_non_admin(self, non_admin_client) -> None:
        r = await non_admin_client.post(
            "/api/v1/training/mark-used?ids=1", headers=_auth()
        )
        assert r.status_code == 403
