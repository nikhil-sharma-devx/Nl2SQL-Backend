"""Integration tests for Profile routes (backend/.../api/routes/profile.py):

  - API keys (BYOK)   GET/PUT/DELETE /api/v1/profile/api-keys
  - Available providers GET          /api/v1/profile/available-providers
  - Database (BYOD)   GET/PUT/DELETE /api/v1/profile/database

IMPORTANT: profile.py does NOT use FastAPI ``Depends`` for the api-key / db
services — it calls module-level helpers ``_get_api_key_service`` and
``_get_user_db_service`` that pull from the DI container. Those are therefore
PATCHED here, not overridden via ``app.dependency_overrides``.
``get_current_user`` IS a real dependency, so it is overridden; with no
override the endpoints return 401.

The error envelope on this app uses ``"message"`` (not ``"detail"``).
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

# ── Constants ─────────────────────────────────────────────────────────────────

USER_ID = "test-user-profile-001"
USER_EMAIL = "profiletest@example.com"

FAKE_USER = UserPublic(
    id=USER_ID,
    email=USER_EMAIL,
    full_name="Profile Test User",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)

# A key long enough (>10 chars) to exercise the masking branch.
SAMPLE_KEY = "sk-abcdef1234567890wxyz"

PROFILE = "nl_to_sql.api.routes.profile"


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


# ── App / client fixtures ─────────────────────────────────────────────────────


def _make_app(*, authed: bool):
    """Build the app; when ``authed`` override get_current_user to FAKE_USER."""
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_current_user

        app = create_app()

        if authed:
            async def _fake_user():
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user

        return app


@pytest.fixture
async def client():
    app = _make_app(authed=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(authed=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Service mock helpers ──────────────────────────────────────────────────────


def _api_key_service(*, get_key_return=None, has_key_return=False):
    svc = MagicMock()
    svc.get_key = AsyncMock(return_value=get_key_return)
    svc.has_key = AsyncMock(return_value=has_key_return)
    svc.save_key = AsyncMock(return_value=None)
    svc.delete_key = AsyncMock(return_value=None)
    return svc


def _connection_info(*, has_dsn=False, url_preview=None):
    from nl_to_sql.services.connection_service import ConnectionInfo

    return ConnectionInfo(
        connection_id="conn-1",
        name="Default",
        db_type="postgresql",
        is_default=True,
        has_dsn=has_dsn,
        url_preview=url_preview,
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


def _connection_service(*, default_info=None):
    svc = MagicMock()
    svc.get_default_info = AsyncMock(return_value=default_info or _connection_info())
    svc.upsert_default_dsn = AsyncMock(return_value=default_info or _connection_info())
    svc.clear_default_dsn = AsyncMock(return_value=None)
    return svc


def _fake_db_client():
    """Mock AsyncDatabaseClient instance whose connection test succeeds."""
    sess = MagicMock()
    sess.execute = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=sess)
    cm.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.session = MagicMock(return_value=cm)
    client.dispose = AsyncMock()
    return client


# ===========================================================================
# GET /api/v1/profile/api-keys
# ===========================================================================


class TestGetApiKeyStatus:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/profile/api-keys")
        assert r.status_code == 401

    async def test_lists_providers_without_user_key(self, client) -> None:
        svc = _api_key_service(get_key_return=None)
        with patch(f"{PROFILE}._get_api_key_service", return_value=svc):
            r = await client.get("/api/v1/profile/api-keys", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        providers = {item["provider"] for item in data["keys"]}
        assert {"groq", "openai", "anthropic", "gemini", "together"} <= providers
        for item in data["keys"]:
            assert item["has_user_key"] is False
            assert item["key_preview"] is None
        assert "active_provider" in data
        assert "active_model" in data

    async def test_lists_providers_with_stored_user_key(self, client) -> None:
        svc = _api_key_service(get_key_return=SAMPLE_KEY)
        with patch(f"{PROFILE}._get_api_key_service", return_value=svc):
            r = await client.get("/api/v1/profile/api-keys", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        item = next(i for i in data["keys"] if i["provider"] == "groq")
        assert item["has_user_key"] is True
        assert item["key_preview"] is not None
        # Masked: first 6 + "..." + last 4 of the sample key.
        assert item["key_preview"] == "sk-abc...wxyz"

    async def test_service_unavailable_still_returns_200(self, client) -> None:
        with patch(f"{PROFILE}._get_api_key_service", return_value=None):
            r = await client.get("/api/v1/profile/api-keys", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert all(i["has_user_key"] is False for i in data["keys"])


# ===========================================================================
# PUT /api/v1/profile/api-keys/{provider}
# ===========================================================================


class TestSaveApiKey:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.put(
            "/api/v1/profile/api-keys/groq", json={"api_key": SAMPLE_KEY}
        )
        assert r.status_code == 401

    async def test_happy_path(self, client) -> None:
        svc = _api_key_service()
        with patch(f"{PROFILE}._get_api_key_service", return_value=svc):
            r = await client.put(
                "/api/v1/profile/api-keys/groq",
                json={"api_key": SAMPLE_KEY},
                headers=_auth(),
            )
        assert r.status_code == 200
        data = r.json()
        assert data["provider"] == "groq"
        assert data["key_preview"] == "sk-abc...wxyz"
        assert "saved" in data["message"].lower()
        svc.save_key.assert_awaited_once()

    async def test_unsupported_provider_returns_400(self, client) -> None:
        svc = _api_key_service()
        with patch(f"{PROFILE}._get_api_key_service", return_value=svc):
            r = await client.put(
                "/api/v1/profile/api-keys/bogus",
                json={"api_key": SAMPLE_KEY},
                headers=_auth(),
            )
        assert r.status_code == 400
        assert "message" in r.json()

    async def test_service_unavailable_returns_503(self, client) -> None:
        with patch(f"{PROFILE}._get_api_key_service", return_value=None):
            r = await client.put(
                "/api/v1/profile/api-keys/groq",
                json={"api_key": SAMPLE_KEY},
                headers=_auth(),
            )
        assert r.status_code == 503
        assert "message" in r.json()


# ===========================================================================
# DELETE /api/v1/profile/api-keys/{provider}
# ===========================================================================


class TestDeleteApiKey:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/profile/api-keys/groq")
        assert r.status_code == 401

    async def test_happy_path(self, client) -> None:
        svc = _api_key_service()
        with patch(f"{PROFILE}._get_api_key_service", return_value=svc):
            r = await client.delete("/api/v1/profile/api-keys/groq", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["provider"] == "groq"
        assert "message" in data
        svc.delete_key.assert_awaited_once()

    async def test_unsupported_provider_returns_400(self, client) -> None:
        svc = _api_key_service()
        with patch(f"{PROFILE}._get_api_key_service", return_value=svc):
            r = await client.delete("/api/v1/profile/api-keys/bogus", headers=_auth())
        assert r.status_code == 400
        assert "message" in r.json()


# ===========================================================================
# GET /api/v1/profile/available-providers
# ===========================================================================


class TestAvailableProviders:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/profile/available-providers")
        assert r.status_code == 401

    async def test_returns_providers_list(self, client) -> None:
        svc = _api_key_service(has_key_return=True)
        with patch(f"{PROFILE}._get_api_key_service", return_value=svc):
            r = await client.get(
                "/api/v1/profile/available-providers", headers=_auth()
            )
        assert r.status_code == 200
        data = r.json()
        assert "providers" in data
        providers = {p["provider"] for p in data["providers"]}
        assert {"groq", "openai", "anthropic", "gemini", "together"} <= providers
        # has_key_return=True -> user source for every provider.
        assert all(p["source"] == "user" for p in data["providers"])

    async def test_returns_providers_without_service(self, client) -> None:
        with patch(f"{PROFILE}._get_api_key_service", return_value=None):
            r = await client.get(
                "/api/v1/profile/available-providers", headers=_auth()
            )
        assert r.status_code == 200
        data = r.json()
        assert len(data["providers"]) == 5


# ===========================================================================
# GET /api/v1/profile/database
# ===========================================================================


class TestGetDatabaseConnection:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/profile/database")
        assert r.status_code == 401

    async def test_has_connection_false_when_no_url(self, client) -> None:
        svc = _connection_service(default_info=_connection_info(has_dsn=False))
        with patch(f"{PROFILE}._get_connection_service", return_value=svc):
            r = await client.get("/api/v1/profile/database", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["has_connection"] is False
        assert data["url_preview"] is None

    async def test_has_connection_true_masks_password(self, client) -> None:
        info = _connection_info(
            has_dsn=True, url_preview="postgresql+asyncpg://user:***@dbhost:5432/mydb"
        )
        svc = _connection_service(default_info=info)
        with patch(f"{PROFILE}._get_connection_service", return_value=svc):
            r = await client.get("/api/v1/profile/database", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["has_connection"] is True
        assert data["url_preview"] is not None
        assert "secretpw" not in data["url_preview"]
        assert "***" in data["url_preview"]

    async def test_has_connection_false_when_service_none(self, client) -> None:
        with patch(f"{PROFILE}._get_connection_service", return_value=None):
            r = await client.get("/api/v1/profile/database", headers=_auth())
        assert r.status_code == 200
        assert r.json()["has_connection"] is False


# ===========================================================================
# PUT /api/v1/profile/database
# ===========================================================================


class TestSaveDatabaseConnection:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.put(
            "/api/v1/profile/database",
            json={"database_url": "postgresql://u:p@h:5432/db"},
        )
        assert r.status_code == 401

    async def test_empty_url_returns_400(self, client) -> None:
        r = await client.put(
            "/api/v1/profile/database",
            json={"database_url": "   "},
            headers=_auth(),
        )
        assert r.status_code == 400
        assert "message" in r.json()

    async def test_happy_path(self, client) -> None:
        info = _connection_info(
            has_dsn=True, url_preview="postgresql+asyncpg://user:***@dbhost:5432/mydb"
        )
        svc = _connection_service(default_info=info)
        with patch(f"{PROFILE}._get_connection_service", return_value=svc):
            r = await client.put(
                "/api/v1/profile/database",
                json={"database_url": "postgresql://user:secretpw@dbhost:5432/mydb"},
                headers=_auth(),
            )
        assert r.status_code == 200
        data = r.json()
        assert "url_preview" in data
        assert "secretpw" not in data["url_preview"]
        assert "message" in data
        svc.upsert_default_dsn.assert_awaited_once()

    async def test_connection_test_failure_returns_400(self, client) -> None:
        from nl_to_sql.core.exceptions import ConnectionTestError

        svc = _connection_service()
        svc.upsert_default_dsn = AsyncMock(side_effect=ConnectionTestError("connection refused"))
        with patch(f"{PROFILE}._get_connection_service", return_value=svc):
            r = await client.put(
                "/api/v1/profile/database",
                json={"database_url": "postgresql://user:secretpw@dbhost:5432/mydb"},
                headers=_auth(),
            )
        assert r.status_code == 400
        assert "message" in r.json()


# ===========================================================================
# DELETE /api/v1/profile/database
# ===========================================================================


class TestDeleteDatabaseConnection:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/profile/database")
        assert r.status_code == 401

    async def test_happy_path(self, client) -> None:
        svc = _connection_service()
        with patch(f"{PROFILE}._get_connection_service", return_value=svc):
            r = await client.delete("/api/v1/profile/database", headers=_auth())
        assert r.status_code == 200
        assert "message" in r.json()
        svc.clear_default_dsn.assert_awaited_once()
