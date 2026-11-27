"""Integration tests for the configuration controller.

Covers every endpoint in ``backend/src/nl_to_sql/api/routes/config.py``:

  GET  /api/v1/config/llm       (unauthenticated)
  PUT  /api/v1/config/llm       (auth, rate-limited)
  GET  /api/v1/config/models    (unauthenticated)
  GET  /api/v1/config/database  (admin)
  PUT  /api/v1/config/database  (auth, rate-limited, SSRF-guarded)
  GET  /api/v1/config/rag       (auth)
  PUT  /api/v1/config/rag       (auth, rate-limited)

Follows the working pattern in ``tests/unit/test_config.py``: overrides
``get_current_user`` with a fake user, uses the REAL DI container for GET
endpoints, patches ``AsyncDatabaseClient`` + ``_reject_ssrf_host`` for the DB
PUT, and reads the error envelope from the ``"message"`` key.

The RAG PUT mutates the live Settings singleton, so those tests capture the
original values and restore them in a ``finally`` block to stay self-cleaning.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nl_to_sql.api.app import create_app
from nl_to_sql.api.dependencies import get_current_user, require_admin
from nl_to_sql.core.models.auth import UserPublic

FAKE_USER = UserPublic(
    id="test-config-user-id",
    email="config_test@example.com",
    full_name="Config Test User",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


@pytest.fixture
def app():
    """App with get_current_user + require_admin overridden — REAL DI container otherwise.

    The mutating config endpoints (PUT /llm, /database, /rag) are admin-only
    (C1); this fixture's user is treated as an admin so the existing
    happy-path tests keep exercising the handler logic. Non-admin rejection
    is covered separately by ``non_admin_client``.
    """
    application = create_app()

    async def fake_get_current_user():
        return FAKE_USER

    async def fake_require_admin():
        return FAKE_USER

    application.dependency_overrides[get_current_user] = fake_get_current_user
    application.dependency_overrides[require_admin] = fake_require_admin
    return application


@pytest.fixture
def sync_client(app):
    return TestClient(app)


@pytest.fixture
def unauth_client():
    """App WITHOUT the auth override — protected endpoints should 401."""
    return TestClient(create_app())


@pytest.fixture
def non_admin_client():
    """App with get_current_user overridden but NOT require_admin.

    FAKE_USER has is_admin=False (the users.is_admin default), so it fails the admin check —
    used to assert non-admins are rejected from the mutating config routes.
    """
    application = create_app()

    async def fake_get_current_user():
        return FAKE_USER

    application.dependency_overrides[get_current_user] = fake_get_current_user
    return TestClient(application)


# ── GET /api/v1/config/llm ───────────────────────────────────────────────────


class TestGetLLMConfig:
    def test_get_llm_config_returns_shape(self, sync_client):
        response = sync_client.get("/api/v1/config/llm")
        assert response.status_code == 200
        data = response.json()
        assert "provider" in data
        assert "model" in data
        assert "available_providers" in data
        assert isinstance(data["available_providers"], list)
        assert set(data["available_providers"]) == {
            "groq",
            "openai",
            "anthropic",
            "gemini",
        }


# ── PUT /api/v1/config/llm ───────────────────────────────────────────────────


class TestUpdateLLMConfig:
    def test_invalid_provider_returns_400(self, sync_client):
        response = sync_client.put(
            "/api/v1/config/llm",
            json={"provider": "bogus", "model": "x"},
        )
        assert response.status_code == 400
        assert "Invalid provider" in response.json()["message"]

    def test_invalid_model_returns_400(self, sync_client):
        response = sync_client.put(
            "/api/v1/config/llm",
            json={"provider": "groq", "model": "nope"},
        )
        assert response.status_code == 400
        assert "Invalid model" in response.json()["message"]

    def test_missing_fields_returns_422(self, sync_client):
        response = sync_client.put("/api/v1/config/llm", json={})
        assert response.status_code == 422

    def test_happy_path_returns_200(self, sync_client):
        """Valid provider/model with the switch patched (no real mutation).

        A user key is forced via the api_key_service so the handler always
        finds a key regardless of the server env — making this deterministic.
        """
        fake_container = MagicMock()
        fake_container.api_key_service.return_value.has_key = AsyncMock(
            return_value=True
        )

        with (
            patch(
                "nl_to_sql.api.routes.config.ApplicationContainer.switch_llm_provider"
            ) as mock_switch,
            patch(
                "nl_to_sql.api.dependencies._get_container",
                return_value=fake_container,
            ),
        ):
            response = sync_client.put(
                "/api/v1/config/llm",
                json={
                    "provider": "groq",
                    "model": "openai/gpt-oss-120b",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "groq"
        assert data["model"] == "openai/gpt-oss-120b"
        assert "message" in data
        mock_switch.assert_called_once()


# ── GET /api/v1/config/models ────────────────────────────────────────────────


class TestGetAvailableModels:
    def test_returns_models_per_provider(self, sync_client):
        response = sync_client.get("/api/v1/config/models")
        assert response.status_code == 200
        data = response.json()
        for provider in ("groq", "openai", "anthropic", "gemini"):
            assert provider in data
            assert isinstance(data[provider], list)
            assert len(data[provider]) > 0


# ── GET /api/v1/config/database ──────────────────────────────────────────────


class TestGetDatabaseConfig:
    def test_returns_masked_url(self, sync_client):
        response = sync_client.get("/api/v1/config/database")
        assert response.status_code == 200
        data = response.json()
        assert "database_url" in data
        assert isinstance(data["database_url"], str)
        assert "available_databases" in data
        assert isinstance(data["available_databases"], dict)

    def test_non_admin_cannot_see_the_platform_database(self, non_admin_client):
        """The platform DB host (even masked) is admin-only, like the PUT."""
        response = non_admin_client.get("/api/v1/config/database")
        assert response.status_code == 403


# ── PUT /api/v1/config/database ──────────────────────────────────────────────


class TestUpdateDatabaseConfig:
    def test_empty_url_returns_400(self, sync_client):
        response = sync_client.put(
            "/api/v1/config/database",
            json={"database_url": "   "},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["message"].lower()

    def test_missing_field_returns_422(self, sync_client):
        response = sync_client.put("/api/v1/config/database", json={})
        assert response.status_code == 422

    @patch(
        "nl_to_sql.api.routes.config.ApplicationContainer.switch_db_client",
        new_callable=AsyncMock,
    )
    @patch("nl_to_sql.api.routes.config.AsyncDatabaseClient", autospec=True)
    def test_happy_path_returns_200(self, mock_client, mock_switch, sync_client):
        """Valid URL that passes the health check + SSRF guard → 200."""
        mock_session = MagicMock()
        mock_session.execute = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_instance = MagicMock()
        mock_instance.session.return_value = mock_ctx
        mock_instance.dispose = AsyncMock()
        mock_client.return_value = mock_instance

        with patch("nl_to_sql.api.routes.config._reject_ssrf_host"):
            response = sync_client.put(
                "/api/v1/config/database",
                json={
                    "database_url": "postgresql+asyncpg://user:pass@db.example.com:5432/testdb"
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "database_url" in data
        # password should be masked in the echoed URL
        assert "pass" not in data["database_url"]
        assert "***" in data["database_url"]
        mock_switch.assert_awaited_once()

    @patch("nl_to_sql.api.routes.config.AsyncDatabaseClient", autospec=True)
    def test_failed_connection_returns_400(self, mock_client, sync_client):
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=Exception("cannot connect"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_instance = MagicMock()
        mock_instance.session.return_value = mock_ctx
        mock_instance.dispose = AsyncMock()
        mock_client.return_value = mock_instance

        with patch("nl_to_sql.api.routes.config._reject_ssrf_host"):
            response = sync_client.put(
                "/api/v1/config/database",
                json={
                    "database_url": "postgresql+asyncpg://user:pass@db.example.com:5432/testdb"
                },
            )

        assert response.status_code == 400
        assert "Database connection failed" in response.json()["message"]

    def test_ssrf_guard_rejects_internal_host(self, sync_client):
        """The real SSRF guard should reject a loopback host with 400."""
        response = sync_client.put(
            "/api/v1/config/database",
            json={"database_url": "postgresql+asyncpg://user:pass@127.0.0.1:5432/db"},
        )
        assert response.status_code == 400
        assert "internal network" in response.json()["message"].lower()


# ── GET /api/v1/config/rag ───────────────────────────────────────────────────

_RAG_FIELDS = (
    "schema_descriptions_enabled",
    "multi_query_enabled",
    "multi_query_max",
    "few_shot_retrieval_enabled",
    "few_shot_top_k",
    "parent_child_chunking_enabled",
    "hyde_enabled",
    "adaptive_top_k_enabled",
    "adaptive_top_k_min",
    "adaptive_top_k_max",
)


class TestGetRagConfig:
    def test_returns_all_flags(self, sync_client):
        response = sync_client.get("/api/v1/config/rag")
        assert response.status_code == 200
        data = response.json()
        for field in _RAG_FIELDS:
            assert field in data


# ── PUT /api/v1/config/rag ───────────────────────────────────────────────────


class TestUpdateRagConfig:
    def test_happy_path_reflects_update_and_restores(self, sync_client):
        """Toggle multi_query_enabled, assert it's reflected, then restore."""
        original = sync_client.get("/api/v1/config/rag").json()
        target = not original["multi_query_enabled"]
        try:
            response = sync_client.put(
                "/api/v1/config/rag",
                json={"multi_query_enabled": target},
            )
            assert response.status_code == 200
            assert response.json()["multi_query_enabled"] is target
        finally:
            # restore the live singleton so we don't leak state into other tests
            sync_client.put(
                "/api/v1/config/rag",
                json={"multi_query_enabled": original["multi_query_enabled"]},
            )

    def test_min_greater_than_max_returns_422(self, sync_client):
        response = sync_client.put(
            "/api/v1/config/rag",
            json={"adaptive_top_k_min": 20, "adaptive_top_k_max": 5},
        )
        assert response.status_code == 422


# ── GET/PUT /api/v1/config/rate-limit ────────────────────────────────────────


class TestRateLimitConfig:
    def test_get_reflects_live_limiter_state(self, sync_client):
        from nl_to_sql.api.middleware.rate_limiter import limiter

        response = sync_client.get("/api/v1/config/rate-limit")
        assert response.status_code == 200
        assert response.json()["enabled"] is limiter.enabled

    def test_put_toggles_live_limiter_and_restores(self, sync_client):
        """Toggling the flag takes effect on the live Limiter object itself —
        not just a value nothing else reads — since that's the whole point
        of the fix (SlowAPI checks ``limiter.enabled`` fresh per request)."""
        from nl_to_sql.api.middleware.rate_limiter import limiter

        original = limiter.enabled
        try:
            response = sync_client.put("/api/v1/config/rate-limit", json={"enabled": False})
            assert response.status_code == 200
            assert response.json()["enabled"] is False
            assert limiter.enabled is False
        finally:
            limiter.enabled = original

    def test_missing_field_returns_422(self, sync_client):
        response = sync_client.put("/api/v1/config/rate-limit", json={})
        assert response.status_code == 422


# ── Authentication ───────────────────────────────────────────────────────────


class TestAuthRequired:
    def test_put_llm_requires_auth(self, unauth_client):
        response = unauth_client.put(
            "/api/v1/config/llm",
            json={"provider": "groq", "model": "openai/gpt-oss-120b"},
        )
        assert response.status_code == 401

    def test_get_rag_requires_auth(self, unauth_client):
        response = unauth_client.get("/api/v1/config/rag")
        assert response.status_code == 401

    def test_get_llm_is_public(self, unauth_client):
        """GET /llm is unauthenticated — should succeed without a token."""
        response = unauth_client.get("/api/v1/config/llm")
        assert response.status_code == 200

    def test_get_models_is_public(self, unauth_client):
        response = unauth_client.get("/api/v1/config/models")
        assert response.status_code == 200


# ── C1: mutating config routes are admin-only ───────────────────────────────


class TestAdminRequired:
    """A logged-in, non-admin user must not be able to mutate platform config."""

    def test_put_llm_requires_admin(self, non_admin_client):
        response = non_admin_client.put(
            "/api/v1/config/llm",
            json={"provider": "groq", "model": "openai/gpt-oss-120b"},
        )
        assert response.status_code == 403

    def test_put_database_requires_admin(self, non_admin_client):
        response = non_admin_client.put(
            "/api/v1/config/database",
            json={"database_url": "postgresql+asyncpg://user:pass@db.example.com:5432/testdb"},
        )
        assert response.status_code == 403

    def test_put_rag_requires_admin(self, non_admin_client):
        response = non_admin_client.put(
            "/api/v1/config/rag",
            json={"multi_query_enabled": True},
        )
        assert response.status_code == 403

    def test_put_rate_limit_requires_admin(self, non_admin_client):
        response = non_admin_client.put(
            "/api/v1/config/rate-limit",
            json={"enabled": False},
        )
        assert response.status_code == 403
