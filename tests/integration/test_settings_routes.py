"""Integration tests for all Settings page routes.

Covers all 5 tabs:
  General / SQL Style  → GET /api/v1/settings, PATCH /api/v1/settings
  Instructions         → GET /api/v1/instructions, PUT /api/v1/instructions
  Data & Privacy       → GET/PUT /api/v1/account/retention, POST /api/v1/data/export
  Security             → GET/DELETE /api/v1/auth-sessions, GET /api/v1/login-activity

DB is mocked via FastAPI dependency_overrides — no real NeonDB connection needed.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "test-user-settings-001"
USER_EMAIL = "settings_test@example.com"

FAKE_USER = UserPublic(
    id=USER_ID,
    email=USER_EMAIL,
    full_name="Test User",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


def _make_app(mock_db=None):
    """Return app with dependency overrides. Pass mock_db=None for unauthenticated app."""
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_current_user, get_session_service

        app = create_app()

        if mock_db is not None:
            async def _fake_user():
                return FAKE_USER

            mock_service = MagicMock()
            mock_service._session_factory = MagicMock(return_value=mock_db)

            app.dependency_overrides[get_current_user] = _fake_user
            app.dependency_overrides[get_session_service] = lambda: mock_service

        return app


@pytest.fixture
def settings_app(mock_db):
    yield _make_app(mock_db)


@pytest.fixture
async def client(settings_app):
    async with AsyncClient(
        transport=ASGITransport(app=settings_app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
async def unauthed_client():
    """No dependency overrides — auth-required routes return 401."""
    app = _make_app(mock_db=None)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: configure mock_db.execute() to return a scalar
# ---------------------------------------------------------------------------


def _scalar_result(value):
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    return r


def _list_result(items: list):
    r = MagicMock()
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=items)))
    return r


# ===========================================================================
# 1. GENERAL + SQL STYLE  —  GET /api/v1/settings, PATCH /api/v1/settings
# ===========================================================================


class TestGetSettings:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.get("/api/v1/settings")
        assert r.status_code == 401

    async def test_returns_defaults_when_no_row(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.get("/api/v1/settings", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["sql_keyword_case"] == "upper"
        assert data["sql_cte_pref"] == "cte"
        assert data["sql_alias_style"] == "as"
        assert data["sql_indent"] == 2
        assert data["max_result_rows"] == 1000
        assert data["auto_execute"] is False
        assert data["data_retention"] == "forever"
        assert data["default_dialect"] is None
        assert data["default_model"] is None

    async def test_returns_stored_values(self, client, mock_db):
        row = MagicMock()
        row.sql_keyword_case = "lower"
        row.sql_cte_pref = "subquery"
        row.sql_alias_style = "implicit"
        row.sql_indent = 4
        row.default_dialect = "mysql"
        row.max_result_rows = 500
        row.auto_execute = True
        row.default_model = "gpt-4"
        row.data_retention = "30d"
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))

        r = await client.get("/api/v1/settings", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["sql_keyword_case"] == "lower"
        assert data["sql_indent"] == 4
        assert data["default_dialect"] == "mysql"
        assert data["auto_execute"] is True
        assert data["data_retention"] == "30d"


class TestPatchSettings:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.patch("/api/v1/settings", json={"sql_indent": 4})
        assert r.status_code == 401

    async def test_creates_row_when_missing(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))

        def _fill_defaults(obj):
            # Simulate what the DB would return after INSERT + refresh
            obj.sql_keyword_case = "lower"
            obj.sql_cte_pref = "cte"
            obj.sql_alias_style = "as"
            obj.sql_indent = 2
            obj.default_dialect = None
            obj.max_result_rows = 1000
            obj.auto_execute = False
            obj.default_model = None
            obj.data_retention = "forever"

        mock_db.refresh = AsyncMock(side_effect=_fill_defaults)

        r = await client.patch(
            "/api/v1/settings",
            json={"sql_keyword_case": "lower"},
            headers=_auth(),
        )
        assert r.status_code == 200
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    async def test_updates_existing_row(self, client, mock_db):
        existing = MagicMock()
        existing.sql_keyword_case = "upper"
        existing.sql_cte_pref = "cte"
        existing.sql_alias_style = "as"
        existing.sql_indent = 2
        existing.default_dialect = None
        existing.max_result_rows = 1000
        existing.auto_execute = False
        existing.default_model = None
        existing.data_retention = "forever"
        mock_db.execute = AsyncMock(return_value=_scalar_result(existing))

        r = await client.patch(
            "/api/v1/settings",
            json={"sql_indent": 4, "auto_execute": True},
            headers=_auth(),
        )
        assert r.status_code == 200
        assert existing.sql_indent == 4
        assert existing.auto_execute is True
        mock_db.commit.assert_awaited_once()

    async def test_rejects_invalid_keyword_case(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.patch(
            "/api/v1/settings",
            json={"sql_keyword_case": "mixed"},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_rejects_invalid_cte_pref(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.patch(
            "/api/v1/settings",
            json={"sql_cte_pref": "invalid"},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_rejects_invalid_alias_style(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.patch(
            "/api/v1/settings",
            json={"sql_alias_style": "bad"},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_rejects_sql_indent_out_of_range(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.patch(
            "/api/v1/settings",
            json={"sql_indent": 99},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_rejects_invalid_data_retention(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.patch(
            "/api/v1/settings",
            json={"data_retention": "1year"},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_empty_patch_is_valid(self, client, mock_db):
        existing = MagicMock()
        existing.sql_keyword_case = "upper"
        existing.sql_cte_pref = "cte"
        existing.sql_alias_style = "as"
        existing.sql_indent = 2
        existing.default_dialect = None
        existing.max_result_rows = 1000
        existing.auto_execute = False
        existing.default_model = None
        existing.data_retention = "forever"
        mock_db.execute = AsyncMock(return_value=_scalar_result(existing))

        r = await client.patch("/api/v1/settings", json={}, headers=_auth())
        assert r.status_code == 200


# ===========================================================================
# 2. INSTRUCTIONS TAB  —  GET /api/v1/instructions, PUT /api/v1/instructions
# ===========================================================================


class TestGetInstructions:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.get("/api/v1/instructions")
        assert r.status_code == 401

    async def test_returns_defaults_when_no_row(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.get("/api/v1/instructions", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["content"] == ""
        assert data["enabled"] is True
        assert data["char_count"] == 0

    async def test_returns_stored_instructions(self, client, mock_db):
        row = MagicMock()
        row.content = "Always use CTEs. Avoid subqueries."
        row.enabled = True
        row.char_count = 34
        row.updated_at = datetime(2024, 1, 1)
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))

        r = await client.get("/api/v1/instructions", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["content"] == "Always use CTEs. Avoid subqueries."
        assert data["char_count"] == 34


class TestPutInstructions:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.put("/api/v1/instructions", json={"content": "hi"})
        assert r.status_code == 401

    async def test_creates_new_instructions(self, client, mock_db):
        stored = MagicMock()
        stored.content = "Use snake_case for aliases."
        stored.enabled = True
        stored.char_count = 26
        stored.updated_at = datetime(2024, 1, 1)
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.refresh = AsyncMock(side_effect=lambda obj: (
            setattr(obj, "content", stored.content)
            or setattr(obj, "enabled", True)
            or setattr(obj, "char_count", 26)
            or setattr(obj, "updated_at", datetime(2024, 1, 1))
        ))

        r = await client.put(
            "/api/v1/instructions",
            json={"content": "Use snake_case for aliases.", "enabled": True},
            headers=_auth(),
        )
        assert r.status_code == 200
        mock_db.add.assert_called()
        mock_db.commit.assert_awaited_once()

    async def test_archives_previous_instructions(self, client, mock_db):
        existing = MagicMock()
        existing.content = "Old instructions"
        existing.enabled = True
        existing.char_count = 16
        existing.updated_at = datetime(2024, 1, 1)
        mock_db.execute = AsyncMock(return_value=_scalar_result(existing))

        r = await client.put(
            "/api/v1/instructions",
            json={"content": "New instructions", "enabled": True},
            headers=_auth(),
        )
        assert r.status_code == 200
        # History row + possibly the instructions row itself
        assert mock_db.add.call_count >= 1

    async def test_rejects_content_over_2000_chars(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.put(
            "/api/v1/instructions",
            json={"content": "x" * 2001},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_accepts_exactly_2000_chars(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.refresh = AsyncMock(side_effect=lambda obj: (
            setattr(obj, "content", "x" * 2000)
            or setattr(obj, "enabled", True)
            or setattr(obj, "char_count", 2000)
            or setattr(obj, "updated_at", datetime(2024, 1, 1))
        ))
        r = await client.put(
            "/api/v1/instructions",
            json={"content": "x" * 2000},
            headers=_auth(),
        )
        assert r.status_code == 200

    async def test_can_disable_instructions(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.refresh = AsyncMock(side_effect=lambda obj: (
            setattr(obj, "content", "")
            or setattr(obj, "enabled", False)
            or setattr(obj, "char_count", 0)
            or setattr(obj, "updated_at", datetime(2024, 1, 1))
        ))
        r = await client.put(
            "/api/v1/instructions",
            json={"content": "", "enabled": False},
            headers=_auth(),
        )
        assert r.status_code == 200


# ===========================================================================
# 3. DATA & PRIVACY  —  /api/v1/account/retention, /api/v1/data/export
# ===========================================================================


class TestGetRetention:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.get("/api/v1/account/retention")
        assert r.status_code == 401

    async def test_returns_forever_when_no_row(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.get("/api/v1/account/retention", headers=_auth())
        assert r.status_code == 200
        assert r.json()["data_retention"] == "forever"

    async def test_returns_stored_value(self, client, mock_db):
        row = MagicMock()
        row.data_retention = "7d"
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))
        r = await client.get("/api/v1/account/retention", headers=_auth())
        assert r.status_code == 200
        assert r.json()["data_retention"] == "7d"


class TestUpdateRetention:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.put(
            "/api/v1/account/retention", json={"data_retention": "30d"}
        )
        assert r.status_code == 401

    @pytest.mark.parametrize("value", ["forever", "30d", "7d", "none"])
    async def test_accepts_all_valid_values(self, value, client, mock_db):
        row = MagicMock()
        row.data_retention = value
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))
        r = await client.put(
            "/api/v1/account/retention",
            json={"data_retention": value},
            headers=_auth(),
        )
        assert r.status_code == 200
        assert r.json()["data_retention"] == value

    async def test_rejects_invalid_value(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.put(
            "/api/v1/account/retention",
            json={"data_retention": "1year"},
            headers=_auth(),
        )
        assert r.status_code == 422


class TestRequestExport:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.post("/api/v1/data/export")
        assert r.status_code == 401

    async def test_queues_export_job(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.refresh = AsyncMock(side_effect=lambda obj: (
            setattr(obj, "id", "export-job-123") or setattr(obj, "status", "queued")
        ))
        r = await client.post("/api/v1/data/export", headers=_auth())
        assert r.status_code == 202
        data = r.json()
        assert data["status"] == "queued"
        assert "job_id" in data


class TestDeleteAccount:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.post(
            "/api/v1/account/delete", json={"confirm": USER_EMAIL}
        )
        assert r.status_code == 401

    async def test_rejects_wrong_confirmation(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.post(
            "/api/v1/account/delete",
            json={"confirm": "wrong@email.com"},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_schedules_deletion_with_correct_email(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.post(
            "/api/v1/account/delete",
            json={"confirm": USER_EMAIL},
            headers=_auth(),
        )
        assert r.status_code == 202
        data = r.json()
        assert data["status"] == "scheduled"
        assert "purge_after" in data
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    async def test_idempotent_if_already_scheduled(self, client, mock_db):
        existing = MagicMock()
        existing.status = "scheduled"
        existing.purge_after = datetime(2024, 12, 31)
        mock_db.execute = AsyncMock(return_value=_scalar_result(existing))
        r = await client.post(
            "/api/v1/account/delete",
            json={"confirm": USER_EMAIL},
            headers=_auth(),
        )
        assert r.status_code == 202
        assert r.json()["status"] == "scheduled"
        mock_db.add.assert_not_called()

    async def test_cancel_deletion(self, client, mock_db):
        pending = MagicMock()
        pending.status = "scheduled"
        mock_db.execute = AsyncMock(return_value=_scalar_result(pending))
        r = await client.post("/api/v1/account/delete/cancel", headers=_auth())
        assert r.status_code == 200
        assert pending.status == "cancelled"
        mock_db.commit.assert_awaited_once()

    async def test_cancel_returns_404_when_none_pending(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.post("/api/v1/account/delete/cancel", headers=_auth())
        assert r.status_code == 404


# ===========================================================================
# 4. SECURITY  —  /api/v1/auth-sessions, /api/v1/login-activity
# ===========================================================================


class TestListAuthSessions:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.get("/api/v1/auth-sessions")
        assert r.status_code == 401

    async def test_returns_empty_list_when_none(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_list_result([]))
        r = await client.get("/api/v1/auth-sessions", headers=_auth())
        assert r.status_code == 200
        assert r.json()["items"] == []

    async def test_returns_active_sessions(self, client, mock_db):
        s = MagicMock()
        s.id = "session-abc"
        s.device = "Windows 11"
        s.browser = "Chrome"
        s.ip = "1.2.3.4"
        s.last_active_at = datetime(2024, 6, 1)
        s.created_at = datetime(2024, 5, 1)
        mock_db.execute = AsyncMock(return_value=_list_result([s]))

        r = await client.get("/api/v1/auth-sessions", headers=_auth())
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["id"] == "session-abc"
        assert items[0]["browser"] == "Chrome"


class TestRevokeAuthSession:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.delete("/api/v1/auth-sessions/some-id")
        assert r.status_code == 401

    async def test_returns_404_for_unknown_session(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.delete("/api/v1/auth-sessions/unknown-id", headers=_auth())
        assert r.status_code == 404

    async def test_revokes_session(self, client, mock_db):
        session = MagicMock()
        session.revoked_at = None
        mock_db.execute = AsyncMock(return_value=_scalar_result(session))
        r = await client.delete("/api/v1/auth-sessions/session-abc", headers=_auth())
        assert r.status_code == 204
        assert session.revoked_at is not None
        mock_db.commit.assert_awaited_once()


class TestRevokeAllAuthSessions:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.delete("/api/v1/auth-sessions")
        assert r.status_code == 401

    async def test_revokes_all_sessions(self, client, mock_db):
        s1, s2 = MagicMock(), MagicMock()
        s1.revoked_at = None
        s2.revoked_at = None
        mock_db.execute = AsyncMock(return_value=_list_result([s1, s2]))
        r = await client.delete("/api/v1/auth-sessions", headers=_auth())
        assert r.status_code == 200
        assert r.json()["revoked"] == 2
        assert s1.revoked_at is not None
        assert s2.revoked_at is not None


class TestLoginActivity:

    async def test_requires_auth(self, unauthed_client):
        r = await unauthed_client.get("/api/v1/login-activity")
        assert r.status_code == 401

    async def test_returns_empty_when_none(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_list_result([]))
        r = await client.get("/api/v1/login-activity", headers=_auth())
        assert r.status_code == 200
        assert r.json()["items"] == []

    async def test_returns_login_events(self, client, mock_db):
        e = MagicMock()
        e.ip = "5.6.7.8"
        e.user_agent = "Mozilla/5.0"
        e.outcome = "success"
        e.created_at = datetime(2024, 6, 15)
        mock_db.execute = AsyncMock(return_value=_list_result([e]))

        r = await client.get("/api/v1/login-activity", headers=_auth())
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["outcome"] == "success"
        assert items[0]["ip"] == "5.6.7.8"

    async def test_limit_param_is_accepted(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_list_result([]))
        r = await client.get("/api/v1/login-activity?limit=5", headers=_auth())
        assert r.status_code == 200

    async def test_limit_must_be_at_least_1(self, client, mock_db):
        r = await client.get("/api/v1/login-activity?limit=0", headers=_auth())
        assert r.status_code == 422
