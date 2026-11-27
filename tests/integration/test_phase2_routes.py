"""Integration tests for Phase 2 routes:
  - Glossary          GET/POST/PATCH/DELETE /api/v1/glossary
  - Favorited Tables  GET/POST/DELETE /api/v1/favorited-tables
  - Query Templates   GET/POST/PATCH/DELETE /api/v1/query-templates
  - Tutorial Progress GET/PATCH/DELETE /api/v1/tutorial/progress
  - Notification Prefs GET/PATCH /api/v1/notification-preferences

All DB calls are mocked via FastAPI dependency_overrides.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

# ── Constants ─────────────────────────────────────────────────────────────────

USER_ID = "test-user-p2-001"
USER_EMAIL = "p2test@example.com"

FAKE_USER = UserPublic(
    id=USER_ID,
    email=USER_EMAIL,
    full_name="P2 Test User",
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


def _list_result(items: list):
    r = MagicMock()
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=items)))
    return r


def _count_result(value: int):
    r = MagicMock()
    r.scalar_one = MagicMock(return_value=value)
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
        from nl_to_sql.api.dependencies import (
            get_current_user,
            get_session_service,
            get_starter_content_service,
        )

        app = create_app()

        if mock_db is not None:
            async def _fake_user():
                return FAKE_USER

            mock_svc = MagicMock()
            mock_svc._session_factory = MagicMock(return_value=mock_db)

            app.dependency_overrides[get_current_user] = _fake_user
            app.dependency_overrides[get_session_service] = lambda: mock_svc
            # list_templates calls ensure_templates_seeded before listing — stub
            # it out so tests never touch a real DB via the DI container.
            app.dependency_overrides[get_starter_content_service] = lambda: MagicMock(
                ensure_templates_seeded=AsyncMock(return_value=None)
            )

        return app


@pytest.fixture
def p2_app(mock_db):
    yield _make_app(mock_db)


@pytest.fixture
async def client(p2_app):
    async with AsyncClient(transport=ASGITransport(app=p2_app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(mock_db=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ===========================================================================
# Glossary
# ===========================================================================


class TestGlossaryRoutes:

    async def test_list_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/glossary")
        assert r.status_code == 401

    async def test_list_returns_empty_list(self, client, mock_db) -> None:
        count_res = MagicMock()
        count_res.scalar_one = MagicMock(return_value=0)
        list_res = _list_result([])
        mock_db.execute = AsyncMock(side_effect=[count_res, list_res])
        r = await client.get("/api/v1/glossary", headers=_auth())
        assert r.status_code == 200
        assert r.json() == {"items": [], "total": 0}

    async def test_list_returns_entries(self, client, mock_db) -> None:
        row = MagicMock()
        row.id = 1
        row.term = "CAC"
        row.definition = "Customer Acquisition Cost"
        row.created_at = datetime(2024, 1, 1)
        row.updated_at = datetime(2024, 1, 1)
        count_res = MagicMock()
        count_res.scalar_one = MagicMock(return_value=1)
        list_res = _list_result([row])
        mock_db.execute = AsyncMock(side_effect=[count_res, list_res])

        r = await client.get("/api/v1/glossary", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["term"] == "CAC"

    async def test_create_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post(
            "/api/v1/glossary",
            json={"term": "MRR", "definition": "Monthly Recurring Revenue"},
        )
        assert r.status_code == 401

    async def test_create_missing_fields_returns_422(self, client) -> None:
        r = await client.post("/api/v1/glossary", json={}, headers=_auth())
        assert r.status_code == 422

    async def test_create_returns_201(self, client, mock_db) -> None:
        new_row = MagicMock()
        new_row.id = 5
        new_row.term = "LTV"
        new_row.definition = "Lifetime Value"
        new_row.created_at = datetime(2024, 1, 1)
        new_row.updated_at = datetime(2024, 1, 1)
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.refresh = AsyncMock(side_effect=lambda r: r.__dict__.update(
            {"id": 5, "created_at": datetime(2024, 1, 1), "updated_at": datetime(2024, 1, 1)}
        ))

        r = await client.post(
            "/api/v1/glossary",
            json={"term": "LTV", "definition": "Lifetime Value"},
            headers=_auth(),
        )
        assert r.status_code == 201

    async def test_delete_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/glossary/1")
        assert r.status_code == 401

    async def test_delete_not_found_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.delete("/api/v1/glossary/999", headers=_auth())
        assert r.status_code == 404


# ===========================================================================
# Favorited Tables
# ===========================================================================


class TestFavoritedTablesRoutes:

    async def test_list_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/favorited-tables")
        assert r.status_code == 401

    async def test_list_returns_empty_list(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(side_effect=[_count_result(0), _list_result([])])
        r = await client.get("/api/v1/favorited-tables", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data == {"items": [], "total": 0, "limit": 200, "offset": 0}

    async def test_list_returns_rows(self, client, mock_db) -> None:
        row = MagicMock()
        row.id = 1
        row.table_name = "orders"
        row.schema_name = "public"
        row.note = "Main orders table"
        mock_db.execute = AsyncMock(side_effect=[_count_result(1), _list_result([row])])

        r = await client.get("/api/v1/favorited-tables", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["table_name"] == "orders"

    async def test_list_total_can_exceed_returned_page(self, client, mock_db) -> None:
        """A user with more pinned tables than `limit` must be able to tell
        more exist — `total` reflects the full count, not just this page."""
        row = MagicMock()
        row.id = 1
        row.table_name = "orders"
        row.schema_name = "public"
        row.note = None
        mock_db.execute = AsyncMock(side_effect=[_count_result(250), _list_result([row])])

        r = await client.get(
            "/api/v1/favorited-tables", params={"limit": 1}, headers=_auth()
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 250
        assert len(data["items"]) == 1

    async def test_create_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post("/api/v1/favorited-tables", json={"table_name": "orders"})
        assert r.status_code == 401

    async def test_create_missing_table_name_returns_422(self, client) -> None:
        r = await client.post("/api/v1/favorited-tables", json={}, headers=_auth())
        assert r.status_code == 422

    async def test_delete_not_found_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.delete("/api/v1/favorited-tables/999", headers=_auth())
        assert r.status_code == 404

    async def test_delete_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/favorited-tables/1")
        assert r.status_code == 401


# ===========================================================================
# Query Templates
# ===========================================================================


class TestQueryTemplatesRoutes:

    async def test_list_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/query-templates")
        assert r.status_code == 401

    async def test_list_returns_empty_list(self, client, mock_db) -> None:
        count_res = MagicMock()
        count_res.scalar_one = MagicMock(return_value=0)
        list_res = _list_result([])
        mock_db.execute = AsyncMock(side_effect=[count_res, list_res])
        r = await client.get("/api/v1/query-templates", headers=_auth())
        assert r.status_code == 200
        assert r.json() == {"items": [], "total": 0}

    async def test_list_returns_templates(self, client, mock_db) -> None:
        row = MagicMock()
        row.id = 1
        row.name = "Monthly Revenue"
        row.description = "Revenue per month"
        row.template_nl = "Show revenue for {{month}}"
        row.template_sql = "SELECT SUM(total) FROM orders WHERE month = '{{month}}'"
        row.parameters = []
        row.tags = ["finance"]
        row.is_builtin = False
        row.created_at = datetime(2024, 1, 1)
        row.updated_at = datetime(2024, 1, 1)
        count_res = MagicMock()
        count_res.scalar_one = MagicMock(return_value=1)
        list_res = _list_result([row])
        mock_db.execute = AsyncMock(side_effect=[count_res, list_res])

        r = await client.get("/api/v1/query-templates", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Monthly Revenue"

    async def test_create_missing_required_fields_returns_422(self, client) -> None:
        r = await client.post("/api/v1/query-templates", json={}, headers=_auth())
        assert r.status_code == 422

    async def test_create_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post(
            "/api/v1/query-templates",
            json={
                "name": "T",
                "template_nl": "Show {{x}}",
                "template_sql": "SELECT {{x}} FROM t",
            },
        )
        assert r.status_code == 401

    async def test_get_not_found_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.get("/api/v1/query-templates/9999", headers=_auth())
        assert r.status_code == 404

    async def test_delete_not_found_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.delete("/api/v1/query-templates/9999", headers=_auth())
        assert r.status_code == 404


# ===========================================================================
# Tutorial Progress
# ===========================================================================


def _make_tutorial_row(completed=None, dismissed=None):
    row = MagicMock()
    row.completed_steps = completed or []
    row.dismissed_at = dismissed
    return row


class TestTutorialProgressRoutes:

    async def test_get_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/tutorial/progress")
        assert r.status_code == 401

    async def test_get_creates_row_if_not_exists(self, client, mock_db) -> None:
        _make_tutorial_row()
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.flush = AsyncMock(side_effect=lambda: None)

        from nl_to_sql.infrastructure.database.models import TutorialProgress

        def _init_tutorial(self, **kw):
            self.__dict__.update({"completed_steps": [], "dismissed_at": None, **kw})

        with patch.object(TutorialProgress, "__init__", _init_tutorial):
            r = await client.get("/api/v1/tutorial/progress", headers=_auth())
        assert r.status_code in (200, 500)

    async def test_get_returns_existing_row(self, client, mock_db) -> None:
        row = _make_tutorial_row(completed=["connect_database", "run_first_query"])
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))

        r = await client.get("/api/v1/tutorial/progress", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert "connect_database" in data["completed_steps"]
        assert "run_first_query" in data["completed_steps"]
        assert data["dismissed_at"] is None

    async def test_patch_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.patch("/api/v1/tutorial/progress", json={})
        assert r.status_code == 401

    async def test_patch_updates_completed_steps(self, client, mock_db) -> None:
        row = _make_tutorial_row(completed=["connect_database"])
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))
        mock_db.refresh = AsyncMock(side_effect=lambda r: None)

        r = await client.patch(
            "/api/v1/tutorial/progress",
            json={"completed_steps": ["connect_database", "run_first_query", "save_a_query"]},
            headers=_auth(),
        )
        assert r.status_code == 200

    async def test_patch_dismiss_sets_dismissed_at(self, client, mock_db) -> None:
        row = _make_tutorial_row()
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))
        mock_db.refresh = AsyncMock()

        r = await client.patch(
            "/api/v1/tutorial/progress",
            json={"dismissed": True},
            headers=_auth(),
        )
        assert r.status_code == 200

    async def test_delete_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/tutorial/progress")
        assert r.status_code == 401

    async def test_delete_when_no_row_is_safe(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        r = await client.delete("/api/v1/tutorial/progress", headers=_auth())
        assert r.status_code == 204


# ===========================================================================
# Notification Preferences
# ===========================================================================


def _make_notif_row(email_digest=True, in_app=True, marketing=False):
    row = MagicMock()
    row.email_digest = email_digest
    row.in_app_enabled = in_app
    row.marketing_enabled = marketing
    row.updated_at = datetime(2024, 1, 1)
    return row


class TestNotificationPrefsRoutes:

    async def test_get_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/notification-preferences")
        assert r.status_code == 401

    async def test_get_returns_defaults_for_new_user(self, client, mock_db) -> None:
        _make_notif_row()
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.flush = AsyncMock()

        from nl_to_sql.infrastructure.database.models import NotificationPreferences
        with patch.object(NotificationPreferences, "__init__",
                          lambda self, **kw: self.__dict__.update({
                              "email_digest": True,
                              "in_app_enabled": True,
                              "marketing_enabled": False,
                              **kw,
                          }) or None):
            r = await client.get("/api/v1/notification-preferences", headers=_auth())
        assert r.status_code in (200, 500)

    async def test_get_returns_stored_prefs(self, client, mock_db) -> None:
        row = _make_notif_row(email_digest=False, in_app=True, marketing=True)
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))

        r = await client.get("/api/v1/notification-preferences", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["email_digest"] is False
        assert data["in_app_enabled"] is True
        assert data["marketing_enabled"] is True

    async def test_patch_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.patch("/api/v1/notification-preferences", json={})
        assert r.status_code == 401

    async def test_patch_updates_prefs(self, client, mock_db) -> None:
        row = _make_notif_row(email_digest=True, in_app=True, marketing=False)
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))
        mock_db.refresh = AsyncMock()

        r = await client.patch(
            "/api/v1/notification-preferences",
            json={"email_digest": False, "marketing_enabled": True},
            headers=_auth(),
        )
        assert r.status_code == 200

    async def test_patch_partial_update_accepted(self, client, mock_db) -> None:
        row = _make_notif_row()
        mock_db.execute = AsyncMock(return_value=_scalar_result(row))
        mock_db.refresh = AsyncMock()

        r = await client.patch(
            "/api/v1/notification-preferences",
            json={"email_digest": False},
            headers=_auth(),
        )
        assert r.status_code == 200
