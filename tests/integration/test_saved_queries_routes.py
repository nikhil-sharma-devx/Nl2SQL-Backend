"""Integration tests for Saved Queries routes (F3):
  - GET    /api/v1/saved-queries           (list)
  - POST   /api/v1/saved-queries           (create)
  - GET    /api/v1/saved-queries/{id}      (get)
  - PATCH  /api/v1/saved-queries/{id}      (update)
  - DELETE /api/v1/saved-queries/{id}      (delete)
  - POST   /api/v1/saved-queries/{id}/run  (run)

All DB calls are mocked via FastAPI dependency_overrides.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import SavedQuery

# ── Constants ─────────────────────────────────────────────────────────────────

USER_ID = "test-user-sq-001"
USER_EMAIL = "sqtest@example.com"

FAKE_USER = UserPublic(
    id=USER_ID,
    email=USER_EMAIL,
    full_name="SQ Test User",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)

DT = datetime(2024, 1, 1, 12, 0, 0)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_saved_query(
    *,
    id: int = 1,
    title: str | None = "My Query",
    nl_prompt: str = "show me all users",
    generated_sql: str = "SELECT 1",
    dialect: str | None = "postgresql",
    starred: bool = False,
    last_run_at: datetime | None = None,
    run_count: int = 0,
) -> SavedQuery:
    """Build a real SavedQuery ORM instance with all typed fields set."""
    q = SavedQuery()
    q.id = id
    q.user_id = USER_ID
    q.title = title
    q.nl_prompt = nl_prompt
    q.generated_sql = generated_sql
    q.dialect = dialect
    q.starred = starred
    q.last_run_at = last_run_at
    q.run_count = run_count
    q.created_at = DT
    q.updated_at = DT
    return q


def _scalar_one_or_none_result(value):
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    return r


def _count_result(total: int):
    r = MagicMock()
    r.scalar_one = MagicMock(return_value=total)
    return r


def _rows_result(items: list):
    r = MagicMock()
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=items)))
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
def sq_app(mock_db):
    yield _make_app(mock_db)


@pytest.fixture
async def client(sq_app):
    async with AsyncClient(transport=ASGITransport(app=sq_app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(mock_db=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ===========================================================================
# GET /api/v1/saved-queries  (list)
# ===========================================================================


class TestListSavedQueries:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/saved-queries")
        assert r.status_code == 401

    async def test_returns_empty_list(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(side_effect=[_count_result(0), _rows_result([])])
        r = await client.get("/api/v1/saved-queries", headers=_auth())
        assert r.status_code == 200
        assert r.json() == {"items": [], "total": 0}

    async def test_returns_items(self, client, mock_db) -> None:
        row = _make_saved_query(id=7, title="Revenue", starred=True)
        mock_db.execute = AsyncMock(side_effect=[_count_result(1), _rows_result([row])])

        r = await client.get("/api/v1/saved-queries", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["id"] == 7
        assert item["title"] == "Revenue"
        assert item["starred"] is True
        assert item["run_count"] == 0

    async def test_accepts_query_params(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(side_effect=[_count_result(0), _rows_result([])])
        r = await client.get(
            "/api/v1/saved-queries",
            params={"search": "rev", "starred": "true", "limit": 5, "offset": 10},
            headers=_auth(),
        )
        assert r.status_code == 200

    async def test_invalid_limit_zero_returns_422(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(side_effect=[_count_result(0), _rows_result([])])
        r = await client.get("/api/v1/saved-queries", params={"limit": 0}, headers=_auth())
        assert r.status_code == 422

    async def test_invalid_limit_too_large_returns_422(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(side_effect=[_count_result(0), _rows_result([])])
        r = await client.get("/api/v1/saved-queries", params={"limit": 101}, headers=_auth())
        assert r.status_code == 422


# ===========================================================================
# POST /api/v1/saved-queries  (create)
# ===========================================================================


class TestCreateSavedQuery:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post(
            "/api/v1/saved-queries",
            json={"nl_prompt": "p", "generated_sql": "SELECT 1"},
        )
        assert r.status_code == 401

    async def test_missing_required_fields_returns_422(self, client) -> None:
        r = await client.post("/api/v1/saved-queries", json={}, headers=_auth())
        assert r.status_code == 422

    async def test_missing_generated_sql_returns_422(self, client) -> None:
        r = await client.post(
            "/api/v1/saved-queries",
            json={"nl_prompt": "only prompt"},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_create_returns_201(self, client, mock_db) -> None:
        def _refresh(obj):
            obj.id = 1
            obj.created_at = DT
            obj.updated_at = DT
            obj.run_count = 0
            obj.starred = False
            obj.last_run_at = None

        mock_db.refresh = AsyncMock(side_effect=_refresh)

        r = await client.post(
            "/api/v1/saved-queries",
            json={
                "title": "New Q",
                "nl_prompt": "show me all users",
                "generated_sql": "SELECT * FROM users",
                "dialect": "postgresql",
            },
            headers=_auth(),
        )
        assert r.status_code == 201
        data = r.json()
        assert data["id"] == 1
        assert data["title"] == "New Q"
        assert data["nl_prompt"] == "show me all users"
        assert data["generated_sql"] == "SELECT * FROM users"
        assert data["starred"] is False
        assert data["run_count"] == 0
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()


# ===========================================================================
# GET /api/v1/saved-queries/{id}  (get)
# ===========================================================================


class TestGetSavedQuery:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/saved-queries/1")
        assert r.status_code == 401

    async def test_found_returns_200(self, client, mock_db) -> None:
        row = _make_saved_query(id=3, title="Found")
        mock_db.execute = AsyncMock(return_value=_scalar_one_or_none_result(row))

        r = await client.get("/api/v1/saved-queries/3", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 3
        assert data["title"] == "Found"

    async def test_not_found_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_one_or_none_result(None))
        r = await client.get("/api/v1/saved-queries/999", headers=_auth())
        assert r.status_code == 404
        assert "message" in r.json()


# ===========================================================================
# PATCH /api/v1/saved-queries/{id}  (update)
# ===========================================================================


class TestPatchSavedQuery:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.patch("/api/v1/saved-queries/1", json={"starred": True})
        assert r.status_code == 401

    async def test_update_returns_200(self, client, mock_db) -> None:
        row = _make_saved_query(id=4, starred=False)
        mock_db.execute = AsyncMock(return_value=_scalar_one_or_none_result(row))
        mock_db.refresh = AsyncMock(side_effect=lambda obj: None)

        r = await client.patch(
            "/api/v1/saved-queries/4",
            json={"starred": True, "title": "Renamed"},
            headers=_auth(),
        )
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 4
        assert data["starred"] is True
        assert data["title"] == "Renamed"

    async def test_not_found_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_one_or_none_result(None))
        r = await client.patch(
            "/api/v1/saved-queries/999",
            json={"starred": True},
            headers=_auth(),
        )
        assert r.status_code == 404
        assert "message" in r.json()


# ===========================================================================
# DELETE /api/v1/saved-queries/{id}  (delete)
# ===========================================================================


class TestDeleteSavedQuery:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/saved-queries/1")
        assert r.status_code == 401

    async def test_delete_returns_204(self, client, mock_db) -> None:
        row = _make_saved_query(id=5)
        mock_db.execute = AsyncMock(return_value=_scalar_one_or_none_result(row))

        r = await client.delete("/api/v1/saved-queries/5", headers=_auth())
        assert r.status_code == 204
        mock_db.delete.assert_awaited_once()
        mock_db.commit.assert_awaited_once()

    async def test_not_found_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_one_or_none_result(None))
        r = await client.delete("/api/v1/saved-queries/999", headers=_auth())
        assert r.status_code == 404
        assert "message" in r.json()


# ===========================================================================
# POST /api/v1/saved-queries/{id}/run  (run)
# ===========================================================================


class TestRunSavedQuery:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post("/api/v1/saved-queries/1/run")
        assert r.status_code == 401

    async def test_run_returns_200(self, client, mock_db) -> None:
        row = _make_saved_query(id=6, generated_sql="SELECT 42", run_count=2)
        mock_db.execute = AsyncMock(return_value=_scalar_one_or_none_result(row))
        mock_db.refresh = AsyncMock(side_effect=lambda obj: None)

        r = await client.post("/api/v1/saved-queries/6/run", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["query_id"] == 6
        assert data["generated_sql"] == "SELECT 42"
        assert data["run_count"] == 3
        assert data["last_run_at"] is not None

    async def test_not_found_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_one_or_none_result(None))
        r = await client.post("/api/v1/saved-queries/999/run", headers=_auth())
        assert r.status_code == 404
        assert "message" in r.json()
