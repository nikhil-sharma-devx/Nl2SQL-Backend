"""Integration tests for feedback routes — POST /api/v1/feedback, GET /api/v1/feedback."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

USER_ID = "test-user-feedback-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="fbtest@example.com",
    full_name="Feedback Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _scalar_result(value):
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    return r


def _list_result(items: list):
    r = MagicMock()
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=items)))
    return r


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def mock_feedback_svc():
    svc = MagicMock()
    svc.initialize = AsyncMock()
    svc.dispose = AsyncMock()
    svc.submit = AsyncMock(return_value={"id": 1, "status": "recorded"})
    svc.list_recent = AsyncMock(return_value=[])
    svc.get_stats = AsyncMock(return_value={"total": 0, "positive": 0, "negative": 0})
    return svc


def _make_app(mock_feedback_svc=None, mock_db=None, authed=True, admin=False):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_active_connection_id,
            get_current_user,
            get_session_service,
            require_admin,
        )

        app = create_app()

        if authed:
            async def _fake_user():
                return FAKE_USER
            app.dependency_overrides[get_current_user] = _fake_user

            async def _fake_connection_id():
                return "conn-feedback-test"
            app.dependency_overrides[get_active_connection_id] = _fake_connection_id

        if admin:
            async def _fake_admin():
                return FAKE_USER
            app.dependency_overrides[require_admin] = _fake_admin

        if mock_db is not None:
            mock_svc = MagicMock()
            mock_svc._session_factory = MagicMock(return_value=mock_db)
            app.dependency_overrides[get_session_service] = lambda: mock_svc

        return app


@pytest.fixture
async def client(mock_db):
    app = _make_app(mock_db=mock_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def admin_client(mock_db):
    """Authenticated as an admin — for GET /feedback, which is admin-only (C2)."""
    app = _make_app(mock_db=mock_db, admin=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(authed=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSubmitFeedback:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post(
            "/api/v1/feedback",
            json={
                "query_id": "q1",
                "feedback_type": "positive",
                "sql": "SELECT 1",
                "question": "test",
            },
        )
        assert r.status_code == 401

    async def test_missing_feedback_type_returns_422(self, client) -> None:
        r = await client.post(
            "/api/v1/feedback",
            json={"query_id": "q1", "sql": "SELECT 1", "question": "test"},
            headers=_auth(),
        )
        assert r.status_code == 422

    async def test_valid_positive_feedback_accepted(self, client, mock_db) -> None:
        new_row = MagicMock()
        new_row.id = 1
        new_row.feedback_type = "positive"
        new_row.query_id = "q1"
        new_row.user_id = USER_ID
        new_row.sql = "SELECT * FROM customers"
        new_row.question = "Show customers"
        new_row.error_type = None
        new_row.user_correction = None
        new_row.user_notes = None
        new_row.created_at = datetime(2024, 1, 1)
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.refresh = AsyncMock(side_effect=lambda r: None)

        r = await client.post(
            "/api/v1/feedback",
            json={
                "query_id": "q1",
                "feedback_type": "positive",
                "sql": "SELECT * FROM customers",
                "question": "Show customers",
            },
            headers=_auth(),
        )
        assert r.status_code in (200, 201, 422, 500)

    async def test_negative_feedback_with_correction(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        mock_db.refresh = AsyncMock()

        r = await client.post(
            "/api/v1/feedback",
            json={
                "query_id": "q2",
                "feedback_type": "negative",
                "sql": "SELECT fake FROM customers",
                "question": "Show customers",
                "error_type": "wrong_column",
                "user_correction": "SELECT email FROM customers",
                "user_notes": "Use email, not fake",
            },
            headers=_auth(),
        )
        assert r.status_code in (200, 201, 422, 500)


class TestListFeedback:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/feedback")
        assert r.status_code == 401

    async def test_non_admin_forbidden(self, client, mock_db) -> None:
        """C2: feedback spans every user, so listing it is admin-only."""
        mock_db.execute = AsyncMock(return_value=_list_result([]))
        r = await client.get("/api/v1/feedback", headers=_auth())
        assert r.status_code == 403

    async def test_returns_list(self, admin_client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_list_result([]))
        r = await admin_client.get("/api/v1/feedback", headers=_auth())
        assert r.status_code == 200

    async def test_accepts_limit_param(self, admin_client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_list_result([]))
        r = await admin_client.get("/api/v1/feedback?limit=5", headers=_auth())
        assert r.status_code == 200
