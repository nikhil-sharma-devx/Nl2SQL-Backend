"""Integration tests for chat-session routes — /api/v1/sessions/*.

Regression guard for IMPROVEMENTS.md item 12: GET /sessions/{id} has a
`response_model` (SessionDetailResponse). Its message `id` is an autoincrement
integer and several response columns are nullable, so the model must accept
int ids and NULL fields — a stricter model (e.g. `id: str`) 500s on real data.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

USER_ID = "test-user-sessions-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="sessions@example.com",
    full_name="Sessions Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _message(msg_id: int, *, sparse: bool):
    """Build a ChatMessage-like object. `sparse` mimics a legacy/NULL row."""
    ts = datetime(2026, 1, 1, 0, 0, msg_id % 60)
    if sparse:
        return SimpleNamespace(
            id=msg_id, question="q", timestamp=ts, sql="", dialect="postgresql",
            is_valid=None, validation_errors=None, retrieved_tables=None,
            used_tables=None, execution_result=None, execution_error=None,
            tokens_used=None, cached=None, message=None, intent_type=None,
            suggested_chart=None, follow_up_questions=None,
        )
    return SimpleNamespace(
        id=msg_id, question="show orders", timestamp=ts, sql="SELECT 1",
        dialect="postgresql", is_valid=True, validation_errors=[],
        retrieved_tables=["orders"], used_tables=["orders"],
        execution_result=[{"a": 1}], execution_error=None, tokens_used=250,
        cached=False, message=None, intent_type="select",
        suggested_chart={"type": "bar"}, follow_up_questions=["more?"],
    )


def _make_app(session=None, authed=True, versions=None):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_current_user, get_session_service

        app = create_app()
        if authed:
            async def _fake_user():
                return FAKE_USER
            app.dependency_overrides[get_current_user] = _fake_user

        mock_svc = MagicMock()
        mock_svc.get_session = AsyncMock(return_value=session)

        # Support the raw-session-factory path used by GET /query/versions/{id}.
        db_sess = MagicMock()
        db_sess.__aenter__ = AsyncMock(return_value=db_sess)
        db_sess.__aexit__ = AsyncMock(return_value=False)
        result = MagicMock()
        result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=versions or []))
        )
        db_sess.execute = AsyncMock(return_value=result)
        mock_svc._session_factory = MagicMock(return_value=db_sess)

        app.dependency_overrides[get_session_service] = lambda: mock_svc
        return app


@pytest.fixture
async def unauthed_client():
    app = _make_app(authed=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestGetSession:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/sessions/abc")
        assert r.status_code == 401

    async def test_returns_404_when_missing(self) -> None:
        app = _make_app(session=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/v1/sessions/missing", headers=_auth())
        assert r.status_code == 404

    @pytest.mark.parametrize("owner", [None, "someone-else"])
    async def test_ownerless_or_foreign_session_is_not_found(self, owner) -> None:
        """Legacy sessions with user_id=NULL belong to nobody — they used to pass
        the ownership check (`if session.user_id and ...`) and open for anyone."""
        session = SimpleNamespace(
            id="legacy", user_id=owner, title="Old chat",
            created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2), messages=[],
        )
        app = _make_app(session=session)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            got = await c.get("/api/v1/sessions/legacy", headers=_auth())
            deleted = await c.delete("/api/v1/sessions/legacy", headers=_auth())
        assert got.status_code == 404
        assert deleted.status_code == 404

    async def test_serializes_integer_message_ids_and_null_fields(self) -> None:
        """The exact regression: int message ids + NULL columns must serialize."""
        session = SimpleNamespace(
            id="uuid-1", user_id=USER_ID, title="Chat",
            created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2),
            messages=[_message(87, sparse=True), _message(88, sparse=False)],
        )
        app = _make_app(session=session)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/v1/sessions/uuid-1", headers=_auth())
        assert r.status_code == 200, r.text
        body = r.json()
        assert [m["id"] for m in body["messages"]] == [87, 88]
        # Null-ish legacy row survives serialization untouched.
        assert body["messages"][0]["response"]["tokens_used"] is None
        # Populated row keeps its values.
        assert body["messages"][1]["response"]["execution_result"] == [{"a": 1}]


class TestAddSessionMessage:
    """Low finding: add_message raising SessionNotFoundError (a rare race —
    session deleted between the ownership check and the write) used to
    surface as a 500 instead of 404."""

    async def test_race_deleted_session_returns_404_not_500(self) -> None:
        from nl_to_sql.core.exceptions import SessionNotFoundError

        owned_session = SimpleNamespace(id="sess-1", user_id=USER_ID)
        app = _make_app(session=owned_session)
        from nl_to_sql.api.dependencies import get_session_service

        mock_svc = app.dependency_overrides[get_session_service]()
        mock_svc.add_message = AsyncMock(
            side_effect=SessionNotFoundError("Session sess-1 not found")
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/v1/sessions/sess-1/messages",
                json={
                    "question": "q",
                    "sql": "SELECT 1",
                    "dialect": "postgresql",
                    "is_valid": True,
                },
                headers=_auth(),
            )
        assert r.status_code == 404


class TestSqlVersions:
    """GET /query/versions/{id} — a message with no saved versions is a 200
    empty collection, not a 404 (which item-11's interceptor would toast)."""

    async def test_no_versions_returns_200_empty(self) -> None:
        app = _make_app(versions=[])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/v1/query/versions/123", headers=_auth())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == {"message_id": 123, "versions": [], "total_versions": 0}

    async def test_oversized_client_id_returns_200_empty(self) -> None:
        """A client timestamp id (> int32) must not 500 (asyncpg int4 overflow)."""
        app = _make_app(versions=[])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/v1/query/versions/1783533067258", headers=_auth())
        assert r.status_code == 200, r.text
        assert r.json()["versions"] == []
