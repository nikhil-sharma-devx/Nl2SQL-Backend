"""Integration tests for Export & Share routes.

  - POST   /api/v1/exports/query        (export CSV/JSON/SQL/PDF)
  - POST   /api/v1/shares               (create share link)
  - GET    /api/v1/shares/{token}       (public snapshot: 200/410/404)
  - DELETE /api/v1/shares/{share_id}    (owner-only revoke)
  - POST   /api/v1/shares/{id}/email    (graceful when unconfigured)
  - POST   /api/v1/shares/{id}/slack    (graceful when unconfigured)

All DB calls are mocked via FastAPI dependency_overrides.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import SharedQuery
from nl_to_sql.services.export_service import make_share_token

USER_ID = "test-user-exp-001"

FAKE_USER = UserPublic(
    id=USER_ID,
    email="exp@example.com",
    full_name="Exp Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)

DT = datetime(2024, 1, 1, 12, 0, 0)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _make_share(
    *,
    id: str = "share-1",
    title: str | None = "My Share",
    revoked_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> SharedQuery:
    s = SharedQuery()
    s.id = id
    s.user_id = USER_ID
    s.title = title
    s.question = "show me all users"
    s.nl_prompt = "show me all users"
    s.generated_sql = "SELECT * FROM users"
    s.result_snapshot = [{"id": 1, "name": "Alice"}]
    s.created_at = DT
    s.expires_at = expires_at
    s.revoked_at = revoked_at
    return s


def _scalar(value):
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    return r


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


def _make_app(mock_db=None, authed=True):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_current_user,
            get_export_service,
            get_session_service,
        )
        from nl_to_sql.config.settings import get_settings
        from nl_to_sql.services.export_service import ExportService

        app = create_app()
        if authed:
            async def _fake_user():
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user
        if mock_db is not None:
            svc = MagicMock()
            svc._session_factory = MagicMock(return_value=mock_db)
            app.dependency_overrides[get_session_service] = lambda: svc

        # Force unconfigured delivery channels so tests never hit real SMTP/Slack
        # regardless of the developer's local .env.
        unconfigured = ExportService(
            settings=get_settings().model_copy(
                update={"smtp_username": "", "smtp_password": "", "slack_webhook_url": ""}
            )
        )
        app.dependency_overrides[get_export_service] = lambda: unconfigured
        return app


@pytest.fixture
async def client(mock_db):
    app = _make_app(mock_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(mock_db=None, authed=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Export ────────────────────────────────────────────────────────────────────


class TestExportQuery:
    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post(
            "/api/v1/exports/query", json={"sql": "SELECT 1", "format": "csv"}
        )
        assert r.status_code == 401

    @pytest.mark.parametrize(
        ("fmt", "ctype"),
        [
            ("csv", "text/csv"),
            ("json", "application/json"),
            ("sql", "application/sql"),
            ("pdf", "application/pdf"),
        ],
    )
    async def test_content_types(self, client, fmt, ctype) -> None:
        r = await client.post(
            "/api/v1/exports/query",
            json={
                "sql": "SELECT * FROM users",
                "question": "all users",
                "rows": [{"id": 1, "name": "Alice"}],
                "format": fmt,
            },
            headers=_auth(),
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(ctype)
        assert "attachment" in r.headers["content-disposition"]

    async def test_pdf_is_valid_pdf(self, client) -> None:
        r = await client.post(
            "/api/v1/exports/query",
            json={"sql": "SELECT 1", "rows": [], "format": "pdf"},
            headers=_auth(),
        )
        assert r.status_code == 200
        assert r.content.startswith(b"%PDF")


# ── Create share ────────────────────────────────────────────────────────────


class TestCreateShare:
    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post("/api/v1/shares", json={"sql": "SELECT 1"})
        assert r.status_code == 401

    async def test_create_returns_201(self, client, mock_db) -> None:
        r = await client.post(
            "/api/v1/shares",
            json={"sql": "SELECT 1", "question": "q", "rows": [{"a": 1}]},
            headers=_auth(),
        )
        assert r.status_code == 201
        data = r.json()
        assert data["id"]
        assert data["token"]
        assert data["url"].endswith(data["token"])
        assert data["expires_at"] is None
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    async def test_create_with_expiry(self, client, mock_db) -> None:
        r = await client.post(
            "/api/v1/shares",
            json={"sql": "SELECT 1", "expires_in_days": 7},
            headers=_auth(),
        )
        assert r.status_code == 201
        assert r.json()["expires_at"] is not None


# ── Public GET share ────────────────────────────────────────────────────────


class TestGetSharedQuery:
    async def test_valid_returns_200(self, client, mock_db) -> None:
        share = _make_share()
        mock_db.execute = AsyncMock(return_value=_scalar(share))
        token = make_share_token(share.id, None)

        r = await client.get(f"/api/v1/shares/{token}")
        assert r.status_code == 200
        data = r.json()
        assert data["sql"] == "SELECT * FROM users"
        assert data["results"] == [{"id": 1, "name": "Alice"}]
        # No owner PII leaked.
        assert "user_id" not in data

    async def test_unknown_token_returns_404(self, client) -> None:
        r = await client.get("/api/v1/shares/bogus-token")
        assert r.status_code == 404
        assert "message" in r.json()

    async def test_valid_signature_missing_row_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar(None))
        token = make_share_token("gone", None)
        r = await client.get(f"/api/v1/shares/{token}")
        assert r.status_code == 404

    async def test_revoked_returns_410(self, client, mock_db) -> None:
        share = _make_share(revoked_at=DT)
        mock_db.execute = AsyncMock(return_value=_scalar(share))
        token = make_share_token(share.id, None)
        r = await client.get(f"/api/v1/shares/{token}")
        assert r.status_code == 410
        assert "message" in r.json()

    async def test_expired_returns_410(self, client, mock_db) -> None:
        past = datetime.utcnow() - timedelta(days=1)
        share = _make_share(expires_at=past)
        mock_db.execute = AsyncMock(return_value=_scalar(share))
        token = make_share_token(share.id, past)
        r = await client.get(f"/api/v1/shares/{token}")
        assert r.status_code == 410


# ── Revoke share (owner-only) ────────────────────────────────────────────────


class TestRevokeShare:
    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/shares/share-1")
        assert r.status_code == 401

    async def test_revoke_returns_204(self, client, mock_db) -> None:
        share = _make_share()
        mock_db.execute = AsyncMock(return_value=_scalar(share))
        r = await client.delete("/api/v1/shares/share-1", headers=_auth())
        assert r.status_code == 204
        assert share.revoked_at is not None
        mock_db.commit.assert_awaited_once()

    async def test_cross_user_returns_404(self, client, mock_db) -> None:
        # The query filters on user_id, so another user's share is invisible.
        mock_db.execute = AsyncMock(return_value=_scalar(None))
        r = await client.delete("/api/v1/shares/someone-elses", headers=_auth())
        assert r.status_code == 404
        assert "message" in r.json()


# ── Email / Slack delivery (graceful when unconfigured) ──────────────────────


class TestDelivery:
    async def test_email_graceful_when_unconfigured(self, client, mock_db) -> None:
        share = _make_share()
        mock_db.execute = AsyncMock(return_value=_scalar(share))
        r = await client.post(
            "/api/v1/shares/share-1/email",
            json={"to_email": "friend@example.com"},
            headers=_auth(),
        )
        assert r.status_code == 200
        assert r.json()["sent"] is False

    async def test_email_cross_user_returns_404(self, client, mock_db) -> None:
        mock_db.execute = AsyncMock(return_value=_scalar(None))
        r = await client.post(
            "/api/v1/shares/nope/email",
            json={"to_email": "friend@example.com"},
            headers=_auth(),
        )
        assert r.status_code == 404

    async def test_slack_graceful_when_unconfigured(self, client, mock_db) -> None:
        share = _make_share()
        mock_db.execute = AsyncMock(return_value=_scalar(share))
        r = await client.post("/api/v1/shares/share-1/slack", headers=_auth())
        assert r.status_code == 200
        assert r.json()["sent"] is False
