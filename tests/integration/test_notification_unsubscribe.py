"""Integration tests for the public one-click digest unsubscribe endpoint."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.services.digest_service import make_unsubscribe_token


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    return db


def _make_app(mock_db):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_session_service

        app = create_app()
        svc = MagicMock()
        svc._session_factory = MagicMock(return_value=mock_db)
        app.dependency_overrides[get_session_service] = lambda: svc
        return app


@pytest.fixture
async def client(mock_db):
    app = _make_app(mock_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestUnsubscribe:
    async def test_valid_token_unsubscribes(self, client, mock_db) -> None:
        # Existing prefs row with digest on → should be flipped off.
        prefs = MagicMock()
        prefs.email_digest = True
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=prefs)
        mock_db.execute = AsyncMock(return_value=result)

        token = make_unsubscribe_token("user-xyz")
        r = await client.get(f"/api/v1/notifications/unsubscribe?token={token}")

        assert r.status_code == 200
        assert "unsubscribed" in r.text.lower()
        assert prefs.email_digest is False
        mock_db.commit.assert_awaited()

    async def test_invalid_token_returns_400(self, client) -> None:
        r = await client.get("/api/v1/notifications/unsubscribe?token=bogus")
        assert r.status_code == 400
        assert "invalid" in r.text.lower()

    async def test_missing_token_returns_422(self, client) -> None:
        r = await client.get("/api/v1/notifications/unsubscribe")
        assert r.status_code == 422

    async def test_creates_row_when_absent(self, client, mock_db) -> None:
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        mock_db.execute = AsyncMock(return_value=result)

        token = make_unsubscribe_token("new-user")
        r = await client.get(f"/api/v1/notifications/unsubscribe?token={token}")

        assert r.status_code == 200
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited()
