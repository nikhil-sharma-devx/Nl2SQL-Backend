"""Integration tests for Connection routes (api/routes/connections.py):

  GET/POST        /api/v1/connections
  PUT/DELETE      /api/v1/connections/{id}
  POST            /api/v1/connections/{id}/test
  POST            /api/v1/connections/{id}/select

``get_current_user`` and ``get_connection_service`` are real FastAPI
dependencies, so they are overridden via ``app.dependency_overrides``. The
service is mocked; domain exceptions raised by the service are mapped to the
canonical error envelope (``"message"`` / ``"code"``) by the global handler.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.exceptions import (
    ConnectionNotFoundError,
    ConnectionTestError,
    ConnectionValidationError,
)
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.connection_service import ConnectionInfo

USER_ID = "conn-routes-user-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="conntest@example.com",
    full_name="Conn Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _info(connection_id="c1", name="Prod", *, is_default=False, has_dsn=True, preview="postgresql+asyncpg://u:***@h/db"):
    return ConnectionInfo(
        connection_id=connection_id,
        name=name,
        db_type="postgresql",
        is_default=is_default,
        has_dsn=has_dsn,
        url_preview=preview,
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


def _make_app(*, authed: bool = True, svc: MagicMock | None = None):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_connection_service,
            get_current_user,
        )

        app = create_app()
        if authed:
            async def _fake_user():
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user
        if svc is not None:
            app.dependency_overrides[get_connection_service] = lambda: svc
        return app


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _svc(**methods) -> MagicMock:
    svc = MagicMock()
    for name, val in methods.items():
        setattr(svc, name, AsyncMock(return_value=val))
    return svc


def _auth() -> dict:
    return {"Authorization": "Bearer fake"}


class TestListConnections:
    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False, svc=_svc(list_connections=[]))
        async with _client(app) as c:
            r = await c.get("/api/v1/connections")
        assert r.status_code == 401

    async def test_lists_without_leaking_dsn(self) -> None:
        svc = _svc(list_connections=[_info(is_default=True)])
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.get("/api/v1/connections", headers=_auth())
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        # Only a masked preview is exposed — never a raw DSN field.
        assert "***" in body[0]["url_preview"]
        assert "encrypted_url" not in body[0]
        assert body[0]["is_default"] is True


class TestCreateConnection:
    async def test_happy_path(self) -> None:
        svc = _svc(create=_info())
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post(
                "/api/v1/connections",
                json={"name": "Prod", "database_url": "postgresql://u:p@h:5432/db"},
                headers=_auth(),
            )
        assert r.status_code == 200
        svc.create.assert_awaited_once()
        assert r.json()["name"] == "Prod"

    async def test_validation_error_maps_to_400(self) -> None:
        svc = _svc()
        svc.create = AsyncMock(side_effect=ConnectionValidationError("duplicate name"))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post(
                "/api/v1/connections",
                json={"name": "Prod", "database_url": "postgresql://u:p@h/db"},
                headers=_auth(),
            )
        assert r.status_code == 400
        assert r.json()["code"] == "CONNECTION_INVALID"

    async def test_bad_body_returns_422(self) -> None:
        app = _make_app(svc=_svc())
        async with _client(app) as c:
            r = await c.post("/api/v1/connections", json={"name": ""}, headers=_auth())
        assert r.status_code == 422


class TestUpdateConnection:
    async def test_rename(self) -> None:
        svc = _svc(update=_info(name="Production"))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.put(
                "/api/v1/connections/c1", json={"name": "Production"}, headers=_auth()
            )
        assert r.status_code == 200
        assert r.json()["name"] == "Production"

    async def test_cross_user_is_404(self) -> None:
        svc = _svc()
        svc.update = AsyncMock(side_effect=ConnectionNotFoundError("Connection not found."))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.put(
                "/api/v1/connections/other", json={"name": "x"}, headers=_auth()
            )
        assert r.status_code == 404
        assert r.json()["code"] == "CONNECTION_NOT_FOUND"


class TestDeleteConnection:
    async def test_delete(self) -> None:
        svc = _svc(delete=None)
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.delete("/api/v1/connections/c1", headers=_auth())
        assert r.status_code == 200
        svc.delete.assert_awaited_once_with(USER_ID, "c1")


class TestTestConnection:
    async def test_ok(self) -> None:
        svc = _svc(test=None)
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/connections/c1/test", headers=_auth())
        assert r.status_code == 200
        assert r.json()["ok"] is True

    async def test_unreachable_maps_to_400(self) -> None:
        svc = _svc()
        svc.test = AsyncMock(side_effect=ConnectionTestError("connection refused"))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/connections/c1/test", headers=_auth())
        assert r.status_code == 400
        assert r.json()["code"] == "CONNECTION_TEST_FAILED"


class TestSelectConnection:
    async def test_select(self) -> None:
        svc = _svc(set_default=_info(is_default=True))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/connections/c1/select", headers=_auth())
        assert r.status_code == 200
        assert r.json()["is_default"] is True
        svc.set_default.assert_awaited_once_with(USER_ID, "c1")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
