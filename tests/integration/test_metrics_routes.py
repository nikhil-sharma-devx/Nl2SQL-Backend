"""Integration tests for Metrics routes (api/routes/metrics.py):

  GET/POST        /api/v1/metrics
  GET/PUT/DELETE  /api/v1/metrics/{id}
  POST            /api/v1/metrics/{id}/certify
  POST            /api/v1/metrics/{id}/uncertify
  POST            /api/v1/metrics/{id}/preview

``get_current_user``, ``get_active_connection_id`` and ``get_metrics_service``
are real FastAPI dependencies, so they are overridden via
``app.dependency_overrides``. The service is mocked; domain exceptions raised
by the service are mapped to the canonical error envelope
(``"message"``/``"code"``) by the global handler.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.exceptions import MetricNotFoundError, MetricValidationError
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.metrics_service import MetricInfo

USER_ID = "metrics-routes-user-001"
CONNECTION_ID = "conn-1"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="metricstest@example.com",
    full_name="Metrics Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _info(metric_id="m1", **overrides) -> MetricInfo:
    defaults = {
        "metric_id": metric_id,
        "connection_id": CONNECTION_ID,
        "name": "Net Revenue",
        "description": "Total revenue minus refunds",
        "sql_definition": "SELECT SUM(amount) FROM orders",
        "dimensions": ["region"],
        "tags": ["finance"],
        "owner": "Finance Team",
        "certified": False,
        "is_builtin": False,
        "validation_errors": [],
        "created_at": datetime(2024, 1, 1),
        "updated_at": datetime(2024, 1, 1),
    }
    defaults.update(overrides)
    return MetricInfo(**defaults)


def _make_app(*, authed: bool = True, svc: MagicMock | None = None):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_active_connection_id,
            get_current_user,
            get_metrics_service,
            get_starter_content_service,
        )

        app = create_app()
        if authed:
            async def _fake_user():
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user

        async def _fake_connection_id():
            return CONNECTION_ID

        app.dependency_overrides[get_active_connection_id] = _fake_connection_id
        app.dependency_overrides[get_starter_content_service] = lambda: MagicMock(
            ensure_metrics_seeded=AsyncMock(return_value=None)
        )
        if svc is not None:
            app.dependency_overrides[get_metrics_service] = lambda: svc
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


class TestListMetrics:
    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False, svc=_svc(list_metrics=([], 0)))
        async with _client(app) as c:
            r = await c.get("/api/v1/metrics")
        assert r.status_code == 401

    async def test_lists_metrics(self) -> None:
        svc = _svc(list_metrics=([_info()], 1))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.get("/api/v1/metrics", headers=_auth())
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["items"][0]["name"] == "Net Revenue"


class TestCreateMetric:
    async def test_happy_path(self) -> None:
        svc = _svc(create=_info())
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post(
                "/api/v1/metrics",
                json={"name": "Net Revenue", "sql_definition": "SELECT SUM(amount) FROM orders"},
                headers=_auth(),
            )
        assert r.status_code == 200
        svc.create.assert_awaited_once()
        assert r.json()["name"] == "Net Revenue"

    async def test_duplicate_name_maps_to_400(self) -> None:
        svc = _svc()
        svc.create = AsyncMock(
            side_effect=MetricValidationError("A metric named 'x' already exists.")
        )
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post(
                "/api/v1/metrics",
                json={"name": "x", "sql_definition": "SELECT 1"},
                headers=_auth(),
            )
        assert r.status_code == 400
        assert r.json()["code"] == "METRIC_INVALID"

    async def test_bad_body_returns_422(self) -> None:
        app = _make_app(svc=_svc())
        async with _client(app) as c:
            r = await c.post("/api/v1/metrics", json={"name": ""}, headers=_auth())
        assert r.status_code == 422


class TestGetMetric:
    async def test_missing_is_404(self) -> None:
        svc = _svc()
        svc.get = AsyncMock(side_effect=MetricNotFoundError("Metric not found."))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.get("/api/v1/metrics/missing", headers=_auth())
        assert r.status_code == 404
        assert r.json()["code"] == "METRIC_NOT_FOUND"


class TestUpdateDelete:
    async def test_update_by_non_creator_is_404(self) -> None:
        svc = _svc()
        svc.update = AsyncMock(side_effect=MetricNotFoundError("Metric not found."))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.put(
                "/api/v1/metrics/m1", json={"name": "New name"}, headers=_auth()
            )
        assert r.status_code == 404

    async def test_delete(self) -> None:
        svc = _svc(delete=None)
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.delete("/api/v1/metrics/m1", headers=_auth())
        assert r.status_code == 200
        svc.delete.assert_awaited_once_with(USER_ID, CONNECTION_ID, "m1")


class TestCertifyUncertify:
    async def test_certify(self) -> None:
        svc = _svc(certify=_info(certified=True))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/metrics/m1/certify", headers=_auth())
        assert r.status_code == 200
        assert r.json()["certified"] is True

    async def test_certify_blocked_by_validation_errors_maps_to_400(self) -> None:
        svc = _svc()
        svc.certify = AsyncMock(
            side_effect=MetricValidationError("Cannot certify with unresolved errors.")
        )
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/metrics/m1/certify", headers=_auth())
        assert r.status_code == 400

    async def test_uncertify(self) -> None:
        svc = _svc(uncertify=_info(certified=False))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/metrics/m1/uncertify", headers=_auth())
        assert r.status_code == 200
        assert r.json()["certified"] is False


class TestPreview:
    async def test_explain_preview(self) -> None:
        svc = _svc(get=_info())
        app = _make_app(svc=svc)
        fake_container = MagicMock()
        # No personal client → the scoped resolver hands back the platform fallback.
        fake_conn_svc = MagicMock(
            get_scoped_client=AsyncMock(side_effect=lambda _user, _conn, fallback: fallback)
        )
        fake_container.connection_service.return_value = fake_conn_svc
        fake_db_client = MagicMock(
            explain=AsyncMock(
                return_value={"supported": True, "estimated_rows": 10, "estimated_cost": 1.5}
            )
        )
        fake_container.db_client.return_value = fake_db_client
        with patch(
            "nl_to_sql.api.dependencies._get_container", return_value=fake_container
        ):
            async with _client(app) as c:
                r = await c.post("/api/v1/metrics/m1/preview", headers=_auth())
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["estimated_rows"] == 10

    async def test_execute_preview(self) -> None:
        svc = _svc(get=_info())
        app = _make_app(svc=svc)
        fake_container = MagicMock()
        # No personal client → the scoped resolver hands back the platform fallback.
        fake_conn_svc = MagicMock(
            get_scoped_client=AsyncMock(side_effect=lambda _user, _conn, fallback: fallback)
        )
        fake_container.connection_service.return_value = fake_conn_svc
        fake_db_client = MagicMock(execute_sql=AsyncMock(return_value=[{"sum": 100}]))
        fake_container.db_client.return_value = fake_db_client
        with patch(
            "nl_to_sql.api.dependencies._get_container", return_value=fake_container
        ):
            async with _client(app) as c:
                r = await c.post("/api/v1/metrics/m1/preview?execute=true", headers=_auth())
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["row_count"] == 1

    async def test_execute_preview_blocks_dangerous_sql(self) -> None:
        """C6: even if a malicious SQL definition was persisted, preview must
        re-validate and refuse to execute it."""
        svc = _svc(get=_info(sql_definition="SELECT pg_read_file('/etc/passwd')"))
        app = _make_app(svc=svc)
        fake_container = MagicMock()
        # No personal client → the scoped resolver hands back the platform fallback.
        fake_conn_svc = MagicMock(
            get_scoped_client=AsyncMock(side_effect=lambda _user, _conn, fallback: fallback)
        )
        fake_container.connection_service.return_value = fake_conn_svc
        fake_db_client = MagicMock(execute_sql=AsyncMock(return_value=[{"sum": 100}]))
        fake_container.db_client.return_value = fake_db_client
        with patch(
            "nl_to_sql.api.dependencies._get_container", return_value=fake_container
        ):
            async with _client(app) as c:
                r = await c.post("/api/v1/metrics/m1/preview?execute=true", headers=_auth())
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        fake_db_client.execute_sql.assert_not_awaited()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
