"""Integration tests for Dashboards routes (Auto Charting & Dashboards):
  - GET    /api/v1/dashboards                     (list)
  - POST   /api/v1/dashboards                     (create)
  - GET    /api/v1/dashboards/{id}                (get)
  - PATCH  /api/v1/dashboards/{id}                (rename)
  - DELETE /api/v1/dashboards/{id}                (delete)
  - POST   /api/v1/dashboards/{id}/duplicate      (duplicate)
  - POST   /api/v1/dashboards/recommend-chart     (recommend)

The DashboardService is replaced via dependency_overrides with an AsyncMock, so
these tests exercise routing / auth / serialization, not the DB layer (which is
unit-tested separately).
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import Dashboard, DashboardWidget

USER_ID = "test-user-dash-001"
DT = datetime(2024, 1, 1, 12, 0, 0)

FAKE_USER = UserPublic(
    id=USER_ID,
    email="dashtest@example.com",
    full_name="Dash Test User",
    auth_provider="email",
    is_verified=True,
    created_at=DT,
)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _make_widget(widget_id: str = "w1", dashboard_id: str = "dash-1") -> DashboardWidget:
    w = DashboardWidget()
    w.id = widget_id
    w.dashboard_id = dashboard_id
    w.title = "Revenue by month"
    w.nl_prompt = "revenue by month"
    w.sql = "SELECT month, revenue FROM sales"
    w.chart_type = "line"
    w.chart_config = {"type": "line", "x_axis": "month", "y_axis": "revenue"}
    w.layout = {"x": 0, "y": 0, "w": 6, "h": 4}
    w.position = 0
    w.created_at = DT
    return w


def _make_dashboard(dashboard_id: str = "dash-1", widgets: list | None = None) -> Dashboard:
    d = Dashboard()
    d.id = dashboard_id
    d.user_id = USER_ID
    d.name = "Sales Dashboard"
    d.is_builtin = False
    d.created_at = DT
    d.updated_at = DT
    d.widgets = widgets if widgets is not None else [_make_widget()]
    return d


@pytest.fixture
def mock_service() -> MagicMock:
    from nl_to_sql.services.dashboard_service import DashboardService

    svc = MagicMock()
    # recommend_chart is a real deterministic staticmethod — keep it real.
    svc.recommend_chart = DashboardService.recommend_chart
    return svc


def _make_app(mock_service: MagicMock | None):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_current_user,
            get_dashboard_service,
            get_starter_content_service,
        )

        app = create_app()
        if mock_service is not None:
            async def _fake_user() -> UserPublic:
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user
            app.dependency_overrides[get_dashboard_service] = lambda: mock_service
            app.dependency_overrides[get_starter_content_service] = lambda: MagicMock(
                ensure_dashboards_seeded=AsyncMock(return_value=None)
            )
        return app


@pytest.fixture
def dash_app(mock_service):
    return _make_app(mock_service)


@pytest.fixture
async def client(dash_app):
    async with AsyncClient(transport=ASGITransport(app=dash_app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(mock_service=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── List ─────────────────────────────────────────────────────────────────────


class TestListDashboards:
    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/dashboards")
        assert r.status_code == 401

    async def test_returns_items(self, client, mock_service) -> None:
        mock_service.list_dashboards = AsyncMock(return_value=([_make_dashboard()], 1))
        r = await client.get("/api/v1/dashboards", headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == "dash-1"
        assert data["items"][0]["widget_count"] == 1

    async def test_invalid_limit_returns_422(self, client, mock_service) -> None:
        mock_service.list_dashboards = AsyncMock(return_value=([], 0))
        r = await client.get("/api/v1/dashboards", params={"limit": 0}, headers=_auth())
        assert r.status_code == 422


# ── Create ───────────────────────────────────────────────────────────────────


class TestCreateDashboard:
    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post("/api/v1/dashboards", json={"name": "X"})
        assert r.status_code == 401

    async def test_create_returns_201(self, client, mock_service) -> None:
        mock_service.create = AsyncMock(return_value=_make_dashboard())
        r = await client.post(
            "/api/v1/dashboards", json={"name": "Sales Dashboard"}, headers=_auth()
        )
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "Sales Dashboard"
        assert len(data["widgets"]) == 1
        assert data["widgets"][0]["chart_type"] == "line"

    async def test_missing_name_uses_default(self, client, mock_service) -> None:
        mock_service.create = AsyncMock(return_value=_make_dashboard())
        r = await client.post("/api/v1/dashboards", json={}, headers=_auth())
        assert r.status_code == 201


# ── Get ──────────────────────────────────────────────────────────────────────


class TestGetDashboard:
    async def test_found(self, client, mock_service) -> None:
        mock_service.get = AsyncMock(return_value=_make_dashboard())
        r = await client.get("/api/v1/dashboards/dash-1", headers=_auth())
        assert r.status_code == 200
        assert r.json()["id"] == "dash-1"

    async def test_cross_user_returns_404(self, client, mock_service) -> None:
        mock_service.get = AsyncMock(return_value=None)
        r = await client.get("/api/v1/dashboards/other", headers=_auth())
        assert r.status_code == 404
        assert "message" in r.json()


# ── Rename ───────────────────────────────────────────────────────────────────


class TestRenameDashboard:
    async def test_rename(self, client, mock_service) -> None:
        renamed = _make_dashboard()
        renamed.name = "Renamed"
        mock_service.rename = AsyncMock(return_value=renamed)
        r = await client.patch(
            "/api/v1/dashboards/dash-1", json={"name": "Renamed"}, headers=_auth()
        )
        assert r.status_code == 200
        assert r.json()["name"] == "Renamed"

    async def test_not_found(self, client, mock_service) -> None:
        mock_service.rename = AsyncMock(return_value=None)
        r = await client.patch("/api/v1/dashboards/x", json={"name": "Y"}, headers=_auth())
        assert r.status_code == 404


# ── Delete ───────────────────────────────────────────────────────────────────


class TestDeleteDashboard:
    async def test_delete_204(self, client, mock_service) -> None:
        mock_service.delete = AsyncMock(return_value=True)
        r = await client.delete("/api/v1/dashboards/dash-1", headers=_auth())
        assert r.status_code == 204

    async def test_delete_not_found(self, client, mock_service) -> None:
        mock_service.delete = AsyncMock(return_value=False)
        r = await client.delete("/api/v1/dashboards/x", headers=_auth())
        assert r.status_code == 404


# ── Duplicate ────────────────────────────────────────────────────────────────


class TestDuplicateDashboard:
    async def test_duplicate_201(self, client, mock_service) -> None:
        clone = _make_dashboard(dashboard_id="dash-2")
        clone.name = "Sales Dashboard (copy)"
        mock_service.duplicate = AsyncMock(return_value=clone)
        r = await client.post("/api/v1/dashboards/dash-1/duplicate", headers=_auth())
        assert r.status_code == 201
        assert r.json()["name"] == "Sales Dashboard (copy)"

    async def test_duplicate_not_found(self, client, mock_service) -> None:
        mock_service.duplicate = AsyncMock(return_value=None)
        r = await client.post("/api/v1/dashboards/x/duplicate", headers=_auth())
        assert r.status_code == 404


# ── Recommend chart ──────────────────────────────────────────────────────────


class TestRecommendChart:
    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post(
            "/api/v1/dashboards/recommend-chart", json={"columns": [], "rows": []}
        )
        assert r.status_code == 401

    async def test_line_recommendation(self, client) -> None:
        r = await client.post(
            "/api/v1/dashboards/recommend-chart",
            json={
                "columns": [{"name": "day", "type": "date"}, {"name": "revenue", "type": "float"}],
                "rows": [{"day": "2024-01-01", "revenue": 10}],
            },
            headers=_auth(),
        )
        assert r.status_code == 200
        data = r.json()
        assert data["chart_type"] == "line"
        assert data["x_axis"] == "day"
        assert data["y_axis"] == "revenue"

    async def test_table_fallback(self, client) -> None:
        r = await client.post(
            "/api/v1/dashboards/recommend-chart",
            json={
                "columns": [{"name": "label", "type": "text"}, {"name": "note", "type": "text"}],
                "rows": [{"label": "x", "note": "y"}],
            },
            headers=_auth(),
        )
        assert r.status_code == 200
        assert r.json()["chart_type"] == "table"


# ── Refresh ──────────────────────────────────────────────────────────────────


class TestRefreshDashboard:
    async def test_refresh_runs_widgets_and_isolates_errors(
        self, client, dash_app, mock_service
    ) -> None:
        good = _make_widget("w-good")
        bad = _make_widget("w-bad")
        bad.sql = "SELECT * FROM missing"
        mock_service.get = AsyncMock(return_value=_make_dashboard(widgets=[good, bad]))

        from nl_to_sql.api.dependencies import get_request_db_client

        async def _exec(sql: str):
            if "missing" in sql:
                raise RuntimeError("relation does not exist")
            return [{"month": "Jan", "revenue": 100}]

        mock_client = MagicMock()
        mock_client.execute_sql = AsyncMock(side_effect=_exec)

        # Override the per-user db client dependency on the live app instance.
        dash_app.dependency_overrides[get_request_db_client] = lambda: mock_client

        r = await client.post("/api/v1/dashboards/dash-1/refresh", headers=_auth())
        assert r.status_code == 200
        widgets = {w["widget_id"]: w for w in r.json()["widgets"]}
        assert widgets["w-good"]["error"] is None
        assert widgets["w-good"]["row_count"] == 1
        assert widgets["w-bad"]["error"] is not None
        assert widgets["w-bad"]["rows"] == []

    async def test_refresh_not_found(self, client, mock_service) -> None:
        mock_service.get = AsyncMock(return_value=None)
        r = await client.post("/api/v1/dashboards/x/refresh", headers=_auth())
        assert r.status_code == 404

    async def test_refresh_blocks_dangerous_widget_sql(
        self, client, dash_app, mock_service
    ) -> None:
        """C5: a widget stored with dangerous SQL must not execute on refresh."""
        evil = _make_widget("w-evil")
        evil.sql = "SELECT pg_sleep(300)"
        mock_service.get = AsyncMock(return_value=_make_dashboard(widgets=[evil]))

        from nl_to_sql.api.dependencies import get_request_db_client

        mock_client = MagicMock()
        mock_client.execute_sql = AsyncMock(return_value=[{"pg_sleep": None}])
        dash_app.dependency_overrides[get_request_db_client] = lambda: mock_client

        r = await client.post("/api/v1/dashboards/dash-1/refresh", headers=_auth())
        assert r.status_code == 200
        widget = r.json()["widgets"][0]
        assert widget["error"] is not None
        assert widget["rows"] == []
        mock_client.execute_sql.assert_not_awaited()
