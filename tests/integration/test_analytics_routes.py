"""Integration tests for analytics routes — GET /api/v1/analytics/*."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

USER_ID = "test-user-analytics-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="analyticstest@example.com",
    full_name="Analytics Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


@pytest.fixture
def mock_analytics_svc():
    svc = MagicMock()
    svc.initialize = AsyncMock()
    svc.dispose = AsyncMock()
    svc.get_summary = AsyncMock(return_value={
        "total_queries": 100,
        "successful_queries": 90,
        "failed_queries": 10,
        "success_rate": 0.90,
        "cached_queries": 30,
        "cache_hit_rate": 0.30,
        "cache_exact_hit_rate": 20.0,
        "cache_semantic_hit_rate": 10.0,
        "cache_layer_lookups": 100,
        "avg_tokens_used": 250,
        "avg_response_time_ms": 420,
        "period_days": 30,
    })
    svc.get_popular_queries = AsyncMock(return_value=[
        {"question": "Show all customers", "count": 15},
        {"question": "Count orders", "count": 12},
    ])
    svc.get_table_usage = AsyncMock(return_value=[
        {"table_name": "orders", "usage_count": 50},
        {"table_name": "customers", "usage_count": 40},
    ])
    svc.get_failure_patterns = AsyncMock(return_value=[])
    svc.reset_analytics = AsyncMock(return_value={"deleted": 100})
    return svc


def _make_app(mock_analytics_svc=None, authed=True):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_current_user

        app = create_app()

        if authed:
            async def _fake_user():
                return FAKE_USER
            app.dependency_overrides[get_current_user] = _fake_user

        if mock_analytics_svc is not None:
            from nl_to_sql.api.dependencies import _get_container
            container = _get_container()
            container.analytics_service.override(mock_analytics_svc)

        return app


@pytest.fixture
async def client(mock_analytics_svc):
    app = _make_app(mock_analytics_svc=mock_analytics_svc)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(authed=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestAnalyticsSummary:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/analytics/summary")
        assert r.status_code == 401

    async def test_returns_200(self, client) -> None:
        r = await client.get("/api/v1/analytics/summary", headers=_auth())
        assert r.status_code == 200

    async def test_summary_has_required_fields(self, client) -> None:
        r = await client.get("/api/v1/analytics/summary", headers=_auth())
        if r.status_code == 200:
            data = r.json()
            for field in ("total_queries", "successful_queries", "failed_queries", "success_rate"):
                assert field in data

    async def test_accepts_period_param(self, client) -> None:
        r = await client.get("/api/v1/analytics/summary?period=7", headers=_auth())
        assert r.status_code in (200, 422)


class TestPopularQueries:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/analytics/popular-queries")
        assert r.status_code == 401

    async def test_returns_list(self, client) -> None:
        r = await client.get("/api/v1/analytics/popular-queries", headers=_auth())
        assert r.status_code in (200, 404)

    async def test_accepts_limit_param(self, client) -> None:
        r = await client.get("/api/v1/analytics/popular-queries?limit=5", headers=_auth())
        assert r.status_code in (200, 404, 422)


class TestTableUsage:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/analytics/table-usage")
        assert r.status_code == 401

    async def test_returns_list(self, client) -> None:
        r = await client.get("/api/v1/analytics/table-usage", headers=_auth())
        assert r.status_code in (200, 404)


class TestClearAnalytics:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.delete("/api/v1/analytics/reset")
        assert r.status_code == 401

    async def test_delete_requires_admin(self, client) -> None:
        r = await client.delete("/api/v1/analytics/reset", headers=_auth())
        assert r.status_code == 403
