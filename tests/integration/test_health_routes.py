"""Integration tests for health routes — GET /api/v1/health, /ready, /health/deep."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_app():
    """Build the app with the LLM SDK patched out (matches history-route tests)."""
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app

        return create_app()


def _dep_mock(health_ok: bool = True, raises: Exception | None = None) -> MagicMock:
    """A dependency whose async .health_check returns `health_ok` or raises."""
    m = MagicMock()
    if raises is not None:
        m.health_check = AsyncMock(side_effect=raises)
    else:
        m.health_check = AsyncMock(return_value=health_ok)
    return m


def _make_container(db_ok=True, vs_ok=True, llm_ok=True,
                    db_raises=None, vs_raises=None, llm_raises=None) -> MagicMock:
    """A container MagicMock whose .db_client()/.vector_store()/.llm_provider()
    return objects with an async .health_check."""
    container = MagicMock()
    container.db_client.return_value = _dep_mock(db_ok, db_raises)
    container.vector_store.return_value = _dep_mock(vs_ok, vs_raises)
    container.llm_provider.return_value = _dep_mock(llm_ok, llm_raises)
    return container


@pytest.fixture
async def client():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── /health (liveness) ──────────────────────────────────────────────────────────


class TestLiveness:

    async def test_returns_200_and_shape(self, client) -> None:
        r = await client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "environment" in data
        assert isinstance(data["environment"], str)


# ── /ready (readiness) ──────────────────────────────────────────────────────────


class TestReadiness:

    async def test_happy_path(self) -> None:
        from nl_to_sql.api.dependencies import get_db_client, get_vector_store

        app = _make_app()
        app.dependency_overrides[get_db_client] = lambda: _dep_mock(True)
        app.dependency_overrides[get_vector_store] = lambda: _dep_mock(True)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/api/v1/ready")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        assert data["checks"] == {"database": True, "vector_store": True}

    async def test_degraded_when_db_unhealthy(self) -> None:
        from nl_to_sql.api.dependencies import get_db_client, get_vector_store

        app = _make_app()
        app.dependency_overrides[get_db_client] = lambda: _dep_mock(False)
        app.dependency_overrides[get_vector_store] = lambda: _dep_mock(True)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/api/v1/ready")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "degraded"
        assert data["checks"]["database"] is False
        assert data["checks"]["vector_store"] is True

    async def test_degraded_when_vector_store_unhealthy(self) -> None:
        from nl_to_sql.api.dependencies import get_db_client, get_vector_store

        app = _make_app()
        app.dependency_overrides[get_db_client] = lambda: _dep_mock(True)
        app.dependency_overrides[get_vector_store] = lambda: _dep_mock(False)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/api/v1/ready")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "degraded"
        assert data["checks"]["vector_store"] is False


# ── /health/deep ────────────────────────────────────────────────────────────────


class TestDeepHealth:

    async def test_happy_path(self) -> None:
        from nl_to_sql.api.dependencies import get_container

        app = _make_app()
        app.dependency_overrides[get_container] = lambda: _make_container()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/api/v1/health/deep")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert set(data["checks"]) == {"database", "vector_store", "llm_provider"}
        for res in data["checks"].values():
            assert res["ok"] is True
            assert res["error"] is None
            assert isinstance(res["latency_ms"], int)

    async def test_degraded_when_llm_unhealthy(self) -> None:
        from nl_to_sql.api.dependencies import get_container

        app = _make_app()
        app.dependency_overrides[get_container] = lambda: _make_container(llm_ok=False)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/api/v1/health/deep")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "degraded"
        assert data["checks"]["llm_provider"]["ok"] is False
        assert data["checks"]["database"]["ok"] is True

    async def test_error_when_check_raises(self) -> None:
        from nl_to_sql.api.dependencies import get_container

        app = _make_app()
        app.dependency_overrides[get_container] = lambda: _make_container(
            db_raises=RuntimeError("db exploded")
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/api/v1/health/deep")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "degraded"
        db_check = data["checks"]["database"]
        assert db_check["ok"] is False
        assert db_check["error"] is not None
        assert "db exploded" in db_check["error"]
        # unaffected checks stay healthy
        assert data["checks"]["vector_store"]["ok"] is True
