"""Integration tests for Schedule routes (api/routes/schedules.py):

  GET/POST        /api/v1/schedules
  GET/PUT/DELETE  /api/v1/schedules/{id}
  POST            /api/v1/schedules/{id}/pause
  POST            /api/v1/schedules/{id}/resume
  POST            /api/v1/schedules/{id}/run-now
  GET             /api/v1/schedules/{id}/history

``get_current_user`` and ``get_scheduled_query_service`` are real FastAPI
dependencies, so they are overridden via ``app.dependency_overrides``. The
service is mocked; domain exceptions raised by the service are mapped to the
canonical error envelope (``"message"``/``"code"``) by the global handler.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.exceptions import ScheduleNotFoundError, ScheduleValidationError
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import ScheduledQuery, ScheduledQueryRun
from nl_to_sql.services.scheduled_query_service import ScheduleInfo

USER_ID = "sched-routes-user-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="schedtest@example.com",
    full_name="Sched Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
)


def _info(schedule_id="s1", **overrides) -> ScheduleInfo:
    defaults = {
        "id": schedule_id,
        "connection_id": "conn-1",
        "name": "Daily revenue",
        "nl_prompt": "total revenue by day",
        "cron_expr": "0 8 * * *",
        "raw_schedule_text": "every morning",
        "timezone": "UTC",
        "is_paused": False,
        "notify_email": True,
        "notify_in_app": False,
        "notify_condition": "always",
        "is_builtin": False,
        "next_run_at": datetime(2026, 1, 1, 8, 0, 0),
        "last_run_at": None,
        "last_status": None,
        "consecutive_failures": 0,
        "created_at": datetime(2024, 1, 1),
        "updated_at": datetime(2024, 1, 1),
    }
    defaults.update(overrides)
    return ScheduleInfo(**defaults)


def _make_app(*, authed: bool = True, svc: MagicMock | None = None):
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_active_connection_id,
            get_current_user,
            get_scheduled_query_service,
            get_starter_content_service,
        )

        app = create_app()
        if authed:
            async def _fake_user():
                return FAKE_USER

            app.dependency_overrides[get_current_user] = _fake_user

        async def _fake_connection_id():
            return "conn-1"

        app.dependency_overrides[get_active_connection_id] = _fake_connection_id
        app.dependency_overrides[get_starter_content_service] = lambda: MagicMock(
            ensure_schedules_seeded=AsyncMock(return_value=None)
        )
        if svc is not None:
            app.dependency_overrides[get_scheduled_query_service] = lambda: svc
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


class TestListSchedules:
    async def test_requires_auth(self) -> None:
        app = _make_app(authed=False, svc=_svc(list_schedules=([], 0)))
        async with _client(app) as c:
            r = await c.get("/api/v1/schedules")
        assert r.status_code == 401

    async def test_lists_schedules(self) -> None:
        svc = _svc(list_schedules=([_info()], 1))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.get("/api/v1/schedules", headers=_auth())
        assert r.status_code == 200
        body = r.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["name"] == "Daily revenue"
        assert body["total"] == 1


class TestCreateSchedule:
    async def test_happy_path(self) -> None:
        svc = _svc(create=_info())
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post(
                "/api/v1/schedules",
                json={
                    "connection_id": "conn-1",
                    "name": "Daily revenue",
                    "nl_prompt": "total revenue by day",
                    "schedule_text": "every morning",
                },
                headers=_auth(),
            )
        assert r.status_code == 200
        svc.create.assert_awaited_once()
        assert r.json()["name"] == "Daily revenue"

    async def test_unknown_connection_maps_to_400(self) -> None:
        svc = _svc()
        svc.create = AsyncMock(side_effect=ScheduleValidationError("Unknown connection."))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post(
                "/api/v1/schedules",
                json={
                    "connection_id": "conn-x",
                    "name": "n",
                    "nl_prompt": "q",
                    "schedule_text": "every morning",
                },
                headers=_auth(),
            )
        assert r.status_code == 400
        assert r.json()["code"] == "SCHEDULE_INVALID"

    async def test_bad_body_returns_422(self) -> None:
        app = _make_app(svc=_svc())
        async with _client(app) as c:
            r = await c.post("/api/v1/schedules", json={"name": ""}, headers=_auth())
        assert r.status_code == 422


class TestGetSchedule:
    async def test_cross_user_is_404(self) -> None:
        svc = _svc()
        svc.get = AsyncMock(side_effect=ScheduleNotFoundError("Schedule not found."))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.get("/api/v1/schedules/other", headers=_auth())
        assert r.status_code == 404
        assert r.json()["code"] == "SCHEDULE_NOT_FOUND"


class TestPauseResume:
    async def test_pause(self) -> None:
        svc = _svc(pause=_info(is_paused=True))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/schedules/s1/pause", headers=_auth())
        assert r.status_code == 200
        assert r.json()["is_paused"] is True

    async def test_resume(self) -> None:
        svc = _svc(resume=_info(is_paused=False))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/schedules/s1/resume", headers=_auth())
        assert r.status_code == 200
        assert r.json()["is_paused"] is False


class TestDeleteSchedule:
    async def test_delete(self) -> None:
        svc = _svc(delete=None)
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.delete("/api/v1/schedules/s1", headers=_auth())
        assert r.status_code == 200
        svc.delete.assert_awaited_once_with(USER_ID, "s1")


class TestHistory:
    async def test_lists_history(self) -> None:
        run = ScheduledQueryRun(
            id="run-1",
            schedule_id="s1",
            started_at=datetime(2026, 1, 1, 8, 0, 0),
            finished_at=datetime(2026, 1, 1, 8, 0, 1),
            status="success",
            row_count=3,
            generated_sql="SELECT 1",
            error=None,
            notified=True,
            duration_ms=1000,
        )
        svc = _svc(list_history=([run], 1))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.get("/api/v1/schedules/s1/history", headers=_auth())
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["items"][0]["status"] == "success"


class TestRunNow:
    async def test_run_now_executes_and_returns_latest_run(self) -> None:
        row = ScheduledQuery(
            id="s1", user_id=USER_ID, connection_id="conn-1", name="n",
            nl_prompt="q", cron_expr="0 8 * * *", timezone="UTC",
        )
        run = ScheduledQueryRun(
            id="run-1", schedule_id="s1", started_at=datetime(2026, 1, 1),
            finished_at=datetime(2026, 1, 1), status="success", row_count=1,
            generated_sql="SELECT 1", error=None, notified=False, duration_ms=10,
        )
        svc = _svc(
            get_owned_row=row,
            list_history=([run], 1),
        )
        app = _make_app(svc=svc)
        with patch(
            "nl_to_sql.workers.scheduled_query_worker.execute_schedule", new=AsyncMock()
        ) as exec_mock:
            async with _client(app) as c:
                r = await c.post("/api/v1/schedules/s1/run-now", headers=_auth())
        assert r.status_code == 200
        exec_mock.assert_awaited_once()
        assert r.json()["status"] == "success"

    async def test_run_now_cross_user_is_404(self) -> None:
        svc = _svc()
        svc.get_owned_row = AsyncMock(side_effect=ScheduleNotFoundError("Schedule not found."))
        app = _make_app(svc=svc)
        async with _client(app) as c:
            r = await c.post("/api/v1/schedules/other/run-now", headers=_auth())
        assert r.status_code == 404


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
