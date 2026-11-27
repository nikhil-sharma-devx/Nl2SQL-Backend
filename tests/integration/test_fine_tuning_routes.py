"""Integration tests for fine-tuning routes — GET /api/v1/fine-tuning/jobs, etc."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.models.auth import UserPublic

USER_ID = "test-user-ft-001"
FAKE_USER = UserPublic(
    id=USER_ID,
    email="fttest@example.com",
    full_name="FineTune Tester",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2024, 1, 1),
    # Fine-tuning is an admin area: the training export is not scoped per user.
    is_admin=True,
)
NON_ADMIN = FAKE_USER.model_copy(update={"id": "test-user-ft-002", "is_admin": False})


def _auth() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _make_app(together_key: str = "", authed: bool = True, user: UserPublic = FAKE_USER):
    """Build the app with optionally mocked API key and auth."""
    with (
        patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"),
    ):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import get_current_user

        app = create_app()

        if authed:
            async def _fake_user():
                return user
            app.dependency_overrides[get_current_user] = _fake_user

        # Override the settings to control whether a key is "configured"
        orig_config = app.container.config

        def _patched_config():
            cfg = orig_config()
            cfg.together_api_key = together_key
            cfg.fine_tuning_provider = "together"
            return cfg

        try:
            app.container.config.override(_patched_config())
        except Exception:
            pass

        return app


@pytest.fixture
async def no_key_client():
    """Authenticated client with no Together AI key configured."""
    app = _make_app(together_key="", authed=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def unauthed_client():
    app = _make_app(together_key="", authed=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def keyed_client():
    """Authenticated, non-admin client with a Together AI key configured
    (bypasses the 400 "no key" short-circuit so admin-gating is reachable)."""
    app = _make_app(together_key="fake-together-key", authed=True, user=NON_ADMIN)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def mock_ft_svc():
    svc = MagicMock()
    svc.initialize = AsyncMock()
    svc.list_jobs = AsyncMock(return_value=[
        {"id": "ft-job-1", "status": "succeeded", "model": "llama-3"},
        {"id": "ft-job-2", "status": "running", "model": "llama-3"},
    ])
    svc.check_job_status = AsyncMock(return_value={"id": "ft-job-1", "status": "succeeded"})
    svc.start_fine_tuning = AsyncMock(return_value="ft-job-3")
    svc.deploy_fine_tuned_model = AsyncMock(return_value=True)
    svc.prepare_training_file = AsyncMock(return_value="/tmp/training.jsonl")
    return svc


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestListJobs:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/fine-tuning/jobs")
        assert r.status_code == 401

    async def test_returns_empty_list_when_no_api_key(self, no_key_client) -> None:
        """Key regression: used to return 400, now returns [] gracefully."""
        r = await no_key_client.get("/api/v1/fine-tuning/jobs", headers=_auth())
        assert r.status_code == 200
        assert r.json() == []

    async def test_accepts_limit_param(self, no_key_client) -> None:
        r = await no_key_client.get("/api/v1/fine-tuning/jobs?limit=20", headers=_auth())
        assert r.status_code == 200

    async def test_invalid_limit_returns_422(self, no_key_client) -> None:
        r = await no_key_client.get("/api/v1/fine-tuning/jobs?limit=0", headers=_auth())
        assert r.status_code == 422

    async def test_limit_too_large_returns_422(self, no_key_client) -> None:
        r = await no_key_client.get("/api/v1/fine-tuning/jobs?limit=999", headers=_auth())
        assert r.status_code == 422


class TestJobStatus:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.get("/api/v1/fine-tuning/status/ft-job-1")
        assert r.status_code == 401

    async def test_no_key_returns_400(self, no_key_client) -> None:
        # Status check (non-list route) still correctly requires an API key
        r = await no_key_client.get("/api/v1/fine-tuning/status/ft-job-1", headers=_auth())
        assert r.status_code == 400

    async def test_400_message_mentions_api_key(self, no_key_client) -> None:
        r = await no_key_client.get("/api/v1/fine-tuning/status/ft-job-1", headers=_auth())
        body = r.json()
        detail = body.get("message", body.get("detail", "")).lower()
        assert "key" in detail or "api" in detail or "together" in detail


class TestStartFineTuning:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post(
            "/api/v1/fine-tuning/start",
            params={"model": "llama-3", "training_file_path": "/tmp/data.jsonl"},
        )
        assert r.status_code == 401

    async def test_no_key_returns_400(self, no_key_client) -> None:
        r = await no_key_client.post(
            "/api/v1/fine-tuning/start",
            params={"model": "llama-3", "training_file_path": "/tmp/data.jsonl"},
            headers=_auth(),
        )
        assert r.status_code == 400

    async def test_non_admin_forbidden(self, keyed_client) -> None:
        """Starting a fine-tuning job consumes platform-wide budget — admin-only."""
        r = await keyed_client.post(
            "/api/v1/fine-tuning/start",
            params={"model": "llama-3", "training_file_path": "/tmp/data.jsonl"},
            headers=_auth(),
        )
        assert r.status_code == 403


class TestPrepareFinetuning:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post("/api/v1/fine-tuning/prepare")
        assert r.status_code == 401

    async def test_no_key_returns_400(self, no_key_client) -> None:
        r = await no_key_client.post(
            "/api/v1/fine-tuning/prepare",
            headers=_auth(),
        )
        assert r.status_code == 400


class TestDeployModel:

    async def test_requires_auth(self, unauthed_client) -> None:
        r = await unauthed_client.post(
            "/api/v1/fine-tuning/deploy",
            params={"model_id": "ft-abc123"},
        )
        assert r.status_code == 401

    async def test_no_key_returns_400(self, no_key_client) -> None:
        r = await no_key_client.post(
            "/api/v1/fine-tuning/deploy",
            params={"model_id": "ft-abc123"},
            headers=_auth(),
        )
        assert r.status_code == 400

    async def test_non_admin_forbidden(self, keyed_client) -> None:
        """C1: deploying a fine-tuned model hot-swaps the LLM for every user — admin-only."""
        r = await keyed_client.post(
            "/api/v1/fine-tuning/deploy",
            params={"model_id": "ft-abc123"},
            headers=_auth(),
        )
        assert r.status_code == 403


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/v1/fine-tuning/prepare"),
        ("get", "/api/v1/fine-tuning/status/ft-job-1"),
        ("get", "/api/v1/fine-tuning/jobs"),
    ],
)
async def test_non_admin_is_refused(method: str, path: str) -> None:
    """The training export holds every user's questions + SQL — admins only."""
    from httpx import ASGITransport, AsyncClient

    app = _make_app(user=NON_ADMIN)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await getattr(client, method)(path, headers=_auth())
    assert r.status_code == 403
