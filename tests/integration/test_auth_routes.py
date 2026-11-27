"""Integration tests for auth routes — /auth/register, /auth/login, /auth/verify-otp, etc."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_app():
    """App with mocked database session so auth routes don't need a real DB."""

    mock_user = MagicMock()
    mock_user.id = "user-123"
    mock_user.email = "test@example.com"
    mock_user.full_name = "Test User"
    mock_user.hashed_password = "$2b$12$fakehash"
    mock_user.auth_provider = "email"
    mock_user.is_active = True
    mock_user.is_verified = False
    mock_user.otp_code = "123456"
    mock_user.otp_expires_at = None
    mock_user.google_sub = None
    mock_user.created_at = __import__("datetime").datetime(2024, 1, 1)
    mock_user.updated_at = __import__("datetime").datetime(2024, 1, 1)

    # Mock session factory and db session
    mock_db_sess = AsyncMock()
    mock_db_sess.__aenter__ = AsyncMock(return_value=mock_db_sess)
    mock_db_sess.__aexit__ = AsyncMock(return_value=False)
    mock_db_sess.execute = AsyncMock()
    mock_db_sess.add = MagicMock()
    mock_db_sess.flush = AsyncMock()
    mock_db_sess.commit = AsyncMock()
    mock_db_sess.refresh = AsyncMock()

    mock_session_factory = MagicMock(return_value=mock_db_sess)

    with (
        patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"),
        patch(
            "nl_to_sql.services.chat_session_service.ChatSessionService._session_factory",
            new_callable=lambda: property(lambda self: mock_session_factory),
        ),
    ):
        from nl_to_sql.api.app import create_app
        app = create_app()
        app.state._mock_user = mock_user
        app.state._mock_db_sess = mock_db_sess
        yield app


@pytest.fixture
async def auth_client(patched_app):
    from httpx import ASGITransport, AsyncClient
    async with AsyncClient(
        transport=ASGITransport(app=patched_app), base_url="http://test"
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


class TestRegister:

    async def test_register_rejects_short_password(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"email": "user@example.com", "password": "short"},
        )
        assert response.status_code == 422

    async def test_register_rejects_invalid_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "validpass123"},
        )
        assert response.status_code == 422

    async def test_register_requires_email_and_password(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/auth/register", json={})
        assert response.status_code == 422

    async def test_register_missing_password(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/register",
            json={"email": "user@example.com"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:

    async def test_login_requires_email_and_password(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/auth/login", json={})
        assert response.status_code == 422

    async def test_login_rejects_invalid_email_format(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": "not-valid", "password": "password123"},
        )
        assert response.status_code == 422

    async def test_login_missing_password(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": "user@example.com"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# OTP verification
# ---------------------------------------------------------------------------


class TestVerifyOTP:

    async def test_verify_otp_requires_email_and_code(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/auth/verify-otp", json={})
        assert response.status_code == 422

    async def test_verify_otp_requires_valid_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/verify-otp",
            json={"email": "bad-email", "otp_code": "123456"},
        )
        assert response.status_code == 422

    async def test_verify_otp_missing_code(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/verify-otp",
            json={"email": "user@example.com"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Resend OTP
# ---------------------------------------------------------------------------


class TestResendOTP:

    async def test_resend_otp_requires_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/auth/resend-otp", json={})
        assert response.status_code == 422

    async def test_resend_otp_rejects_invalid_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/resend-otp",
            json={"email": "not-an-email"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Forgot / Reset Password
# ---------------------------------------------------------------------------


class TestForgotPassword:

    async def test_forgot_password_requires_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/auth/forgot-password", json={})
        assert response.status_code == 422

    async def test_forgot_password_rejects_invalid_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "invalid"},
        )
        assert response.status_code == 422


class TestResetPassword:

    async def test_reset_password_requires_all_fields(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/auth/reset-password", json={})
        assert response.status_code == 422

    async def test_reset_password_rejects_short_password(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/reset-password",
            json={"email": "user@example.com", "otp_code": "123456", "new_password": "short"},
        )
        assert response.status_code == 422

    async def test_reset_password_rejects_invalid_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/reset-password",
            json={"email": "bad-email", "otp_code": "123456", "new_password": "validpass123"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Get current user (requires auth)
# ---------------------------------------------------------------------------


class TestGetMe:

    async def test_get_me_requires_auth(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/api/v1/auth/me")
        assert response.status_code == 401

    async def test_get_me_rejects_invalid_token(self, async_client: AsyncClient) -> None:
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401

    async def test_get_me_rejects_malformed_bearer(self, async_client: AsyncClient) -> None:
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "NotBearer sometoken"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Refresh / Logout
# ---------------------------------------------------------------------------


class TestRefresh:

    async def test_refresh_requires_token(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/auth/refresh", json={})
        assert response.status_code == 422


class TestLogout:

    async def test_logout_requires_auth(self, async_client: AsyncClient) -> None:
        # Logout revokes the current login session (identified by the JWT's sid),
        # so it requires a valid access token.
        response = await async_client.post("/api/v1/auth/logout")
        assert response.status_code == 401

    async def test_logout_rejects_invalid_token(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Google Auth
# ---------------------------------------------------------------------------


class TestGoogleAuth:

    async def test_google_auth_requires_credential(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/auth/google", json={})
        assert response.status_code == 422

    async def test_google_auth_rejects_empty_credential(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/v1/auth/google",
            json={"credential": ""},
        )
        # Empty string fails Pydantic min-length or Google token verification
        assert response.status_code in (401, 422)
