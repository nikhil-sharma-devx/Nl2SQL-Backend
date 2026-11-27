"""Unit tests for auth_service — password hashing, JWT, OTP."""
from datetime import UTC, timedelta
from unittest.mock import patch

import pytest
from jose import JWTError, jwt

from nl_to_sql.services.auth_service import (
    create_access_token,
    decode_access_token,
    generate_otp,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_token_expiry,
    verify_password,
)

# ── Refresh Token Utilities ─────────────────────────────────────────────────────


class TestRefreshTokenUtils:

    def test_generate_returns_nonempty_unique_tokens(self) -> None:
        t1 = generate_refresh_token()
        t2 = generate_refresh_token()
        assert t1 and t2
        assert t1 != t2  # high-entropy random

    def test_hash_is_deterministic_sha256_hex(self) -> None:
        raw = generate_refresh_token()
        h1 = hash_refresh_token(raw)
        h2 = hash_refresh_token(raw)
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex digest
        assert all(c in "0123456789abcdef" for c in h1)

    def test_hash_differs_for_different_tokens(self) -> None:
        assert hash_refresh_token("a") != hash_refresh_token("b")

    def test_expiry_is_in_the_future(self) -> None:
        from datetime import datetime

        expiry = refresh_token_expiry()
        # Stored as naive UTC, matching the other auth models.
        assert expiry.tzinfo is None
        assert expiry > datetime.utcnow()


# ── Password Utilities ────────────────────────────────────────────────────────


class TestPasswordUtils:

    def test_hash_and_verify_roundtrip(self) -> None:
        pw = "correcthorsebatterystaple"
        hashed = hash_password(pw)
        assert verify_password(pw, hashed) is True

    def test_wrong_password_fails(self) -> None:
        hashed = hash_password("rightpassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_hash_is_bcrypt_prefixed(self) -> None:
        hashed = hash_password("somepassword")
        assert hashed.startswith("$2b$")

    def test_same_password_different_hashes(self) -> None:
        pw = "samepassword"
        h1 = hash_password(pw)
        h2 = hash_password(pw)
        assert h1 != h2  # bcrypt generates random salts

    def test_empty_password_hashes_and_verifies(self) -> None:
        hashed = hash_password("")
        assert verify_password("", hashed) is True

    def test_password_over_72_bytes_treated_as_72(self) -> None:
        long_pw = "a" * 73
        truncated_pw = "a" * 72
        hashed = hash_password(long_pw)
        # bcrypt silently truncates at 72 bytes — both should verify
        assert verify_password(long_pw, hashed) is True
        assert verify_password(truncated_pw, hashed) is True

    def test_verify_returns_false_on_corrupt_hash(self) -> None:
        assert verify_password("anything", "notahash") is False


# ── JWT Utilities ─────────────────────────────────────────────────────────────


class TestJWT:

    def test_create_and_decode_roundtrip(self) -> None:
        token = create_access_token(user_id="uid-1", email="user@example.com")
        data = decode_access_token(token)
        assert data.user_id == "uid-1"
        assert data.email == "user@example.com"

    def test_session_id_embedded_and_returned(self) -> None:
        token = create_access_token(
            user_id="uid-2", email="user@example.com", session_id="sess-abc"
        )
        data = decode_access_token(token)
        assert data.session_id == "sess-abc"

    def test_no_session_id_returns_none(self) -> None:
        token = create_access_token(user_id="uid-3", email="user@example.com")
        data = decode_access_token(token)
        assert data.session_id is None

    def test_tampered_token_raises(self) -> None:
        token = create_access_token(user_id="uid-4", email="user@example.com")
        tampered = token[:-4] + "XXXX"
        with pytest.raises(JWTError):
            decode_access_token(tampered)

    def test_completely_invalid_token_raises(self) -> None:
        with pytest.raises(JWTError):
            decode_access_token("this.is.notavalidjwt")

    def test_expired_token_raises(self) -> None:
        from nl_to_sql.config.settings import get_settings
        settings = get_settings()
        from datetime import datetime
        payload = {
            "sub": "uid-5",
            "email": "user@example.com",
            "exp": datetime.now(UTC) - timedelta(seconds=1),
            "iat": datetime.now(UTC) - timedelta(minutes=60),
        }
        expired_token = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        with pytest.raises(JWTError):
            decode_access_token(expired_token)

    def test_missing_sub_raises(self) -> None:
        from nl_to_sql.config.settings import get_settings
        settings = get_settings()
        from datetime import datetime
        payload = {
            "email": "user@example.com",
            "exp": datetime.now(UTC) + timedelta(minutes=30),
        }
        token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        with pytest.raises(JWTError):
            decode_access_token(token)

    def test_missing_email_raises(self) -> None:
        from nl_to_sql.config.settings import get_settings
        settings = get_settings()
        from datetime import datetime
        payload = {
            "sub": "uid-6",
            "exp": datetime.now(UTC) + timedelta(minutes=30),
        }
        token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        with pytest.raises(JWTError):
            decode_access_token(token)


# ── OTP ───────────────────────────────────────────────────────────────────────


class TestGenerateOTP:

    def test_is_six_characters(self) -> None:
        otp = generate_otp()
        assert len(otp) == 6

    def test_only_digits(self) -> None:
        otp = generate_otp()
        assert otp.isdigit()

    def test_generates_different_values(self) -> None:
        otps = {generate_otp() for _ in range(20)}
        assert len(otps) > 1  # statistically impossible to get 20 identical OTPs


# ── Google Token Verification (mocked) ───────────────────────────────────────


class TestVerifyGoogleToken:

    async def test_returns_claims_on_valid_token(self) -> None:
        from nl_to_sql.services.auth_service import verify_google_token

        fake_idinfo = {
            "sub": "google-sub-123",
            "email": "user@gmail.com",
            "email_verified": True,
            "name": "Test User",
            "picture": "https://example.com/photo.jpg",
        }

        with (
            patch("google.oauth2.id_token.verify_oauth2_token",
                  return_value=fake_idinfo),
            patch("nl_to_sql.services.auth_service.get_settings") as mock_settings,
        ):
            mock_settings.return_value.google_client_id = "fake-client-id"
            result = await verify_google_token("fake-credential")

        assert result["sub"] == "google-sub-123"
        assert result["email"] == "user@gmail.com"
        assert result["email_verified"] is True

    async def test_raises_when_no_client_id_configured(self) -> None:
        from nl_to_sql.core.exceptions import AuthenticationError
        from nl_to_sql.services.auth_service import verify_google_token

        with patch("nl_to_sql.services.auth_service.get_settings") as mock_settings:
            mock_settings.return_value.google_client_id = ""
            with pytest.raises(AuthenticationError, match="GOOGLE_CLIENT_ID"):
                await verify_google_token("any-credential")

    async def test_raises_on_invalid_token(self) -> None:
        from nl_to_sql.core.exceptions import AuthenticationError
        from nl_to_sql.services.auth_service import verify_google_token

        with (
            patch("google.oauth2.id_token.verify_oauth2_token",
                  side_effect=ValueError("Token invalid")),
            patch("nl_to_sql.services.auth_service.get_settings") as mock_settings,
        ):
            mock_settings.return_value.google_client_id = "fake-client-id"
            with pytest.raises(AuthenticationError, match="Invalid Google token"):
                await verify_google_token("bad-credential")
