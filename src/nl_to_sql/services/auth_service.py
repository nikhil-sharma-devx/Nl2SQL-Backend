"""Authentication service — password hashing, JWT generation, Google token verification."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import structlog
from jose import JWTError, jwt

from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.exceptions import AuthenticationError
from nl_to_sql.core.models.auth import TokenData, UserPublic

logger = structlog.get_logger(__name__)


def to_user_public(user: Any) -> UserPublic:
    """Build a ``UserPublic`` from an ORM ``User`` row.

    The single place every route constructs a ``UserPublic``. ``is_admin`` and
    ``is_owner`` come straight from the ``users`` row (managed only by
    ``AdminService``) — the frontend needs them to hide admin-only controls up
    front rather than rendering them for every user and failing on submit.
    """
    return UserPublic.model_validate(user)


# ── Password Utilities ────────────────────────────────────────────────────────


def hash_password(plain_password: str) -> str:
    """Return a bcrypt hash of the given plain-text password."""
    # bcrypt has a 72-byte limit; truncate to be safe
    password_bytes = plain_password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if plain_password matches hashed_password."""
    try:
        password_bytes = plain_password.encode("utf-8")[:72]
        return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))
    except Exception:
        return False


# ── JWT Utilities ─────────────────────────────────────────────────────────────


def create_access_token(user_id: str, email: str, session_id: str | None = None) -> str:
    """Create a signed JWT for the given user.

    Args:
        user_id: The user's primary-key UUID string.
        email: The user's email address.
        session_id: Optional login session UUID to embed for server-side revocation checks.

    Returns:
        A signed JWT string.
    """
    settings = get_settings()
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.now(UTC),
    }
    if session_id:
        payload["sid"] = session_id
    return str(jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm))


def decode_access_token(token: str) -> TokenData:
    """Decode and validate a JWT.

    Args:
        token: The raw JWT string from the Authorization header.

    Returns:
        TokenData with user_id, email, and optional session_id.

    Raises:
        JWTError: If the token is invalid or expired.
    """
    settings = get_settings()
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    # Access tokens carry no ``typ``; any typed token (e.g. a reactivation
    # token) is for one narrow purpose and must never authenticate a request.
    if payload.get("typ") not in (None, "access"):
        raise JWTError("Not an access token")
    user_id: str = payload.get("sub", "")
    email: str = payload.get("email", "")
    if not user_id or not email:
        raise JWTError("Missing subject or email in token")
    return TokenData(user_id=user_id, email=email, session_id=payload.get("sid"))


# ── Reactivation Token (deactivated accounts) ──────────────────────────────────

_REACTIVATION_TYP = "reactivation"
_REACTIVATION_TTL = timedelta(minutes=30)


def create_reactivation_token(user_id: str) -> str:
    """Short-lived token proving the holder just signed in to a deactivated account.

    Issued only after the password / Google credential checked out, so it lets
    that person — and nobody guessing emails — ask an admin for reactivation.
    It has no ``email`` claim and a ``typ``, so ``decode_access_token`` rejects it.
    """
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {"sub": user_id, "typ": _REACTIVATION_TYP, "iat": now, "exp": now + _REACTIVATION_TTL}
    return str(jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm))


def decode_reactivation_token(token: str) -> str:
    """Return the user id in a valid reactivation token.

    Raises:
        AuthenticationError: if the token is invalid, expired or another type.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise AuthenticationError(
            "This request link has expired. Sign in again to send it."
        ) from exc
    user_id = payload.get("sub", "")
    if payload.get("typ") != _REACTIVATION_TYP or not user_id:
        raise AuthenticationError("This request link has expired. Sign in again to send it.")
    return str(user_id)


# ── Refresh Token Utilities ────────────────────────────────────────────────────


def generate_refresh_token() -> str:
    """Return a new opaque, URL-safe refresh token (returned to the client once)."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(raw_token: str) -> str:
    """Return the SHA-256 hex digest stored server-side for a raw refresh token.

    Refresh tokens are high-entropy random strings, so a plain (unsalted) SHA-256
    is sufficient and lets us look rows up by an indexed equality match.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def refresh_token_expiry() -> datetime:
    """Return the naive-UTC expiry for a newly issued refresh token."""
    settings = get_settings()
    return datetime.now(UTC).replace(tzinfo=None) + timedelta(
        days=settings.refresh_token_expire_days
    )


# ── Google Token Verification ─────────────────────────────────────────────────


async def verify_google_token(credential: str) -> dict[str, Any]:
    """Verify a Google ID token and return the decoded claims.

    Args:
        credential: The Google ID token string from the frontend.

    Returns:
        Decoded token claims dict with keys: sub, email, name, picture.

    Raises:
        AuthenticationError: If the token is invalid or the audience doesn't match.
    """
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token

    settings = get_settings()
    if not settings.google_client_id:
        raise AuthenticationError("GOOGLE_CLIENT_ID is not configured on the server")

    try:
        idinfo = google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            settings.google_client_id,
        )
        return {
            "sub": idinfo["sub"],
            "email": idinfo["email"],
            "email_verified": idinfo.get("email_verified", False),
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
        }
    except Exception as exc:
        logger.warning("Google token verification failed", error=str(exc))
        raise AuthenticationError(f"Invalid Google token: {exc}") from exc


# ── OTP Utilities ─────────────────────────────────────────────────────────────


def generate_otp() -> str:
    """Generate a random 6-digit OTP string."""
    import secrets

    return "".join(str(secrets.randbelow(10)) for _ in range(6))


async def send_email(to_email: str, subject: str, body: str) -> None:
    """Send a plain-text email with the configured SMTP settings.

    Never raises: a missing SMTP config or a send failure is logged, so callers
    can fire it in the background without failing the request.
    """
    from email.message import EmailMessage

    import aiosmtplib

    settings = get_settings()
    if not settings.smtp_username or not settings.smtp_password:
        logger.warning(
            "SMTP credentials not configured — email not sent.", to_email=to_email, subject=subject
        )
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from_email
    msg["To"] = to_email
    msg.set_content(body)
    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            use_tls=False,
            start_tls=settings.smtp_port == 587,
        )
        logger.info("Email sent", to_email=to_email, subject=subject)
    except Exception as exc:
        logger.error("Failed to send email", error=str(exc), to_email=to_email, subject=subject)


async def send_otp_email(to_email: str, otp: str) -> None:
    """Send an OTP email using configured SMTP settings.

    Args:
        to_email: The recipient's email address.
        otp: The 6-digit OTP code.
    """
    from email.message import EmailMessage

    import aiosmtplib

    settings = get_settings()

    if not settings.smtp_username or not settings.smtp_password:
        logger.warning(
            "SMTP credentials not configured — OTP generated but not sent.",
            to_email=to_email,
        )
        return

    msg = EmailMessage()
    msg["Subject"] = "Your Verification Code"
    msg["From"] = settings.smtp_from_email
    msg["To"] = to_email

    msg.set_content(f"""
Hello,

Your verification code is: {otp}

This code will expire in 10 minutes.
If you did not request this, please ignore this email.

Thanks,
Vectrix Team
    """)

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            use_tls=False,
            start_tls=True if settings.smtp_port == 587 else False,
        )
        logger.info("OTP email sent successfully", to_email=to_email)
    except Exception as exc:
        logger.error("Failed to send OTP email", error=str(exc), to_email=to_email)
