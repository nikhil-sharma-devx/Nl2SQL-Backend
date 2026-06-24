"""Auth routes — register, login, Google sign-in, and current user."""
from __future__ import annotations

from datetime import UTC
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.core.models.auth import (
    ForgotPasswordRequest,
    GoogleAuthRequest,
    OTPResendRequest,
    OTPVerifyRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserPublic,
)
from nl_to_sql.infrastructure.database.models import LoginEvent, User, UserLoginSession
from nl_to_sql.services.auth_service import (
    create_access_token,
    hash_password,
    verify_google_token,
    verify_password,
)
from nl_to_sql.services.chat_session_service import ChatSessionService

# In-memory OTP failure counter keyed by email.
# Safe for single-process deployments; move to Redis for multi-process.
_otp_failures: dict[str, int] = {}
_OTP_MAX_ATTEMPTS = 5

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


def _parse_ua(ua: str | None) -> tuple[str | None, str | None]:
    """Return (device, browser) from a raw User-Agent string."""
    if not ua:
        return None, None
    ua_lower = ua.lower()
    # Device
    if "mobile" in ua_lower or "android" in ua_lower or "iphone" in ua_lower:
        device = "Mobile"
    elif "windows" in ua_lower:
        device = "Windows"
    elif "macintosh" in ua_lower or "mac os" in ua_lower:
        device = "Mac"
    elif "linux" in ua_lower:
        device = "Linux"
    else:
        device = "Unknown"
    # Browser
    if "edg/" in ua_lower or "edge/" in ua_lower:
        browser = "Edge"
    elif "opr/" in ua_lower or "opera" in ua_lower:
        browser = "Opera"
    elif "chrome/" in ua_lower:
        browser = "Chrome"
    elif "firefox/" in ua_lower:
        browser = "Firefox"
    elif "safari/" in ua_lower:
        browser = "Safari"
    else:
        browser = "Unknown"
    return device, browser


async def _record_login(
    user_id: str,
    session_factory: Any,
    request: Request,
    outcome: str = "success",
) -> str | None:
    """Write a LoginEvent and (on success) a UserLoginSession row.

    Returns the new UserLoginSession.id on success, None otherwise.
    """
    from datetime import datetime
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    device, browser = _parse_ua(ua)

    async with session_factory() as db:
        event = LoginEvent(
            user_id=user_id,
            ip=ip,
            user_agent=ua,
            outcome=outcome,
            created_at=datetime.utcnow(),
        )
        db.add(event)

        session_id: str | None = None
        if outcome == "success":
            session = UserLoginSession(
                user_id=user_id,
                device=device,
                browser=browser,
                ip=ip,
                last_active_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )
            db.add(session)
            await db.flush()
            session_id = session.id

        await db.commit()
        return session_id


def _build_token_response(user: User, session_id: str | None = None) -> TokenResponse:
    """Helper: build a TokenResponse for a given User ORM object."""
    token = create_access_token(user_id=user.id, email=user.email, session_id=session_id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserPublic.model_validate(user),
    )


@router.post(
    "/register",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Register a new user and send OTP",
)
@limiter.limit("5/minute")
async def register(
    request: Request,
    body: UserCreate,
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict[str, str]:
    """Create a new user account (unverified) and send an OTP."""
    from datetime import datetime, timedelta

    from nl_to_sql.services.auth_service import generate_otp, send_otp_email

    hashed = hash_password(body.password)
    otp = generate_otp()
    # Expire in 10 minutes
    expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=10)

    new_user = User(
        email=body.email.lower().strip(),
        full_name=body.full_name,
        hashed_password=hashed,
        auth_provider="email",
        is_verified=False,
        otp_code=otp,
        otp_expires_at=expires_at,
    )
    try:
        from nl_to_sql.infrastructure.database.models import PasswordHistory
        async with session_service._session_factory() as db_sess:
            db_sess.add(new_user)
            await db_sess.flush()  # to get new_user.id

            # Record initial password history
            pw_history = PasswordHistory(
                user_id=new_user.id,
                hashed_password=hashed
            )
            db_sess.add(pw_history)

            await db_sess.commit()
            await db_sess.refresh(new_user)
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        ) from None

    # Send OTP asynchronously
    import asyncio
    _bg = asyncio.create_task(send_otp_email(new_user.email, otp))
    _bg.add_done_callback(lambda t: None)  # prevent GC

    logger.info("New user registered (unverified)", email=new_user.email, provider="email")
    return {"message": "OTP sent to email", "email": new_user.email}


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate with email and password",
)
@limiter.limit("5/minute")
async def login(
    request: Request,
    body: UserLogin,
    session_service: ChatSessionService = Depends(get_session_service),
) -> TokenResponse:
    """Validate credentials and return a JWT."""
    async with session_service._session_factory() as db_sess:
        result = await db_sess.execute(
            select(User).where(User.email == body.email.lower().strip())
        )
        user = result.scalar_one_or_none()

    if user is None or not user.hashed_password or not verify_password(body.password, user.hashed_password):
        if user:
            await _record_login(user.id, session_service._session_factory, request, outcome="failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    if not user.is_verified and user.auth_provider == "email":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unverified email. Please verify your OTP.",
        )

    session_id = await _record_login(user.id, session_service._session_factory, request, outcome="success")
    logger.info("User logged in", email=user.email, provider="email")
    return _build_token_response(user, session_id=session_id)


@router.post(
    "/google",
    response_model=TokenResponse,
    summary="Sign in or register via Google OAuth",
)
@limiter.limit("10/minute")
async def google_auth(
    request: Request,
    body: GoogleAuthRequest,
    session_service: ChatSessionService = Depends(get_session_service),
) -> TokenResponse:
    """Verify a Google ID token and login/register the user."""
    try:
        claims = await verify_google_token(body.credential)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    if not claims.get("email_verified"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google email address is not verified.",
        )

    google_sub: str = claims["sub"]
    email: str = claims["email"].lower().strip()
    full_name: str | None = claims.get("name")

    async with session_service._session_factory() as db_sess:
        # Try to find by Google sub first (most stable identifier)
        result = await db_sess.execute(
            select(User).where(User.google_sub == google_sub)
        )
        user = result.scalar_one_or_none()

        if user is None:
            # Try by email (user may have registered by email first)
            result = await db_sess.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()

        if user is None:
            # New Google user — auto-register
            user = User(
                email=email,
                full_name=full_name,
                auth_provider="google",
                google_sub=google_sub,
            )
            db_sess.add(user)
        else:
            # Existing user — link Google sub if not yet linked
            if user.google_sub is None:
                user.google_sub = google_sub
            if full_name and user.full_name is None:
                user.full_name = full_name

        await db_sess.commit()
        await db_sess.refresh(user)

    session_id = await _record_login(user.id, session_service._session_factory, request, outcome="success")
    logger.info("User authenticated via Google", email=user.email)
    return _build_token_response(user, session_id=session_id)


@router.get(
    "/me",
    response_model=UserPublic,
    summary="Get the currently authenticated user",
)
async def get_me(
    current_user: UserPublic = Depends(get_current_user),
) -> UserPublic:
    """Return the authenticated user's profile."""
    return current_user


@router.post(
    "/verify-otp",
    response_model=TokenResponse,
    summary="Verify OTP and activate account",
)
@limiter.limit("3/minute")
async def verify_otp(
    request: Request,
    body: OTPVerifyRequest,
    session_service: ChatSessionService = Depends(get_session_service),
) -> TokenResponse:
    """Verify an OTP and return a JWT."""
    from datetime import datetime

    email_key = body.email.lower().strip()

    async with session_service._session_factory() as db_sess:
        result = await db_sess.execute(
            select(User).where(User.email == email_key)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user.is_verified:
            _otp_failures.pop(email_key, None)
            return _build_token_response(user)

        # Enforce attempt limit before checking the code
        attempts = _otp_failures.get(email_key, 0)
        if attempts >= _OTP_MAX_ATTEMPTS:
            # Invalidate the OTP so attacker must request a new one
            user.otp_code = None
            user.otp_expires_at = None
            await db_sess.commit()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed attempts. Request a new OTP.",
            )

        if not user.otp_code or user.otp_code != body.otp_code:
            _otp_failures[email_key] = attempts + 1
            raise HTTPException(status_code=400, detail="Invalid OTP code")

        now = datetime.now(UTC).replace(tzinfo=None)
        if user.otp_expires_at and now > user.otp_expires_at:
            raise HTTPException(status_code=400, detail="OTP code has expired")

        # Verify success — clear failure counter and OTP
        _otp_failures.pop(email_key, None)
        user.is_verified = True
        user.otp_code = None
        user.otp_expires_at = None

        await db_sess.commit()
        await db_sess.refresh(user)

    session_id = await _record_login(user.id, session_service._session_factory, request, outcome="success")
    logger.info("User verified via OTP", email=user.email)
    return _build_token_response(user, session_id=session_id)


@router.post(
    "/resend-otp",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resend verification OTP",
)
@limiter.limit("3/minute")
async def resend_otp(
    request: Request,
    body: OTPResendRequest,
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict[str, str]:
    """Generate a new OTP and email it to the user."""
    import asyncio
    from datetime import datetime, timedelta

    from nl_to_sql.services.auth_service import generate_otp, send_otp_email

    async with session_service._session_factory() as db_sess:
        result = await db_sess.execute(
            select(User).where(User.email == body.email.lower().strip())
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user.is_verified:
            raise HTTPException(status_code=400, detail="User is already verified")

        # Reset failure counter so the fresh OTP gets a clean slate
        _otp_failures.pop(user.email.lower(), None)
        otp = generate_otp()
        expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=10)

        user.otp_code = otp
        user.otp_expires_at = expires_at

        await db_sess.commit()

    _bg = asyncio.create_task(send_otp_email(user.email, otp))
    _bg.add_done_callback(lambda t: None)  # prevent GC

    logger.info("Resent OTP email", email=user.email)
    return {"message": "New OTP sent to email", "email": user.email}


@router.post(
    "/forgot-password",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Initiate password reset flow",
)
@limiter.limit("3/minute")
async def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict[str, str]:
    """Generate an OTP for password reset."""
    import asyncio
    from datetime import datetime, timedelta

    from nl_to_sql.services.auth_service import generate_otp, send_otp_email

    async with session_service._session_factory() as db_sess:
        result = await db_sess.execute(
            select(User).where(User.email == body.email.lower().strip())
        )
        user = result.scalar_one_or_none()

        # Always return success to prevent email enumeration; only send if user exists
        if user and user.auth_provider == "email":
            # Also reset the failure counter so a fresh OTP gets a clean slate
            _otp_failures.pop(user.email.lower(), None)
            otp = generate_otp()
            expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=10)

            user.otp_code = otp
            user.otp_expires_at = expires_at

            await db_sess.commit()

            _bg = asyncio.create_task(send_otp_email(user.email, otp))
            _bg.add_done_callback(lambda t: None)  # prevent GC
            logger.info("Forgot password OTP generated", email=user.email)

    return {"message": "If that email exists, an OTP has been sent."}


@router.post(
    "/reset-password",
    response_model=TokenResponse,
    summary="Reset password with OTP",
)
@limiter.limit("3/minute")
async def reset_password(
    request: Request,
    body: ResetPasswordRequest,
    session_service: ChatSessionService = Depends(get_session_service),
) -> TokenResponse:
    """Validate OTP and set a new password, enforcing password history."""
    from datetime import datetime

    from nl_to_sql.infrastructure.database.models import PasswordHistory

    async with session_service._session_factory() as db_sess:
        result = await db_sess.execute(
            select(User).where(User.email == body.email.lower().strip())
        )
        user = result.scalar_one_or_none()

        if not user or user.auth_provider != "email":
            raise HTTPException(status_code=400, detail="Invalid request")

        if not user.otp_code or user.otp_code != body.otp_code:
            raise HTTPException(status_code=400, detail="Invalid OTP code")

        now = datetime.now(UTC).replace(tzinfo=None)
        if user.otp_expires_at and now > user.otp_expires_at:
            raise HTTPException(status_code=400, detail="OTP code has expired")

        # Check password history (last 3 passwords)
        result = await db_sess.execute(
            select(PasswordHistory)
            .where(PasswordHistory.user_id == user.id)
            .order_by(PasswordHistory.created_at.desc())
            .limit(3)
        )
        history = result.scalars().all()

        for record in history:
            if verify_password(body.new_password, record.hashed_password):
                raise HTTPException(
                    status_code=400,
                    detail="Password must not be one of your last 3 passwords."
                )

        # Hash new password
        hashed = hash_password(body.new_password)

        # Update user
        user.hashed_password = hashed
        user.otp_code = None
        user.otp_expires_at = None

        # If they weren't verified, verify them now
        user.is_verified = True

        # Insert new password history
        new_history = PasswordHistory(
            user_id=user.id,
            hashed_password=hashed
        )
        db_sess.add(new_history)

        # Optional: delete history older than last 2 to keep table small (new one makes 3)
        # But we can just leave them or clean them up later

        await db_sess.commit()
        await db_sess.refresh(user)

    logger.info("User reset password", email=user.email)
    return _build_token_response(user)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


@router.post(
    "/change-password",
    status_code=status.HTTP_200_OK,
    summary="Change password while authenticated",
)
@limiter.limit("5/minute")
async def change_password(
    request: Request,
    body: ChangePasswordRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict[str, str]:
    """Validate current password, enforce history, then update to the new password."""
    from nl_to_sql.infrastructure.database.models import PasswordHistory

    async with session_service._session_factory() as db_sess:
        result = await db_sess.execute(select(User).where(User.id == current_user.id))
        user = result.scalar_one_or_none()

        if not user or not user.hashed_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password change is not available for this account type",
            )

        if not verify_password(body.current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )

        # Check last 3 passwords
        result = await db_sess.execute(
            select(PasswordHistory)
            .where(PasswordHistory.user_id == user.id)
            .order_by(PasswordHistory.created_at.desc())
            .limit(3)
        )
        history = result.scalars().all()
        for record in history:
            if verify_password(body.new_password, record.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password must not be one of your last 3 passwords",
                )

        hashed = hash_password(body.new_password)
        user.hashed_password = hashed
        db_sess.add(PasswordHistory(user_id=user.id, hashed_password=hashed))
        await db_sess.commit()

    logger.info("User changed password", user_id=current_user.id)
    return {"message": "Password updated successfully"}
