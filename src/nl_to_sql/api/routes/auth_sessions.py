"""F8 - Login session management and login activity routes."""
from datetime import datetime
from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy import select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import LoginEvent, UserLoginSession
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Auth Sessions"])

_bearer = HTTPBearer(auto_error=False)


def _get_session_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)] = None,
) -> str | None:
    """Extract session_id from the current JWT without re-validating (get_current_user already did)."""
    if credentials is None:
        return None
    try:
        from nl_to_sql.services.auth_service import decode_access_token
        return decode_access_token(credentials.credentials).session_id
    except Exception:
        return None


class LoginSessionOut(BaseModel):
    id: str
    device: Optional[str]
    browser: Optional[str]
    ip: Optional[str]
    last_active_at: datetime
    created_at: datetime
    current: bool = False


class LoginActivityOut(BaseModel):
    ip: Optional[str]
    user_agent: Optional[str]
    outcome: str
    created_at: datetime


@router.get("/auth-sessions", summary="List active login sessions")
async def list_auth_sessions(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
    current_session_id: str | None = Depends(_get_session_id),
) -> dict:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserLoginSession).where(
                UserLoginSession.user_id == current_user.id,
                UserLoginSession.revoked_at.is_(None),
            ).order_by(UserLoginSession.last_active_at.desc())
        )
        sessions = result.scalars().all()

    return {
        "items": [
            LoginSessionOut(
                id=s.id,
                device=s.device,
                browser=s.browser,
                ip=s.ip,
                last_active_at=s.last_active_at,
                created_at=s.created_at,
                current=(s.id == current_session_id),
            ).model_dump()
            for s in sessions
        ]
    }


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke the current session (logout)")
async def logout(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
    current_session_id: str | None = Depends(_get_session_id),
) -> None:
    if not current_session_id:
        return
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserLoginSession).where(
                UserLoginSession.id == current_session_id,
                UserLoginSession.user_id == current_user.id,
            )
        )
        s = result.scalar_one_or_none()
        if s:
            s.revoked_at = datetime.utcnow()
            await db.commit()


@router.delete("/auth-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke a login session")
async def revoke_auth_session(
    session_id: str,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> None:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserLoginSession).where(
                UserLoginSession.id == session_id,
                UserLoginSession.user_id == current_user.id,
            )
        )
        s = result.scalar_one_or_none()
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        s.revoked_at = datetime.utcnow()
        await db.commit()


@router.delete("/auth-sessions", summary="Revoke all other login sessions")
async def revoke_all_auth_sessions(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict:
    revoked = 0
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserLoginSession).where(
                UserLoginSession.user_id == current_user.id,
                UserLoginSession.revoked_at.is_(None),
            )
        )
        sessions = result.scalars().all()
        now = datetime.utcnow()
        for s in sessions:
            s.revoked_at = now
            revoked += 1
        await db.commit()

    return {"revoked": revoked}


@router.get("/login-activity", summary="Get login event history")
async def get_login_activity(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(LoginEvent)
            .where(LoginEvent.user_id == current_user.id)
            .order_by(LoginEvent.created_at.desc())
            .limit(limit)
        )
        events = result.scalars().all()

    return {
        "items": [
            LoginActivityOut(
                ip=e.ip,
                user_agent=e.user_agent,
                outcome=e.outcome,
                created_at=e.created_at,
            ).model_dump()
            for e in events
        ]
    }
