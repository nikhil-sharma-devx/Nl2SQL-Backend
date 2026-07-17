"""P2 - Notification Preferences: email digest, in-app alerts, and marketing opt-ins."""
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import NotificationPreferences
from nl_to_sql.services.chat_session_service import ChatSessionService
from nl_to_sql.services.digest_service import verify_unsubscribe_token

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Notification Preferences"])


class NotificationPrefsOut(BaseModel):
    email_digest: bool
    in_app_enabled: bool
    marketing_enabled: bool


class NotificationPrefsPatch(BaseModel):
    email_digest: bool | None = None
    in_app_enabled: bool | None = None
    marketing_enabled: bool | None = None


def _default_prefs() -> NotificationPrefsOut:
    return NotificationPrefsOut(email_digest=False, in_app_enabled=True, marketing_enabled=False)


def _to_out(p: NotificationPreferences) -> NotificationPrefsOut:
    return NotificationPrefsOut(
        email_digest=p.email_digest,
        in_app_enabled=p.in_app_enabled,
        marketing_enabled=p.marketing_enabled,
    )


@router.get("/notifications/preferences", response_model=NotificationPrefsOut, summary="Get notification preferences")
@router.get("/notification-preferences", response_model=NotificationPrefsOut, include_in_schema=False)
async def get_notification_prefs(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> NotificationPrefsOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(NotificationPreferences).where(NotificationPreferences.user_id == current_user.id)
        )
        row = result.scalar_one_or_none()
    return _to_out(row) if row else _default_prefs()


@router.patch("/notifications/preferences", response_model=NotificationPrefsOut, summary="Update notification preferences")
@router.patch("/notification-preferences", response_model=NotificationPrefsOut, include_in_schema=False)
async def patch_notification_prefs(
    body: NotificationPrefsPatch,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> NotificationPrefsOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(NotificationPreferences).where(NotificationPreferences.user_id == current_user.id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = NotificationPreferences(user_id=current_user.id)
            db.add(row)

        if body.email_digest is not None:
            row.email_digest = body.email_digest
        if body.in_app_enabled is not None:
            row.in_app_enabled = body.in_app_enabled
        if body.marketing_enabled is not None:
            row.marketing_enabled = body.marketing_enabled
        row.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(row)

    logger.info("notification prefs updated", user_id=current_user.id)
    return _to_out(row)


def _unsub_page(message: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Email digest</title></head>"
        "<body style='font-family:system-ui,sans-serif;background:#0b0f14;color:#e5e7eb;"
        "display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0'>"
        "<div style='max-width:420px;text-align:center;padding:32px'>"
        "<h1 style='font-size:20px;margin:0 0 8px'>Email digest</h1>"
        f"<p style='color:#9ca3af;line-height:1.5'>{message}</p>"
        "</div></body></html>"
    )


@router.get(
    "/notifications/unsubscribe",
    response_class=HTMLResponse,
    summary="One-click unsubscribe from the activity email digest",
)
async def unsubscribe_digest(
    token: str,
    session_service: ChatSessionService = Depends(get_session_service),
) -> HTMLResponse:
    """Public, token-authenticated: turn off a user's email digest from an email link."""
    user_id = verify_unsubscribe_token(token)
    if user_id is None:
        return HTMLResponse(
            _unsub_page("This unsubscribe link is invalid or has expired."),
            status_code=400,
        )
    async with session_service._session_factory() as db:
        row = (
            await db.execute(
                select(NotificationPreferences).where(
                    NotificationPreferences.user_id == user_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = NotificationPreferences(user_id=user_id)
            db.add(row)
        row.email_digest = False
        row.updated_at = datetime.utcnow()
        await db.commit()
    logger.info("digest unsubscribe via email link", user_id=user_id)
    return HTMLResponse(_unsub_page("You've been unsubscribed from the email digest."))
