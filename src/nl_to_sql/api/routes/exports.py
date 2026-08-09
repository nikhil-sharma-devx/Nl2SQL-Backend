"""Export & Share routes.

Export a query + its result as CSV / JSON / SQL / PDF, and create secure,
optionally-expiring share links that can be revoked or delivered via email or
Slack. Public link access is by signed token only; every mutating/owner action
is scoped to ``current_user.id`` (cross-user access returns 404, never 403).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from starlette.responses import StreamingResponse

from nl_to_sql.api.dependencies import (
    get_current_user,
    get_export_service,
    get_session_service,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import SharedQuery
from nl_to_sql.services.chat_session_service import ChatSessionService
from nl_to_sql.services.export_service import (
    ExportFormat,
    ExportService,
    cap_rows,
    make_share_token,
    verify_share_token,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Export & Share"])


# ── Request / response models ─────────────────────────────────────────────────


class ExportRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=100_000)
    question: str | None = Field(default=None, max_length=4000)
    # Accept either `rows` or `results` for the result set.
    rows: list[dict[str, Any]] | None = None
    results: list[dict[str, Any]] | None = None
    format: ExportFormat = "csv"

    def resolved_rows(self) -> list[dict[str, Any]]:
        return self.rows if self.rows is not None else (self.results or [])


class ShareCreate(BaseModel):
    sql: str = Field(..., min_length=1, max_length=100_000)
    question: str | None = Field(default=None, max_length=4000)
    title: str | None = Field(default=None, max_length=200)
    rows: list[dict[str, Any]] | None = None
    results: list[dict[str, Any]] | None = None
    expires_in_days: int | None = Field(default=None, ge=1, le=365)

    def resolved_rows(self) -> list[dict[str, Any]]:
        return self.rows if self.rows is not None else (self.results or [])


class ShareCreateResponse(BaseModel):
    id: str
    token: str
    url: str
    expires_at: datetime | None = None


class SharedSnapshot(BaseModel):
    """Public view of a share — snapshot only, no owner PII."""

    title: str | None
    question: str
    sql: str
    results: list[dict[str, Any]]
    created_at: datetime
    expires_at: datetime | None


class ShareEmailRequest(BaseModel):
    to_email: str = Field(..., min_length=3, max_length=320)


class DeliveryResponse(BaseModel):
    sent: bool
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _share_url(token: str) -> str:
    """Absolute link to the shared query.

    Prefers the configured public app base URL (frontend ``/shared/<token>``
    view); falls back to the first CORS origin, then to the relative API path so
    the link always resolves to the public GET endpoint.
    """
    settings = get_settings()
    base = settings.app_base_url.strip().rstrip("/")
    if not base and settings.cors_origin_list:
        base = settings.cors_origin_list[0].rstrip("/")
    if base:
        return f"{base}/shared/{token}"
    return f"/api/v1/shares/{token}"


# ── Export ────────────────────────────────────────────────────────────────────


@router.post("/exports/query", summary="Export a query + result as CSV/JSON/SQL/PDF")
@limiter.limit("30/minute")
async def export_query(
    request: Request,
    body: ExportRequest,
    current_user: UserPublic = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service),
) -> StreamingResponse:
    """Build the requested artifact and stream it as a download."""
    content, media_type, filename = export_service.build(
        body.format, body.question or "", body.sql, body.resolved_rows()
    )
    logger.info(
        "export built", user_id=current_user.id, format=body.format, bytes=len(content)
    )
    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Shares ────────────────────────────────────────────────────────────────────


@router.post(
    "/shares",
    response_model=ShareCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a shareable link for a query + result",
)
@limiter.limit("20/minute")
async def create_share(
    request: Request,
    body: ShareCreate,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> ShareCreateResponse:
    share_id = str(uuid4())
    expires_at = (
        datetime.utcnow() + timedelta(days=body.expires_in_days)
        if body.expires_in_days
        else None
    )
    question = body.question or ""
    snapshot = cap_rows(body.resolved_rows())

    async with session_service._session_factory() as db:
        share = SharedQuery(
            id=share_id,
            user_id=current_user.id,
            title=body.title,
            question=question,
            nl_prompt=question,
            generated_sql=body.sql,
            result_snapshot=snapshot,
            expires_at=expires_at,
        )
        db.add(share)
        await db.commit()

    token = make_share_token(share_id, expires_at)
    logger.info("share created", user_id=current_user.id, share_id=share_id)
    return ShareCreateResponse(
        id=share_id, token=token, url=_share_url(token), expires_at=expires_at
    )


@router.get(
    "/shares/{token}",
    response_model=SharedSnapshot,
    summary="Public: fetch a shared query snapshot by token",
)
async def get_shared_query(
    token: str,
    session_service: ChatSessionService = Depends(get_session_service),
) -> SharedSnapshot:
    """Public, token-authenticated. Returns only the shared snapshot."""
    share_id = verify_share_token(token)
    if share_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share not found")

    async with session_service._session_factory() as db:
        share = (
            await db.execute(select(SharedQuery).where(SharedQuery.id == share_id))
        ).scalar_one_or_none()

    if share is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share not found")
    if share.revoked_at is not None:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="This share has been revoked")
    if share.expires_at is not None and share.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="This share has expired")

    return SharedSnapshot(
        title=share.title,
        question=share.question or "",
        sql=share.generated_sql or "",
        results=share.result_snapshot or [],
        created_at=share.created_at,
        expires_at=share.expires_at,
    )


async def _get_owned_share(
    session_service: ChatSessionService, share_id: str, user_id: str
) -> SharedQuery:
    """Load a share owned by ``user_id`` or raise 404 (cross-user is invisible)."""
    async with session_service._session_factory() as db:
        share = (
            await db.execute(
                select(SharedQuery).where(
                    SharedQuery.id == share_id, SharedQuery.user_id == user_id
                )
            )
        ).scalar_one_or_none()
    if share is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share not found")
    return share


@router.delete(
    "/shares/{share_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke a share (owner only)",
)
async def revoke_share(
    share_id: str,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> None:
    async with session_service._session_factory() as db:
        share = (
            await db.execute(
                select(SharedQuery).where(
                    SharedQuery.id == share_id, SharedQuery.user_id == current_user.id
                )
            )
        ).scalar_one_or_none()
        if share is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share not found")
        share.revoked_at = datetime.utcnow()
        await db.commit()
    logger.info("share revoked", user_id=current_user.id, share_id=share_id)


@router.post(
    "/shares/{share_id}/email",
    response_model=DeliveryResponse,
    summary="Email a share link (owner only)",
)
@limiter.limit("10/minute")
async def email_share(
    request: Request,
    share_id: str,
    body: ShareEmailRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
    export_service: ExportService = Depends(get_export_service),
) -> DeliveryResponse:
    share = await _get_owned_share(session_service, share_id, current_user.id)
    if not export_service.smtp_configured:
        return DeliveryResponse(sent=False, message="Email delivery is not configured.")

    token = make_share_token(share.id, share.expires_at)
    url = _share_url(token)
    title = share.title or "a query"
    subject = f"{current_user.full_name or 'Someone'} shared {title} with you"
    text = (
        f"{current_user.full_name or 'Someone'} shared a Vectrix query with you.\n\n"
        f"View it here: {url}\n"
    )
    html = (
        f"<div style=\"font-family:system-ui,sans-serif\">"
        f"<p>{_esc(current_user.full_name or 'Someone')} shared a Vectrix query with you.</p>"
        f"<p><a href=\"{_esc(url)}\">Open the shared query</a></p></div>"
    )
    sent = await export_service.send_share_email(body.to_email, subject, text, html)
    logger.info("share email attempted", user_id=current_user.id, share_id=share_id, sent=sent)
    return DeliveryResponse(
        sent=sent,
        message="Email sent." if sent else "Could not send the email.",
    )


@router.post(
    "/shares/{share_id}/slack",
    response_model=DeliveryResponse,
    summary="Send a share link to Slack (owner only)",
)
@limiter.limit("10/minute")
async def slack_share(
    request: Request,
    share_id: str,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
    export_service: ExportService = Depends(get_export_service),
) -> DeliveryResponse:
    share = await _get_owned_share(session_service, share_id, current_user.id)
    settings = get_settings()
    if not export_service.slack_configured:
        return DeliveryResponse(sent=False, message="Slack delivery is not configured.")

    token = make_share_token(share.id, share.expires_at)
    url = _share_url(token)
    text = f"*{share.title or 'Shared query'}*\n{url}"
    sent = await export_service.send_to_slack(settings.slack_webhook_url, text)
    logger.info("share slack attempted", user_id=current_user.id, share_id=share_id, sent=sent)
    return DeliveryResponse(
        sent=sent,
        message="Sent to Slack." if sent else "Could not send to Slack.",
    )


def _esc(s: str) -> str:
    """Minimal HTML escaping for user-supplied strings in the email body."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
