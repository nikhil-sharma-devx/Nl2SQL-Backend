"""Email digest service — builds and sends a periodic activity summary.

Pure helpers (no DI): the digest worker calls these with a DB session. Sending
is gated by the caller on SMTP being configured + per-user opt-in + cadence, so
this module never decides *whether* to send — only *how* to build and deliver.

Unsubscribe uses a signed (HS256) token over the user id so the one-click link
in the email needs no auth and can't be forged.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nl_to_sql.config.settings import get_settings
from nl_to_sql.infrastructure.database.models import ChatMessage, ChatSession

logger = structlog.get_logger(__name__)

_UNSUB_PURPOSE = "unsub_digest"


# ── Unsubscribe token ────────────────────────────────────────────────────────


def make_unsubscribe_token(user_id: str) -> str:
    """Return a signed, non-expiring token that authorises digest unsubscribe."""
    from jose import jwt

    settings = get_settings()
    token: str = jwt.encode(
        {"sub": user_id, "purpose": _UNSUB_PURPOSE},
        settings.secret_key,
        algorithm="HS256",
    )
    return token


def verify_unsubscribe_token(token: str) -> str | None:
    """Return the user id if ``token`` is a valid unsubscribe token, else None."""
    from jose import JWTError, jwt

    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except JWTError:
        return None
    if payload.get("purpose") != _UNSUB_PURPOSE:
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) and sub else None


def _unsubscribe_url(token: str) -> str:
    """Build the absolute unsubscribe URL for the email body."""
    settings = get_settings()
    base = settings.app_base_url.strip().rstrip("/")
    if not base and settings.cors_allowed_origins:
        base = settings.cors_allowed_origins.split(",")[0].strip().rstrip("/")
    return f"{base}/api/v1/notifications/unsubscribe?token={token}"


def build_unsubscribe_url(user_id: str) -> str:
    """Return a ready-to-embed one-click unsubscribe URL for a user."""
    return _unsubscribe_url(make_unsubscribe_token(user_id))


# ── Digest content ───────────────────────────────────────────────────────────


async def build_user_digest(
    db: AsyncSession, user_id: str, since: datetime
) -> dict[str, Any] | None:
    """Aggregate a user's query activity since ``since``.

    Returns None when there is nothing worth emailing (no queries in the window).
    """
    row = (
        await db.execute(
            select(
                func.count(ChatMessage.id),
                func.coalesce(func.sum(ChatMessage.tokens_used), 0),
            )
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .where(
                ChatSession.user_id == user_id,
                ChatMessage.timestamp >= since,
                ChatMessage.deleted_at.is_(None),
                ChatMessage.sql != "",
            )
        )
    ).one()

    queries = int(row[0])
    if queries == 0:
        return None

    recent = (
        (
            await db.execute(
                select(ChatMessage.question)
                .join(ChatSession, ChatMessage.session_id == ChatSession.id)
                .where(
                    ChatSession.user_id == user_id,
                    ChatMessage.timestamp >= since,
                    ChatMessage.deleted_at.is_(None),
                    ChatMessage.sql != "",
                )
                .order_by(ChatMessage.timestamp.desc())
                .limit(5)
            )
        )
        .scalars()
        .all()
    )

    return {
        "queries": queries,
        "tokens": int(row[1]),
        "recent_questions": [q for q in recent if q],
    }


def render_digest_email(
    full_name: str | None, digest: dict[str, Any], period_days: int, unsubscribe_url: str
) -> tuple[str, str, str]:
    """Return (subject, plain_text, html) for the digest email."""
    name = full_name or "there"
    queries = digest["queries"]
    tokens = digest["tokens"]
    recent = digest.get("recent_questions", [])
    subject = f"Your NL2SQL activity — {queries} quer{'y' if queries == 1 else 'ies'} this week"

    recent_txt = "\n".join(f"  • {q}" for q in recent) or "  (none)"
    text = (
        f"Hi {name},\n\n"
        f"Here's your activity from the last {period_days} days:\n\n"
        f"  Queries run: {queries}\n"
        f"  Tokens used: {tokens:,}\n\n"
        f"Recent questions:\n{recent_txt}\n\n"
        f"— NL2SQL\n\n"
        f"Unsubscribe from these emails: {unsubscribe_url}\n"
    )

    recent_html = "".join(f"<li>{_esc(q)}</li>" for q in recent) or "<li>(none)</li>"
    html = (
        f"<div style=\"font-family:system-ui,sans-serif;max-width:520px;margin:0 auto;color:#111\">"
        f"<h2 style=\"margin:0 0 4px\">Your NL2SQL activity</h2>"
        f"<p style=\"color:#555;margin:0 0 16px\">Last {period_days} days</p>"
        f"<p>Hi {_esc(name)},</p>"
        f"<table style=\"border-collapse:collapse;margin:12px 0\">"
        f"<tr><td style=\"padding:6px 16px 6px 0\"><b>{queries}</b></td>"
        f"<td style=\"color:#555\">queries run</td></tr>"
        f"<tr><td style=\"padding:6px 16px 6px 0\"><b>{tokens:,}</b></td>"
        f"<td style=\"color:#555\">tokens used</td></tr></table>"
        f"<p style=\"margin:16px 0 4px\"><b>Recent questions</b></p>"
        f"<ul style=\"color:#333\">{recent_html}</ul>"
        f"<hr style=\"border:none;border-top:1px solid #eee;margin:20px 0\">"
        f"<p style=\"font-size:12px;color:#999\">"
        f"You're receiving this because you enabled the email digest. "
        f"<a href=\"{_esc(unsubscribe_url)}\">Unsubscribe</a>.</p>"
        f"</div>"
    )
    return subject, text, html


def _esc(s: str) -> str:
    """Minimal HTML escaping for user-supplied strings in the email body."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ── Delivery ─────────────────────────────────────────────────────────────────


async def send_digest_email(to_email: str, subject: str, text: str, html: str) -> bool:
    """Send one digest email via SMTP. Returns True on success, False otherwise."""
    from email.message import EmailMessage

    import aiosmtplib

    settings = get_settings()
    if not settings.smtp_username or not settings.smtp_password:
        logger.warning("digest: SMTP not configured — skipping send", to_email=to_email)
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from_email
    msg["To"] = to_email
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

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
        logger.info("digest: email sent", to_email=to_email)
        return True
    except Exception as exc:
        logger.error("digest: email send failed", to_email=to_email, error=str(exc))
        return False


def digest_window_start(interval_days: int) -> datetime:
    """Return the start of the digest window (now - interval)."""
    return datetime.utcnow() - timedelta(days=interval_days)
