"""Email digest worker — sends a periodic activity summary to opted-in users.

Idempotent and safe by default: a digest is sent only when ALL of these hold —
  1. the master flag ``email_digest_enabled`` is on,
  2. SMTP credentials are configured,
  3. the user opted in (``NotificationPreferences.email_digest``),
  4. the send cadence has elapsed (``last_digest_sent_at`` older than the interval),
  5. the user actually has query activity in the window.
Run on the maintenance scheduler loop; the per-user cadence guard means the loop
interval can be shorter than the digest interval without over-sending.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.config.settings import get_settings
from nl_to_sql.infrastructure.database.models import NotificationPreferences, User
from nl_to_sql.services.digest_service import (
    build_unsubscribe_url,
    build_user_digest,
    digest_window_start,
    render_digest_email,
    send_digest_email,
)

logger = structlog.get_logger(__name__)


async def run_digest_cycle(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, int]:
    """Send digests to every eligible opted-in user. Returns a summary."""
    settings = get_settings()

    if not settings.email_digest_enabled:
        return {"digests_sent": 0, "digests_skipped": 0}
    if not (settings.smtp_username and settings.smtp_password):
        logger.info("digest_worker: SMTP not configured — skipping cycle")
        return {"digests_sent": 0, "digests_skipped": 0}

    interval = settings.email_digest_interval_days
    now = datetime.utcnow()
    cadence_cutoff = now - timedelta(days=interval)
    since = digest_window_start(interval)

    async with session_factory() as db:
        candidates = (
            await db.execute(
                select(NotificationPreferences, User)
                .join(User, User.id == NotificationPreferences.user_id)
                .where(
                    NotificationPreferences.email_digest.is_(True),
                    User.is_active.is_(True),
                    or_(
                        NotificationPreferences.last_digest_sent_at.is_(None),
                        NotificationPreferences.last_digest_sent_at < cadence_cutoff,
                    ),
                )
            )
        ).all()

    sent = 0
    skipped = 0
    for _prefs, user in candidates:
        try:
            async with session_factory() as db:
                digest = await build_user_digest(db, user.id, since)
            if digest is None:
                skipped += 1  # No activity in the window — nothing worth emailing.
                continue

            subject, text, html = render_digest_email(
                user.full_name, digest, interval, build_unsubscribe_url(user.id)
            )
            if await send_digest_email(user.email, subject, text, html):
                async with session_factory() as db:
                    prefs = await db.get(NotificationPreferences, user.id)
                    if prefs is not None:
                        prefs.last_digest_sent_at = now
                        await db.commit()
                sent += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.error("digest_worker: user send failed", user_id=user.id, error=str(exc))
            skipped += 1

    logger.info("digest_worker: cycle complete", digests_sent=sent, digests_skipped=skipped)
    return {"digests_sent": sent, "digests_skipped": skipped}
