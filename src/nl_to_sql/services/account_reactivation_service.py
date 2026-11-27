"""AccountReactivationService — deactivated accounts asking to come back.

Flow:
  1. A deactivated account signs in with valid credentials. ``deny_sign_in``
     refuses it (no tokens are issued), emails the owner that the account
     tried to get in (at most once per account per cooldown), and raises an
     error carrying a short-lived reactivation token.
  2. The sign-in page uses that token to ``create_request`` (with an optional
     message). Every active admin is emailed.
  3. An admin approves (reactivates the account) or rejects it from the Admin
     page via ``decide``. The user is emailed the outcome.

This service is the only writer of ``account_reactivation_requests``. Routes
gate who may call what: ``create_request`` needs a valid reactivation token,
``list_requests`` / ``decide`` need ``require_admin``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from nl_to_sql.core.exceptions import (
    AccountDeactivatedError,
    AdminConflictError,
    ReactivationPendingError,
    SessionNotFoundError,
)
from nl_to_sql.infrastructure.database.models import AccountReactivationRequest, User
from nl_to_sql.services.auth_service import create_reactivation_token

logger = structlog.get_logger(__name__)

EmailSender = Callable[[str, str, str], Coroutine[Any, Any, None]]

MAX_MESSAGE_LENGTH = 1000
STATUSES = ("pending", "approved", "rejected")


@dataclass
class ReactivationRequestView:
    id: str
    user_id: str
    email: str
    full_name: str | None
    message: str | None
    status: str
    created_at: datetime
    decided_at: datetime | None
    decided_by_email: str | None


class AccountReactivationService:
    """Blocked sign-ins, reactivation requests, and admin decisions.

    SOLID:
      S — only the deactivated-account → request → decision lifecycle.
      D — session factory and email sender constructor-injected (tests pass a
          recorder instead of SMTP).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        send_email: EmailSender,
        admin_page_url: str = "",
        notify_cooldown_seconds: int = 3600,
    ) -> None:
        self._session_factory = session_factory
        self._send_email = send_email
        self._admin_page_url = admin_page_url
        self._cooldown = notify_cooldown_seconds
        # Per-process: worst case each worker sends one email per account per
        # cooldown. Good enough to stop a retry loop flooding the owner's inbox.
        self._last_owner_alert: dict[str, float] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    # ── Email plumbing ───────────────────────────────────────────────────────

    def _email_later(self, to: str, subject: str, body: str) -> None:
        """Send without blocking the request; failures are logged by the sender."""
        task: asyncio.Task[None] = asyncio.create_task(self._send_email(to, subject, body))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for queued emails (tests; graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def _where(self) -> str:
        return (
            f"Review it on the Admin page → Requests: {self._admin_page_url}"
            if self._admin_page_url
            else ("Review it on the Admin page → Requests.")
        )

    async def _admin_emails(self, db: AsyncSession, *, owner_only: bool) -> list[str]:
        query = select(User.email).where(User.is_admin.is_(True), User.is_active.is_(True))
        if owner_only:
            owner = (await db.execute(query.where(User.is_owner.is_(True)))).scalars().all()
            if owner:
                return list(owner)
            # No owner yet (setup pending): tell whoever administers instead.
        return list((await db.execute(query)).scalars().all())

    # ── 1. Blocked sign-in ───────────────────────────────────────────────────

    async def deny_sign_in(self, user_id: str, email: str) -> NoReturn:
        """Refuse a deactivated account's sign-in and alert the owner.

        Always raises: ``ReactivationPendingError`` when a request is already
        waiting, otherwise ``AccountDeactivatedError`` carrying a reactivation
        token (in ``detail``) for the sign-in page to send a request with.
        """
        async with self._session_factory() as db:
            pending = await self._pending(db, user_id)
            recipients = await self._admin_emails(db, owner_only=True)

        now = time.monotonic()
        last = self._last_owner_alert.get(user_id)
        if last is None or now - last >= self._cooldown:
            self._last_owner_alert[user_id] = now
            when = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            for to in recipients:
                self._email_later(
                    to,
                    f"Deactivated account tried to sign in: {email}",
                    f"Hello,\n\nThe deactivated account {email} tried to sign in to Vectrix at {when}. "
                    "The sign-in was blocked.\n\n"
                    + (
                        "They already have a reactivation request waiting. "
                        if pending
                        else "They can send a reactivation request from the sign-in page. "
                    )
                    + f"{self._where()}\n\nVectrix",
                )
        logger.info(
            "Blocked sign-in of deactivated account", user_id=user_id, pending=bool(pending)
        )

        if pending is not None:
            raise ReactivationPendingError(
                "This account is deactivated. Your reactivation request is waiting for an admin."
            )
        raise AccountDeactivatedError(
            "This account is deactivated. You can ask an admin to reactivate it.",
            detail=create_reactivation_token(user_id),
        )

    # ── 2. Request ───────────────────────────────────────────────────────────

    @staticmethod
    async def _pending(db: AsyncSession, user_id: str) -> AccountReactivationRequest | None:
        return (
            await db.execute(
                select(AccountReactivationRequest).where(
                    AccountReactivationRequest.user_id == user_id,
                    AccountReactivationRequest.status == "pending",
                )
            )
        ).scalar_one_or_none()

    async def create_request(self, user_id: str, message: str | None) -> ReactivationRequestView:
        """File a reactivation request for a deactivated account and email the admins."""
        text = " ".join((message or "").split())[:MAX_MESSAGE_LENGTH] or None
        async with self._session_factory() as db:
            user = await db.get(User, user_id)
            if user is None:
                raise SessionNotFoundError("Account not found.")
            if user.is_active:
                raise AdminConflictError("This account is already active. Sign in again.")
            if await self._pending(db, user_id) is not None:
                raise AdminConflictError("Your request is already waiting for an admin.")
            req = AccountReactivationRequest(user_id=user_id, message=text)
            db.add(req)
            try:
                await db.commit()
            except IntegrityError as exc:  # a concurrent duplicate lost the race
                await db.rollback()
                raise AdminConflictError("Your request is already waiting for an admin.") from exc
            await db.refresh(req)
            recipients = await self._admin_emails(db, owner_only=False)
            view = self._view(req, user, None)

        quoted = f'\n\nTheir message:\n"{text}"' if text else ""
        for to in recipients:
            self._email_later(
                to,
                f"Reactivation request from {user.email}",
                f"Hello,\n\n{user.email} is asking for their deactivated Vectrix account to be "
                f"reactivated.{quoted}\n\n{self._where()} An admin can approve or reject it.\n\nVectrix",
            )
        logger.info("Reactivation requested", user_id=user_id, request_id=view.id)
        return view

    # ── 3. Admin review ──────────────────────────────────────────────────────

    @staticmethod
    def _view(
        r: AccountReactivationRequest, u: User, decided_by_email: str | None
    ) -> ReactivationRequestView:
        return ReactivationRequestView(
            id=r.id,
            user_id=u.id,
            email=u.email,
            full_name=u.full_name,
            message=r.message,
            status=r.status,
            created_at=r.created_at,
            decided_at=r.decided_at,
            decided_by_email=decided_by_email,
        )

    async def count_pending(self) -> int:
        async with self._session_factory() as db:
            n = (
                await db.execute(
                    select(func.count())
                    .select_from(AccountReactivationRequest)
                    .where(AccountReactivationRequest.status == "pending")
                )
            ).scalar_one()
        return int(n)

    async def list_requests(
        self, status: str | None = "pending", limit: int = 50, offset: int = 0
    ) -> tuple[list[ReactivationRequestView], int]:
        """Requests newest first; ``status=None`` lists every status."""
        decider = aliased(User)
        conditions = [AccountReactivationRequest.status == status] if status else []
        async with self._session_factory() as db:
            total = (
                await db.execute(
                    select(func.count()).select_from(AccountReactivationRequest).where(*conditions)
                )
            ).scalar_one()
            rows = (
                await db.execute(
                    select(AccountReactivationRequest, User, decider.email)
                    .join(User, User.id == AccountReactivationRequest.user_id)
                    .outerjoin(decider, decider.id == AccountReactivationRequest.decided_by)
                    .where(*conditions)
                    .order_by(AccountReactivationRequest.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        return [self._view(r, u, by) for r, u, by in rows], int(total)

    async def close_pending_on_reactivate(self, admin_id: str, user_id: str) -> None:
        """An admin reactivated the account directly (Users tab): mark its
        pending request approved so it doesn't linger in the queue."""
        async with self._session_factory() as db:
            req = await self._pending(db, user_id)
            if req is None:
                return
            req.status = "approved"
            req.decided_at = datetime.utcnow()
            req.decided_by = admin_id
            await db.commit()
        self._last_owner_alert.pop(user_id, None)
        logger.info("Pending reactivation closed by direct reactivation", user_id=user_id)

    async def decide(
        self, admin_id: str, request_id: str, approve: bool
    ) -> ReactivationRequestView:
        """Approve (reactivate the account) or reject a pending request."""
        now = datetime.utcnow()
        async with self._session_factory() as db:
            req = await db.get(AccountReactivationRequest, request_id)
            if req is None:
                raise SessionNotFoundError("Request not found.")
            if req.status != "pending":
                raise AdminConflictError(f"This request was already {req.status}.")
            user = await db.get(User, req.user_id)
            if user is None:
                raise SessionNotFoundError("Account not found.")
            admin = await db.get(User, admin_id)
            req.status = "approved" if approve else "rejected"
            req.decided_at = now
            req.decided_by = admin_id
            if approve:
                user.is_active = True
                user.updated_at = now
            await db.commit()
            await db.refresh(req)
            view = self._view(req, user, admin.email if admin else None)

        self._last_owner_alert.pop(user.id, None)
        if approve:
            self._email_later(
                user.email,
                "Your Vectrix account is active again",
                "Hello,\n\nAn admin approved your request. Your Vectrix account is active again "
                "and you can sign in.\n\nVectrix",
            )
        else:
            self._email_later(
                user.email,
                "Your Vectrix reactivation request",
                "Hello,\n\nAn admin reviewed your request and your Vectrix account stays "
                "deactivated. Contact your admin if you think this is a mistake.\n\nVectrix",
            )
        logger.info(
            "Reactivation request decided",
            admin_id=admin_id,
            request_id=request_id,
            user_id=user.id,
            approved=approve,
        )
        return view
