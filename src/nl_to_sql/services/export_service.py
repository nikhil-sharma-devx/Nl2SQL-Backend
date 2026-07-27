"""Export & Share service — build downloadable artifacts and deliver share links.

Two concerns live here:

1. **Export builders** (``ExportService.to_csv/to_json/to_sql/to_pdf``): pure,
   side-effect-free functions turning ``(question, sql, rows)`` into ``bytes``
   for a StreamingResponse. They only ever serialise data the caller already
   holds — no DSNs, secrets, or other users' data ever reach them.

2. **Share tokens + delivery**: signed (HS256) share tokens over
   ``settings.secret_key`` (mirroring ``digest_service`` unsubscribe tokens) so a
   public share link needs no auth and can't be forged, plus Slack (httpx
   webhook) and email (aiosmtplib) delivery that degrade gracefully when the
   channel isn't configured.

The DB row (``SharedQuery``) is authoritative for expiry/revocation — the token
merely authenticates the share id (and carries an informational ``exp`` when an
expiry is set), so the route can distinguish 410 (expired/revoked) from 404
(unknown/forged).
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Literal

import structlog

from nl_to_sql.config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)

ExportFormat = Literal["csv", "json", "sql", "pdf"]

# Bound the result snapshot persisted with a share and rendered into a PDF so a
# huge result set can never blow up storage or the document.
MAX_SNAPSHOT_ROWS = 1000
MAX_PDF_ROWS = 100

_SHARE_PURPOSE = "share"

_CONTENT_TYPES: dict[str, tuple[str, str]] = {
    "csv": ("text/csv", "csv"),
    "json": ("application/json", "json"),
    "sql": ("application/sql", "sql"),
    "pdf": ("application/pdf", "pdf"),
}


# ── Share token (signed, HS256 over secret_key) ───────────────────────────────


def make_share_token(share_id: str, expires_at: datetime | None) -> str:
    """Return a signed token that authenticates a share id.

    Includes an ``exp`` claim when ``expires_at`` is set (informational — the
    DB row is authoritative for expiry), plus a ``purpose`` claim so a token
    minted for another feature can't be replayed here.
    """
    from jose import jwt

    settings = get_settings()
    claims: dict[str, Any] = {"sub": share_id, "purpose": _SHARE_PURPOSE}
    if expires_at is not None:
        claims["exp"] = int(expires_at.timestamp())
    token: str = jwt.encode(claims, settings.secret_key, algorithm="HS256")
    return token


def verify_share_token(token: str) -> str | None:
    """Return the share id if ``token`` is a valid share token, else None.

    Signature and ``purpose`` are enforced; the ``exp`` claim is intentionally
    NOT enforced here so an expired token still resolves to its share id and the
    route can return 410 (expired) rather than 404 (unknown) — the DB drives the
    expiry/revocation decision.
    """
    from jose import JWTError, jwt

    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=["HS256"],
            options={"verify_exp": False},
        )
    except JWTError:
        return None
    if payload.get("purpose") != _SHARE_PURPOSE:
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) and sub else None


def cap_rows(rows: list[dict[str, Any]] | None, limit: int = MAX_SNAPSHOT_ROWS) -> list[dict[str, Any]]:
    """Return at most ``limit`` rows (defensive copy), never None."""
    if not rows:
        return []
    return list(rows[:limit])


def _latin1(text: str) -> str:
    """Coerce text to the latin-1 subset fpdf core fonts can render."""
    return text.encode("latin-1", "replace").decode("latin-1")


class ExportService:
    """Builds export artifacts and delivers share links.

    SOLID:
      S — one domain: turning a query+result into shareable/downloadable form.
      D — receives its configuration (SMTP/Slack) via an injected ``Settings``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ── Pure builders ─────────────────────────────────────────────────────────

    def to_csv(self, question: str, sql: str, rows: list[dict[str, Any]] | None) -> bytes:
        """Serialise the result rows as CSV (header from the first row's keys)."""
        capped = cap_rows(rows)
        buffer = io.StringIO()
        if capped:
            fieldnames = list(capped[0].keys())
            writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in capped:
                writer.writerow({k: _stringify(row.get(k)) for k in fieldnames})
        return buffer.getvalue().encode("utf-8")

    def to_json(self, question: str, sql: str, rows: list[dict[str, Any]] | None) -> bytes:
        """Serialise the full query + bounded result set as JSON."""
        payload = {
            "question": question,
            "sql": sql,
            "row_count": len(rows or []),
            "results": cap_rows(rows),
        }
        return json.dumps(payload, indent=2, default=str).encode("utf-8")

    def to_sql(self, question: str, sql: str, rows: list[dict[str, Any]] | None) -> bytes:
        """Return the SQL text as bytes, prefixed with the question as a comment."""
        header = ""
        if question:
            safe = question.replace("\r", " ").replace("\n", " ")
            header = f"-- {safe}\n"
        body = sql or ""
        if body and not body.endswith("\n"):
            body += "\n"
        return (header + body).encode("utf-8")

    def to_pdf(self, question: str, sql: str, rows: list[dict[str, Any]] | None) -> bytes:
        """Build a simple PDF report: question, SQL, and a bounded result table."""
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos

        def line(pdf: FPDF, text: str, height: float = 7.0) -> None:
            pdf.cell(0, height, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        line(pdf, "NL2SQL Query Export", 10)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(120, 120, 120)
        line(pdf, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), 6)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        if question:
            pdf.set_font("Helvetica", "B", 11)
            line(pdf, "Question")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, _latin1(question))
            pdf.ln(2)

        pdf.set_font("Helvetica", "B", 11)
        line(pdf, "SQL")
        pdf.set_font("Courier", "", 9)
        pdf.multi_cell(0, 5, _latin1(sql or ""))
        pdf.ln(2)

        capped = cap_rows(rows, MAX_PDF_ROWS)
        pdf.set_font("Helvetica", "B", 11)
        line(pdf, f"Results ({len(rows or [])} rows)")
        if capped:
            columns = list(capped[0].keys())
            col_width = max(20.0, (pdf.epw / max(1, len(columns))))
            pdf.set_font("Helvetica", "B", 8)
            for col in columns:
                pdf.cell(col_width, 6, _latin1(str(col))[:18], border=1)
            pdf.ln()
            pdf.set_font("Helvetica", "", 8)
            for row in capped:
                for col in columns:
                    pdf.cell(col_width, 6, _latin1(_stringify(row.get(col)))[:18], border=1)
                pdf.ln()
        else:
            pdf.set_font("Helvetica", "", 10)
            line(pdf, "No rows.", 6)

        out = pdf.output()
        return bytes(out)

    def build(
        self, fmt: ExportFormat, question: str, sql: str, rows: list[dict[str, Any]] | None
    ) -> tuple[bytes, str, str]:
        """Return ``(content, media_type, filename)`` for the requested format."""
        builders = {
            "csv": self.to_csv,
            "json": self.to_json,
            "sql": self.to_sql,
            "pdf": self.to_pdf,
        }
        content = builders[fmt](question, sql, rows)
        media_type, ext = _CONTENT_TYPES[fmt]
        return content, media_type, f"query_export.{ext}"

    # ── Delivery ──────────────────────────────────────────────────────────────

    @property
    def slack_configured(self) -> bool:
        return bool(self._settings.slack_webhook_url)

    @property
    def smtp_configured(self) -> bool:
        return bool(self._settings.smtp_username and self._settings.smtp_password)

    async def send_to_slack(self, webhook_url: str, text: str) -> bool:
        """POST ``text`` to a Slack incoming webhook. Graceful False when unset."""
        if not webhook_url:
            logger.warning("share: Slack webhook not configured — skipping send")
            return False
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook_url, json={"text": text})
            ok = 200 <= resp.status_code < 300
            if not ok:
                logger.error("share: Slack webhook rejected", status=resp.status_code)
            return ok
        except Exception as exc:
            logger.error("share: Slack send failed", error=str(exc))
            return False

    async def send_share_email(
        self, to_email: str, subject: str, text: str, html: str
    ) -> bool:
        """Send a share link email via SMTP. Returns True on success, else False."""
        from email.message import EmailMessage

        import aiosmtplib

        settings = self._settings
        if not self.smtp_configured:
            logger.warning("share: SMTP not configured — skipping send", to_email=to_email)
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
            logger.info("share: email sent", to_email=to_email)
            return True
        except Exception as exc:
            logger.error("share: email send failed", to_email=to_email, error=str(exc))
            return False


def _stringify(value: Any) -> str:
    """Render a cell value as a string (empty for NULL)."""
    if value is None:
        return ""
    return str(value)
