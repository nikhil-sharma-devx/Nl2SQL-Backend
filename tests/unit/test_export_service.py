"""Unit tests for ExportService builders + share token/slack helpers."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta

import pytest

from nl_to_sql.config.settings import get_settings
from nl_to_sql.services.export_service import (
    ExportService,
    make_share_token,
    verify_share_token,
)

ROWS = [
    {"id": 1, "name": "Alice", "amount": 10},
    {"id": 2, "name": "Bob", "amount": 20},
]


@pytest.fixture
def service() -> ExportService:
    return ExportService(settings=get_settings())


class TestBuilders:
    def test_to_csv_content(self, service: ExportService) -> None:
        raw = service.to_csv("q", "SELECT 1", ROWS)
        text = raw.decode("utf-8")
        reader = list(csv.DictReader(io.StringIO(text)))
        assert [r["name"] for r in reader] == ["Alice", "Bob"]
        assert reader[0]["amount"] == "10"

    def test_to_csv_empty_rows(self, service: ExportService) -> None:
        assert service.to_csv("q", "SELECT 1", []) == b""

    def test_to_json_content(self, service: ExportService) -> None:
        raw = service.to_json("my question", "SELECT 1", ROWS)
        data = json.loads(raw)
        assert data["question"] == "my question"
        assert data["sql"] == "SELECT 1"
        assert data["row_count"] == 2
        assert data["results"] == ROWS

    def test_to_sql_bytes(self, service: ExportService) -> None:
        raw = service.to_sql("count users", "SELECT count(*) FROM users", ROWS)
        text = raw.decode("utf-8")
        assert "SELECT count(*) FROM users" in text
        assert text.startswith("-- count users")

    def test_to_pdf_starts_with_magic(self, service: ExportService) -> None:
        raw = service.to_pdf("q", "SELECT 1", ROWS)
        assert isinstance(raw, bytes)
        assert raw.startswith(b"%PDF")

    def test_to_pdf_handles_unicode(self, service: ExportService) -> None:
        raw = service.to_pdf("Queé 你好", "SELECT 'é'", ROWS)
        assert raw.startswith(b"%PDF")

    def test_build_returns_media_type_and_filename(self, service: ExportService) -> None:
        _content, media_type, filename = service.build("csv", "q", "SELECT 1", ROWS)
        assert media_type == "text/csv"
        assert filename == "query_export.csv"
        _, pdf_media, pdf_name = service.build("pdf", "q", "SELECT 1", ROWS)
        assert pdf_media == "application/pdf"
        assert pdf_name == "query_export.pdf"


class TestShareToken:
    def test_make_and_verify_roundtrip(self) -> None:
        token = make_share_token("share-123", None)
        assert verify_share_token(token) == "share-123"

    def test_verify_rejects_garbage(self) -> None:
        assert verify_share_token("not-a-token") is None

    def test_verify_rejects_wrong_purpose(self) -> None:
        # A token minted for a different purpose (digest unsubscribe) must not
        # be accepted as a share token.
        from nl_to_sql.services.digest_service import make_unsubscribe_token

        assert verify_share_token(make_unsubscribe_token("u1")) is None

    def test_expired_token_still_resolves_id(self) -> None:
        # exp is informational (DB is authoritative) — an expired token must
        # still resolve to its share id so the route can return 410, not 404.
        past = datetime.utcnow() - timedelta(days=1)
        token = make_share_token("share-exp", past)
        assert verify_share_token(token) == "share-exp"


@pytest.fixture
def unconfigured_service() -> ExportService:
    """A service whose SMTP/Slack channels are explicitly empty (env-independent)."""
    settings = get_settings().model_copy(
        update={"smtp_username": "", "smtp_password": "", "slack_webhook_url": ""}
    )
    return ExportService(settings=settings)


class TestSlackDelivery:
    async def test_slack_graceful_when_unconfigured(
        self, unconfigured_service: ExportService
    ) -> None:
        assert unconfigured_service.slack_configured is False
        assert await unconfigured_service.send_to_slack("", "hello") is False

    async def test_email_graceful_when_unconfigured(
        self, unconfigured_service: ExportService
    ) -> None:
        assert unconfigured_service.smtp_configured is False
        sent = await unconfigured_service.send_share_email("to@x.com", "s", "t", "<p>t</p>")
        assert sent is False
