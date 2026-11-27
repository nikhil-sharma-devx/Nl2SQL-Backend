"""Unit tests for the email digest service (token, content, rendering)."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from jose import jwt

from nl_to_sql.config.settings import get_settings
from nl_to_sql.services.digest_service import (
    build_user_digest,
    digest_window_start,
    make_unsubscribe_token,
    render_digest_email,
    verify_unsubscribe_token,
)


class TestUnsubscribeToken:
    def test_round_trip(self) -> None:
        token = make_unsubscribe_token("user-abc")
        assert verify_unsubscribe_token(token) == "user-abc"

    def test_garbage_is_rejected(self) -> None:
        assert verify_unsubscribe_token("not-a-token") is None

    def test_wrong_purpose_is_rejected(self) -> None:
        # A validly-signed token with the wrong purpose must not unsubscribe.
        bad = jwt.encode(
            {"sub": "user-abc", "purpose": "something_else"},
            get_settings().secret_key,
            algorithm="HS256",
        )
        assert verify_unsubscribe_token(bad) is None

    def test_wrong_secret_is_rejected(self) -> None:
        forged = jwt.encode(
            {"sub": "user-abc", "purpose": "unsub_digest"},
            "the-wrong-secret",
            algorithm="HS256",
        )
        assert verify_unsubscribe_token(forged) is None


class TestRenderDigestEmail:
    def test_contains_stats_and_unsubscribe(self) -> None:
        subject, text, html = render_digest_email(
            "Nik",
            {"queries": 5, "tokens": 1200, "recent_questions": ["top customers"]},
            7,
            "https://api.example.com/api/v1/notifications/unsubscribe?token=abc",
        )
        assert "5" in subject
        assert "top customers" in text
        assert "unsubscribe" in text.lower()
        assert "1,200" in html  # thousands-formatted tokens
        assert "https://api.example.com/api/v1/notifications/unsubscribe?token=abc" in html

    def test_html_escapes_user_content(self) -> None:
        _s, _t, html = render_digest_email(
            "Nik",
            {"queries": 1, "tokens": 0, "recent_questions": ["<script>alert(1)</script>"]},
            7,
            "https://x/u",
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_handles_missing_name(self) -> None:
        _s, text, _h = render_digest_email(
            None, {"queries": 2, "tokens": 10, "recent_questions": []}, 7, "https://x/u"
        )
        assert "there" in text  # fallback greeting


class TestBuildUserDigest:
    async def test_returns_none_when_no_activity(self) -> None:
        db = MagicMock()
        count_res = MagicMock()
        count_res.one = MagicMock(return_value=(0, 0))
        db.execute = AsyncMock(return_value=count_res)
        result = await build_user_digest(db, "user-1", digest_window_start(7))
        assert result is None

    async def test_aggregates_activity(self) -> None:
        db = MagicMock()
        count_res = MagicMock()
        count_res.one = MagicMock(return_value=(3, 450))
        recent_res = MagicMock()
        recent_res.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=["q1", "q2", "q3"]))
        )
        db.execute = AsyncMock(side_effect=[count_res, recent_res])
        result = await build_user_digest(db, "user-1", digest_window_start(7))
        assert result == {"queries": 3, "tokens": 450, "recent_questions": ["q1", "q2", "q3"]}


class TestDigestWindow:
    def test_window_start_is_in_the_past(self) -> None:
        assert digest_window_start(7) < datetime.utcnow()
