"""Unit tests for the email digest worker — safety gates + happy path."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from nl_to_sql.workers import digest_worker


def _factory(db):
    """A session_factory() returning an async-context-manager over ``db``."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=db)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


def _settings(**over):
    base = {
        "email_digest_enabled": True,
        "smtp_username": "user",
        "smtp_password": "pass",
        "email_digest_interval_days": 7,
    }
    base.update(over)
    return SimpleNamespace(**base)


class TestDigestGates:
    async def test_skips_when_master_flag_off(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock()
        with patch.object(
            digest_worker, "get_settings", return_value=_settings(email_digest_enabled=False)
        ):
            result = await digest_worker.run_digest_cycle(_factory(db))
        assert result == {"digests_sent": 0, "digests_skipped": 0}
        db.execute.assert_not_called()  # never even queries candidates

    async def test_skips_when_smtp_not_configured(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock()
        with patch.object(
            digest_worker, "get_settings", return_value=_settings(smtp_username="")
        ):
            result = await digest_worker.run_digest_cycle(_factory(db))
        assert result == {"digests_sent": 0, "digests_skipped": 0}
        db.execute.assert_not_called()


class TestDigestSend:
    async def test_sends_and_stamps_cadence(self) -> None:
        user = SimpleNamespace(id="u1", email="u1@example.com", full_name="U One")
        prefs = SimpleNamespace(user_id="u1", email_digest=True, last_digest_sent_at=None)

        candidates = MagicMock()
        candidates.all = MagicMock(return_value=[(prefs, user)])
        db = MagicMock()
        db.execute = AsyncMock(return_value=candidates)
        db.get = AsyncMock(return_value=prefs)
        db.commit = AsyncMock()

        send = AsyncMock(return_value=True)
        digest = {"queries": 4, "tokens": 99, "recent_questions": ["q"]}
        with (
            patch.object(digest_worker, "get_settings", return_value=_settings()),
            patch.object(digest_worker, "build_user_digest", new=AsyncMock(return_value=digest)),
            patch.object(digest_worker, "send_digest_email", new=send),
        ):
            result = await digest_worker.run_digest_cycle(_factory(db))

        assert result == {"digests_sent": 1, "digests_skipped": 0}
        send.assert_awaited_once()
        assert prefs.last_digest_sent_at is not None  # cadence stamped so it won't resend
        db.commit.assert_awaited()

    async def test_skips_user_with_no_activity(self) -> None:
        user = SimpleNamespace(id="u2", email="u2@example.com", full_name=None)
        prefs = SimpleNamespace(user_id="u2", email_digest=True, last_digest_sent_at=None)
        candidates = MagicMock()
        candidates.all = MagicMock(return_value=[(prefs, user)])
        db = MagicMock()
        db.execute = AsyncMock(return_value=candidates)
        db.commit = AsyncMock()

        send = AsyncMock(return_value=True)
        with (
            patch.object(digest_worker, "get_settings", return_value=_settings()),
            patch.object(digest_worker, "build_user_digest", new=AsyncMock(return_value=None)),
            patch.object(digest_worker, "send_digest_email", new=send),
        ):
            result = await digest_worker.run_digest_cycle(_factory(db))

        assert result == {"digests_sent": 0, "digests_skipped": 1}
        send.assert_not_awaited()  # no email for an inactive user
