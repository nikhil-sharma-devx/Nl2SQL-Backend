"""Unit tests for cron_utils — NL schedule phrase parsing + next-run computation.

Pure functions, no DB/session mocking needed.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from nl_to_sql.core.exceptions import ScheduleValidationError
from nl_to_sql.services.cron_utils import (
    compute_next_run,
    parse_schedule_text,
    validate_cron,
)


class TestParseScheduleText:
    def test_every_morning(self) -> None:
        assert parse_schedule_text("every morning") == "0 8 * * *"

    def test_every_monday(self) -> None:
        assert parse_schedule_text("every Monday") == "0 8 * * 1"

    def test_daily_at_9(self) -> None:
        assert parse_schedule_text("daily at 9") == "0 9 * * *"

    def test_every_day_at_9am(self) -> None:
        assert parse_schedule_text("every day at 9am") == "0 9 * * *"

    def test_every_day_at_9pm(self) -> None:
        assert parse_schedule_text("every day at 9pm") == "0 21 * * *"

    def test_every_hour(self) -> None:
        assert parse_schedule_text("every hour") == "0 * * * *"

    def test_weekdays_at_9(self) -> None:
        assert parse_schedule_text("weekdays at 9") == "0 9 * * 1-5"

    def test_monday_at_930pm(self) -> None:
        assert parse_schedule_text("every Monday at 9:30pm") == "30 21 * * 1"

    def test_literal_cron_passthrough(self) -> None:
        assert parse_schedule_text("15 4 * * *") == "15 4 * * *"

    def test_empty_raises(self) -> None:
        with pytest.raises(ScheduleValidationError):
            parse_schedule_text("")

    def test_unrecognized_phrase_raises(self) -> None:
        with pytest.raises(ScheduleValidationError):
            parse_schedule_text("when the moon is full")

    def test_invalid_literal_cron_raises(self) -> None:
        with pytest.raises(ScheduleValidationError):
            parse_schedule_text("99 99 * * *")


class TestValidateCron:
    def test_valid_expression_ok(self) -> None:
        validate_cron("0 8 * * *")  # no raise

    def test_invalid_expression_raises(self) -> None:
        with pytest.raises(ScheduleValidationError):
            validate_cron("not a cron")


class TestComputeNextRun:
    def test_next_run_is_in_the_future(self) -> None:
        after = datetime(2026, 1, 1, 0, 0, 0)
        next_run = compute_next_run("0 8 * * *", "UTC", after=after)
        assert next_run > after
        assert next_run == datetime(2026, 1, 1, 8, 0, 0)

    def test_timezone_conversion_to_utc(self) -> None:
        # 8am in America/New_York (UTC-5 in January) is 13:00 UTC.
        after = datetime(2026, 1, 1, 0, 0, 0)
        next_run = compute_next_run("0 8 * * *", "America/New_York", after=after)
        assert next_run == datetime(2026, 1, 1, 13, 0, 0)

    def test_unknown_timezone_raises(self) -> None:
        with pytest.raises(ScheduleValidationError):
            compute_next_run("0 8 * * *", "Not/A_Zone")

    def test_invalid_cron_raises(self) -> None:
        with pytest.raises(ScheduleValidationError):
            compute_next_run("bad cron", "UTC")
