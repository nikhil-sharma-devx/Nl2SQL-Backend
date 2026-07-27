"""Cron parsing helpers for Scheduled Queries & Alerts.

Translates a small set of common natural-language schedule phrases ("every
morning", "daily at 9") into canonical 5-field cron expressions, and computes
the next UTC fire time for a cron expression + IANA timezone. A literal cron
expression typed directly is also accepted and passed through unchanged.

All ``next_run_at``/``last_run_at``-style timestamps stored in the database are
naive UTC (matching every other ``DateTime`` column in this codebase) — this
module is the only place that converts between a user's timezone and UTC.
"""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from nl_to_sql.core.exceptions import ScheduleValidationError

# A literal 5-field cron expression (minute hour day month weekday), each field
# built from digits/`*`/`/`/`,`/`-`.
_CRON_LITERAL_RE = re.compile(r"^[\d*/,\-]+(?:\s+[\d*/,\-]+){4}$")

# "at 9", "at 9am", "at 9:30 pm" — captures hour, optional minute, optional am/pm.
_AT_TIME_RE = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE
)

_WEEKDAY_NAMES = {
    "monday": 1, "mon": 1,
    "tuesday": 2, "tue": 2, "tues": 2,
    "wednesday": 3, "wed": 3,
    "thursday": 4, "thu": 4, "thurs": 4,
    "friday": 5, "fri": 5,
    "saturday": 6, "sat": 6,
    "sunday": 0, "sun": 0,
}

_DEFAULT_HOUR = 8  # "every morning" / bare "daily" default fire hour


def _parse_at_time(text: str) -> tuple[int, int] | None:
    """Extract an (hour, minute) 24h pair from an "at H[:MM][am|pm]" phrase."""
    match = _AT_TIME_RE.search(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def parse_schedule_text(text: str) -> str:
    """Translate a natural-language schedule phrase into a canonical cron string.

    Recognized phrases (case-insensitive): "every morning", "every hour",
    "every day"/"daily", "every <weekday>", "weekdays", optionally combined
    with "at H[:MM][am|pm]" (e.g. "daily at 9", "every Monday at 9:30pm",
    "weekdays at 9am"). A literal 5-field cron expression typed directly is
    passed through unchanged (after validation). Anything else raises
    ``ScheduleValidationError``.
    """
    raw = text.strip()
    if not raw:
        raise ScheduleValidationError("Schedule text cannot be empty.")

    if _CRON_LITERAL_RE.match(raw):
        validate_cron(raw)
        return raw

    lowered = raw.lower()
    at_time = _parse_at_time(lowered)

    if "every hour" in lowered or lowered == "hourly":
        return "0 * * * *"

    if "weekday" in lowered:
        hour, minute = at_time or (_DEFAULT_HOUR, 0)
        return f"{minute} {hour} * * 1-5"

    for name, iso_dow in _WEEKDAY_NAMES.items():
        if re.search(rf"\b{name}\b", lowered):
            hour, minute = at_time or (_DEFAULT_HOUR, 0)
            return f"{minute} {hour} * * {iso_dow}"

    if "every morning" in lowered or "morning" in lowered:
        hour, minute = at_time or (_DEFAULT_HOUR, 0)
        return f"{minute} {hour} * * *"

    if "every day" in lowered or "daily" in lowered or lowered.startswith("day"):
        hour, minute = at_time or (_DEFAULT_HOUR, 0)
        return f"{minute} {hour} * * *"

    if at_time is not None:
        hour, minute = at_time
        return f"{minute} {hour} * * *"

    raise ScheduleValidationError(
        f"Could not understand schedule '{text}'. Try phrases like "
        "'every morning', 'daily at 9am', 'every Monday', 'weekdays at 9', "
        "or a literal 5-field cron expression."
    )


def validate_cron(expr: str) -> None:
    """Raise ``ScheduleValidationError`` if ``expr`` is not a valid cron string."""
    if not croniter.is_valid(expr):
        raise ScheduleValidationError(f"Invalid cron expression: '{expr}'.")


def compute_next_run(
    cron_expr: str, tz_name: str, after: datetime | None = None
) -> datetime:
    """Return the next UTC fire time for ``cron_expr`` evaluated in ``tz_name``.

    ``after`` defaults to now (UTC). The result is naive UTC, matching every
    other ``DateTime`` column in this codebase.
    """
    validate_cron(cron_expr)
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise ScheduleValidationError(f"Unknown timezone: '{tz_name}'.") from exc

    base = after or datetime.utcnow()
    if base.tzinfo is None:
        base = base.replace(tzinfo=ZoneInfo("UTC"))
    base_local = base.astimezone(tz)

    itr = croniter(cron_expr, base_local)
    next_local = itr.get_next(datetime)
    return next_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
