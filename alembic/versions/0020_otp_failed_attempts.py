"""DB-backed OTP failure counter (High finding).

Revision ID: 0020
Revises: 0019
Create Date: 2026-08-13

``_otp_failures`` in ``api/routes/auth.py`` was an in-memory, per-process
dict backing the OTP lockout on both ``/verify-otp`` and ``/reset-password``.
On a multi-worker deployment (``gunicorn -w N``, multiple pods) each worker
keeps its own counter, so an attacker effectively gets ``N * 5`` attempts
before any single worker's lockout reliably trips.

Adds an ``otp_failed_attempts`` column to ``users`` so the lockout counter
lives on the row itself — shared by every worker — instead of process memory.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_column(insp: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if not _has_column(insp, "users", "otp_failed_attempts"):
        op.add_column(
            "users",
            sa.Column("otp_failed_attempts", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if _has_column(insp, "users", "otp_failed_attempts"):
        op.drop_column("users", "otp_failed_attempts")
