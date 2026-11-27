"""Mark Google-linked accounts as verified (data fix).

Revision ID: 0023
Revises: 0022
Create Date: 2026-11-27

Google sign-in rejects any Google account whose email isn't verified, but it
created / linked users without setting ``users.is_verified``. Those accounts
were therefore treated as unfinished sign-ups: hidden from the team directory
and impossible to invite. A user only ever gets a ``google_sub`` after Google
confirmed their email, so every such row is verified.

Downgrade is a no-op: the rows this fixed can't be told apart from accounts
that were verified some other way, and un-verifying them would only
re-introduce the bug.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE users SET is_verified = :yes WHERE google_sub IS NOT NULL AND is_verified = :no"
        ).bindparams(yes=True, no=False)
    )


def downgrade() -> None:
    # Intentionally irreversible (see module docstring).
    pass
