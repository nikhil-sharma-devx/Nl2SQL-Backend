"""Versioned KDF for encrypted-at-rest secrets (deferred Low finding).

Revision ID: 0019
Revises: 0018
Create Date: 2026-08-11

`api_key_service.py::_make_fernet` derived the Fernet key via a plain
``sha256(secret_key)`` digest — not a formal KDF. Upgrading the derivation
unconditionally would make every already-encrypted `user_api_keys.encrypted_key`
and `user_database_connections.encrypted_url` permanently undecryptable (the
derived key changes for the same ``secret_key``), with no migration path.

Adds a `kdf_version` column (default 1 = legacy sha256) to both tables so
existing rows keep decrypting with the legacy derivation while every
new/updated row is encrypted with the new HKDF-SHA256 derivation
(`CURRENT_KDF_VERSION = 2`) and stamped accordingly — an opportunistic
migration on next write, not a bulk backfill (backfilling would mean
decrypting and re-encrypting every row for no behavioral gain, and
`secret_key` is already meant to be high-entropy).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("user_api_keys", "user_database_connections")


def _has_column(insp: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    for table in _TABLES:
        if not _has_column(insp, table, "kdf_version"):
            op.add_column(
                table,
                sa.Column(
                    "kdf_version", sa.Integer(), nullable=False, server_default="1"
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    for table in _TABLES:
        if _has_column(insp, table, "kdf_version"):
            op.drop_column(table, "kdf_version")
