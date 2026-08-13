"""Versioned Fernet encryption — shared KDF-versioning helper.

``APIKeyService`` and ``ConnectionService`` both encrypt secrets at rest
(API keys, database DSNs) with the same versioned-KDF scheme: v1 is a plain
``sha256(secret_key)`` digest (not a formal KDF), v2 is HKDF-SHA256. New/
updated rows always encrypt with ``CURRENT_KDF_VERSION``; existing rows keep
decrypting with whichever version they were written with (opportunistic
migration on next write, not a bulk backfill — see migration 0019).

Centralizing the derivation + encrypt/decrypt logic here means adding a v3
KDF is one edit instead of two services kept in sync by hand.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

CURRENT_KDF_VERSION = 2

_HKDF_INFO = b"nl2sql-fernet-v2"


def _make_fernet_v1(secret_key: str) -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(secret_key.encode()).digest())
    return Fernet(key)


def _make_fernet_v2(secret_key: str) -> Fernet:
    derived = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_HKDF_INFO).derive(
        secret_key.encode()
    )
    return Fernet(base64.urlsafe_b64encode(derived))


class VersionedFernet:
    """Encrypts with the current KDF version; decrypts with whichever
    version a given row was originally written with."""

    def __init__(self, secret_key: str) -> None:
        self._fernet_v1 = _make_fernet_v1(secret_key)
        self._fernet_v2 = _make_fernet_v2(secret_key)

    async def encrypt(self, value: str) -> str:
        """Encrypt with the current (v2/HKDF) key — every new/updated row upgrades."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._fernet_v2.encrypt(value.encode()).decode()
        )

    async def decrypt(self, value: str, kdf_version: int = 1) -> str:
        """Decrypt with the Fernet matching ``kdf_version`` (legacy rows default to v1)."""
        fernet = self._fernet_v2 if kdf_version >= CURRENT_KDF_VERSION else self._fernet_v1
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fernet.decrypt(value.encode()).decode())
