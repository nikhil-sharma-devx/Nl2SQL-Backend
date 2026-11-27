"""Unit tests for infrastructure.crypto.VersionedFernet — the shared
KDF-versioning helper deduplicated out of APIKeyService/ConnectionService."""
from __future__ import annotations

import pytest
from cryptography.fernet import InvalidToken

from nl_to_sql.infrastructure.crypto import (
    CURRENT_KDF_VERSION,
    VersionedFernet,
    _make_fernet_v1,
)


class TestVersionedFernet:
    async def test_round_trip_with_current_version(self) -> None:
        crypto = VersionedFernet("unit-test-secret")
        token = await crypto.encrypt("super-secret-value")
        assert await crypto.decrypt(token, CURRENT_KDF_VERSION) == "super-secret-value"

    async def test_legacy_v1_token_still_decrypts(self) -> None:
        crypto = VersionedFernet("unit-test-secret")
        legacy_fernet = _make_fernet_v1("unit-test-secret")
        legacy_token = legacy_fernet.encrypt(b"legacy-value").decode()
        assert await crypto.decrypt(legacy_token, kdf_version=1) == "legacy-value"

    async def test_default_kdf_version_is_legacy(self) -> None:
        """Rows with no stored kdf_version must default to v1, not the current version."""
        crypto = VersionedFernet("unit-test-secret")
        legacy_fernet = _make_fernet_v1("unit-test-secret")
        legacy_token = legacy_fernet.encrypt(b"legacy-value").decode()
        assert await crypto.decrypt(legacy_token) == "legacy-value"

    async def test_different_secret_keys_are_not_interchangeable(self) -> None:
        crypto_a = VersionedFernet("secret-a")
        crypto_b = VersionedFernet("secret-b")
        token = await crypto_a.encrypt("value")
        with pytest.raises(InvalidToken):
            await crypto_b.decrypt(token, CURRENT_KDF_VERSION)
