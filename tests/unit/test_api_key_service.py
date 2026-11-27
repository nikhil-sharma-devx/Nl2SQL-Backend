"""Unit tests for APIKeyService — encryption round trip and versioned KDF.

Runs against a temp SQLite metadata database. Exercises the versioned-KDF
scheme directly: legacy (v1, sha256) rows must keep decrypting correctly,
while every new/updated row is written with the current (v2, HKDF) key.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from nl_to_sql.infrastructure.crypto import CURRENT_KDF_VERSION, _make_fernet_v1
from nl_to_sql.services.api_key_service import APIKeyService


@pytest.fixture
async def svc():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    service = APIKeyService(
        secret_key="unit-test-secret", database_url=f"sqlite+aiosqlite:///{path}"
    )
    await service.initialize()
    try:
        yield service
    finally:
        await service.dispose()
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_save_and_get_round_trip(svc: APIKeyService) -> None:
    await svc.save_key("u1", "openai", "sk-abc123")
    assert await svc.get_key("u1", "openai") == "sk-abc123"


@pytest.mark.asyncio
async def test_new_key_is_written_with_current_kdf_version(svc: APIKeyService) -> None:
    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import UserAPIKey

    await svc.save_key("u1", "groq", "gsk-xyz")
    async with svc._session_factory() as sess:  # type: ignore[union-attr]
        row = (
            await sess.execute(
                select(UserAPIKey.kdf_version).where(
                    UserAPIKey.user_id == "u1", UserAPIKey.provider == "groq"
                )
            )
        ).scalar_one()
    assert row == CURRENT_KDF_VERSION


@pytest.mark.asyncio
async def test_legacy_v1_encrypted_row_still_decrypts(svc: APIKeyService) -> None:
    """A row encrypted before the KDF-versioning fix (kdf_version=1, sha256 key)."""
    from sqlalchemy import insert

    from nl_to_sql.infrastructure.database.models import UserAPIKey

    legacy_fernet = _make_fernet_v1("unit-test-secret")
    legacy_ciphertext = legacy_fernet.encrypt(b"legacy-key-value").decode()

    async with svc._session_factory() as sess:  # type: ignore[union-attr]
        await sess.execute(
            insert(UserAPIKey).values(
                user_id="u2",
                provider="anthropic",
                encrypted_key=legacy_ciphertext,
                kdf_version=1,
            )
        )
        await sess.commit()

    assert await svc.get_key("u2", "anthropic") == "legacy-key-value"


@pytest.mark.asyncio
async def test_overwriting_a_legacy_key_upgrades_its_kdf_version(svc: APIKeyService) -> None:
    from sqlalchemy import insert, select

    from nl_to_sql.infrastructure.database.models import UserAPIKey

    legacy_fernet = _make_fernet_v1("unit-test-secret")
    async with svc._session_factory() as sess:  # type: ignore[union-attr]
        await sess.execute(
            insert(UserAPIKey).values(
                user_id="u3",
                provider="gemini",
                encrypted_key=legacy_fernet.encrypt(b"old-value").decode(),
                kdf_version=1,
            )
        )
        await sess.commit()

    await svc.save_key("u3", "gemini", "new-value")

    async with svc._session_factory() as sess:  # type: ignore[union-attr]
        row = (
            await sess.execute(
                select(UserAPIKey.kdf_version).where(
                    UserAPIKey.user_id == "u3", UserAPIKey.provider == "gemini"
                )
            )
        ).scalar_one()
    assert row == CURRENT_KDF_VERSION
    assert await svc.get_key("u3", "gemini") == "new-value"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
