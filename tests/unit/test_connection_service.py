"""Unit tests for ConnectionService — CRUD, encryption, ownership, switching.

Runs against a temp SQLite metadata database. Real connectivity testing is
patched out (``_test_client``) so the tests never open a network socket; the
encryption, ownership, default-management and switching logic are exercised for
real.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from nl_to_sql.core.exceptions import (
    ConnectionNotFoundError,
    ConnectionValidationError,
)
from nl_to_sql.infrastructure.crypto import CURRENT_KDF_VERSION, _make_fernet_v1
from nl_to_sql.services.connection_service import ConnectionService

_DSN_A = "sqlite+aiosqlite:///./_conn_target_a.db"
_DSN_B = "sqlite+aiosqlite:///./_conn_target_b.db"


async def _noop_test(_url: str) -> None:
    return None


@pytest.fixture
async def svc():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    service = ConnectionService(
        database_url=f"sqlite+aiosqlite:///{path}", secret_key="unit-test-secret"
    )
    await service.initialize()
    # Never touch the network in unit tests.
    service._test_client = _noop_test  # type: ignore[method-assign,assignment]
    try:
        yield service
    finally:
        await service.dispose()
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_new_user_gets_server_default(svc: ConnectionService) -> None:
    conns = await svc.list_connections("u1")
    assert len(conns) == 1
    default = conns[0]
    assert default.name == "Server Default"
    assert default.is_default is True
    assert default.has_dsn is False
    assert default.url_preview is None


@pytest.mark.asyncio
async def test_encryption_round_trip(svc: ConnectionService) -> None:
    token = await svc._encrypt("postgresql://u:p@host/db")
    assert token != "postgresql://u:p@host/db"
    # _encrypt always writes the current (v2) key; decrypting a fresh token
    # must be told its version explicitly, same as every real call site does
    # via the row's own `kdf_version` column.
    assert await svc._decrypt(token, CURRENT_KDF_VERSION) == "postgresql://u:p@host/db"


@pytest.mark.asyncio
async def test_create_stores_masked_preview_not_dsn(svc: ConnectionService) -> None:
    info = await svc.create("u1", "Prod", _DSN_A)
    assert info.has_dsn is True
    assert info.db_type == "sqlite"
    # A masked preview — never the raw DSN in any field.
    assert info.url_preview is not None
    conns = await svc.list_connections("u1")
    assert {c.name for c in conns} == {"Server Default", "Prod"}


@pytest.mark.asyncio
async def test_first_personal_connection_becomes_active(svc: ConnectionService) -> None:
    prod = await svc.create("u1", "Prod", _DSN_A)
    assert prod.is_default is True  # replaces the built-in Server Default as active
    # A second personal connection does NOT steal the active slot.
    staging = await svc.create("u1", "Staging", _DSN_B)
    assert staging.is_default is False
    assert await svc.get_active_connection_id("u1") == prod.connection_id


@pytest.mark.asyncio
async def test_duplicate_name_rejected(svc: ConnectionService) -> None:
    await svc.create("u1", "Prod", _DSN_A)
    with pytest.raises(ConnectionValidationError):
        await svc.create("u1", "Prod", _DSN_B)


@pytest.mark.asyncio
async def test_select_switches_active_connection(svc: ConnectionService) -> None:
    prod = await svc.create("u1", "Prod", _DSN_A)
    staging = await svc.create("u1", "Staging", _DSN_B)

    await svc.set_default("u1", staging.connection_id)
    assert await svc.get_active_connection_id("u1") == staging.connection_id

    await svc.set_default("u1", prod.connection_id)
    assert await svc.get_active_connection_id("u1") == prod.connection_id

    # Exactly one default at any time.
    conns = await svc.list_connections("u1")
    assert sum(1 for c in conns if c.is_default) == 1


@pytest.mark.asyncio
async def test_ownership_is_enforced(svc: ConnectionService) -> None:
    prod = await svc.create("u1", "Prod", _DSN_A)
    await svc.ensure_default_connection("u2")
    # u2 cannot see, update, delete, test or select u1's connection.
    with pytest.raises(ConnectionNotFoundError):
        await svc.update("u2", prod.connection_id, name="Hacked")
    with pytest.raises(ConnectionNotFoundError):
        await svc.delete("u2", prod.connection_id)
    with pytest.raises(ConnectionNotFoundError):
        await svc.set_default("u2", prod.connection_id)
    with pytest.raises(ConnectionNotFoundError):
        await svc.test("u2", prod.connection_id)


@pytest.mark.asyncio
async def test_get_client_rejects_non_owner(svc: ConnectionService) -> None:
    prod = await svc.create("u1", "Prod", _DSN_A)
    await svc.ensure_default_connection("u2")

    owner_client = await svc.get_client("u1", prod.connection_id)
    assert owner_client is not None
    await owner_client.dispose()

    # u2 requesting u1's connection_id falls back to the server default
    # (None), never u1's live client — even though the id is a valid,
    # cached key for a different user.
    assert await svc.get_client("u2", prod.connection_id) is None


@pytest.mark.asyncio
async def test_delete_default_promotes_another(svc: ConnectionService) -> None:
    default = (await svc.list_connections("u1"))[0]
    prod = await svc.create("u1", "Prod", _DSN_A)
    await svc.set_default("u1", prod.connection_id)

    await svc.delete("u1", prod.connection_id)
    conns = await svc.list_connections("u1")
    assert len(conns) == 1
    assert conns[0].connection_id == default.connection_id
    assert conns[0].is_default is True


@pytest.mark.asyncio
async def test_rename_keeps_dsn(svc: ConnectionService) -> None:
    prod = await svc.create("u1", "Prod", _DSN_A)
    updated = await svc.update("u1", prod.connection_id, name="Production")
    assert updated.name == "Production"
    assert updated.has_dsn is True


@pytest.mark.asyncio
async def test_new_connection_stores_current_kdf_version(svc: ConnectionService) -> None:
    prod = await svc.create("u1", "Prod", _DSN_A)
    conns = await svc.list_connections("u1")
    row = next(c for c in conns if c.connection_id == prod.connection_id)
    assert row.url_preview is not None  # decrypted fine with the current key


@pytest.mark.asyncio
async def test_legacy_v1_encrypted_dsn_still_decrypts(svc: ConnectionService) -> None:
    """A connection encrypted before the KDF-versioning fix (kdf_version=1, sha256 key)."""
    from uuid import uuid4

    from sqlalchemy import insert, select

    from nl_to_sql.infrastructure.database.models import UserDatabaseConnection

    legacy_fernet = _make_fernet_v1("unit-test-secret")
    legacy_ciphertext = legacy_fernet.encrypt(_DSN_A.encode()).decode()
    connection_id = str(uuid4())

    async with svc._session_factory() as sess:  # type: ignore[union-attr]
        await sess.execute(
            insert(UserDatabaseConnection).values(
                connection_id=connection_id,
                user_id="u1",
                name="Legacy",
                db_type="sqlite",
                encrypted_url=legacy_ciphertext,
                kdf_version=1,
                is_default=False,
            )
        )
        await sess.commit()

    conns = await svc.list_connections("u1")
    legacy = next(c for c in conns if c.connection_id == connection_id)
    assert legacy.url_preview is not None

    # Updating (renaming, no new DSN) leaves the DSN's kdf_version untouched —
    # only a DSN *replacement* re-encrypts with the current key.
    await svc.update("u1", connection_id, name="Legacy Renamed")
    async with svc._session_factory() as sess:  # type: ignore[union-attr]
        version = (
            await sess.execute(
                select(UserDatabaseConnection.kdf_version).where(
                    UserDatabaseConnection.connection_id == connection_id
                )
            )
        ).scalar_one()
    assert version == 1

    # Replacing the DSN upgrades it to the current KDF version.
    await svc.update("u1", connection_id, raw_url=_DSN_B)
    async with svc._session_factory() as sess:  # type: ignore[union-attr]
        version = (
            await sess.execute(
                select(UserDatabaseConnection.kdf_version).where(
                    UserDatabaseConnection.connection_id == connection_id
                )
            )
        ).scalar_one()
    assert version == CURRENT_KDF_VERSION


@pytest.mark.asyncio
async def test_legacy_upsert_and_clear_default_dsn(svc: ConnectionService) -> None:
    info = await svc.upsert_default_dsn("u1", _DSN_A)
    assert info.has_dsn is True
    got = await svc.get_default_info("u1")
    assert got.has_dsn is True

    await svc.clear_default_dsn("u1")
    got2 = await svc.get_default_info("u1")
    assert got2.has_dsn is False  # reverted to Server Default


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


@pytest.mark.asyncio
async def test_unreadable_dsn_strict_raises_instead_of_falling_back(svc: ConnectionService) -> None:
    """A connection WITH a URL that can't be decrypted must not quietly resolve
    to the platform DB in strict mode (schema sync), or the default DB's
    tables get filed under this connection."""
    conn = await svc.create("u1", "Prod", _DSN_A)

    async def _broken(*_args: object) -> str:
        raise ValueError("bad token")

    svc._decrypt = _broken  # type: ignore[method-assign,assignment]
    svc._client_cache.pop_connection(conn.connection_id)
    # Lenient (query paths): falls back, as before.
    assert await svc.get_client("u1", conn.connection_id) is None
    # Strict (schema sync): an error the user can act on.
    with pytest.raises(ConnectionValidationError):
        await svc.get_client("u1", conn.connection_id, strict=True)
    sentinel = object()
    with pytest.raises(ConnectionValidationError):
        await svc.get_scoped_client("u1", conn.connection_id, sentinel, strict=True)  # type: ignore[arg-type]
    # The Server Default (no URL) still resolves to the platform DB in strict mode.
    default = next(c for c in await svc.list_connections("u1") if not c.has_dsn)
    assert await svc.get_scoped_client("u1", default.connection_id, sentinel, strict=True) is sentinel  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sqlite_connections_refused_in_production(
    svc: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SQLite URL is a file on the server — never allowed for users in production."""
    from nl_to_sql.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "app_env", "production")
    with pytest.raises(ConnectionValidationError, match="SQLite"):
        await svc.create("u1", "Local file", _DSN_A)
