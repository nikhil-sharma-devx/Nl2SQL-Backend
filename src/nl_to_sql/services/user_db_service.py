"""UserDbConnectionService — per-user encrypted database URL storage + connection cache."""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import socket
from collections import OrderedDict
from datetime import datetime
from urllib.parse import urlparse

import structlog
from cryptography.fernet import Fernet
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nl_to_sql.infrastructure.database.models import Base, UserDatabaseConnection
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url

logger = structlog.get_logger(__name__)

_CACHE_MAX_SIZE = 100


class _LRUClientCache:
    """Thread-unsafe LRU cache for AsyncDatabaseClient instances.

    Thread-safety is not required because asyncio is single-threaded and all
    mutations happen within the same event loop.
    """

    def __init__(self, maxsize: int) -> None:
        self._cache: OrderedDict[str, AsyncDatabaseClient] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> AsyncDatabaseClient | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: AsyncDatabaseClient) -> AsyncDatabaseClient | None:
        """Insert/update an entry. Returns the evicted client if the cache was full."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return None
        evicted: AsyncDatabaseClient | None = None
        if len(self._cache) >= self._maxsize:
            _, evicted = self._cache.popitem(last=False)
        self._cache[key] = value
        return evicted

    def pop(self, key: str) -> AsyncDatabaseClient | None:
        return self._cache.pop(key, None)

    def drain(self) -> list[AsyncDatabaseClient]:
        """Remove all entries and return them for disposal."""
        clients = list(self._cache.values())
        self._cache.clear()
        return clients


_ALLOWED_SCHEMES = frozenset({"postgresql+asyncpg", "mysql+aiomysql", "sqlite+aiosqlite"})

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _validate_db_url(url: str) -> None:
    """Raise ValueError if the URL scheme is not allowed or the host resolves to a private IP."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Database URL scheme '{scheme}' is not allowed.")
    host = parsed.hostname
    if not host:
        raise ValueError("Database URL must include a host.")
    # sqlite is file-based; no host to validate
    if scheme == "sqlite+aiosqlite":
        return
    try:
        results = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve database host '{host}': {exc}") from exc
    for _, _, _, _, sockaddr in results:
        addr = ipaddress.ip_address(sockaddr[0])
        if any(addr in net for net in _PRIVATE_NETS):
            raise ValueError(
                f"Database host '{host}' resolves to a private/internal IP address and is not allowed."
            )


def _make_fernet(secret_key: str) -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(secret_key.encode()).digest())
    return Fernet(key)


class UserDbConnectionService:
    """Stores per-user PostgreSQL URLs encrypted at rest and caches live connections.

    Storage: `user_database_connections` table in the app metadata database.
    Encryption: Fernet symmetric (same key derivation as APIKeyService).
    Cache: in-memory dict of user_id → AsyncDatabaseClient (disposed on delete).
    """

    def __init__(self, database_url: str, secret_key: str) -> None:
        from nl_to_sql.services.api_key_service import _to_async_url
        self._database_url = _to_async_url(database_url)
        self._fernet = _make_fernet(secret_key)
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._client_cache = _LRUClientCache(maxsize=_CACHE_MAX_SIZE)

    async def initialize(self) -> None:
        self._engine = create_async_engine(
            self._database_url,
            pool_pre_ping=False,
            pool_size=1,
            max_overflow=1,
            pool_recycle=300,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("UserDbConnectionService initialized")

    async def dispose(self) -> None:
        for client in self._client_cache.drain():
            await client.dispose()
        if self._engine:
            await self._engine.dispose()

    # ── Encryption helpers ────────────────────────────────────────────────────

    async def _encrypt(self, value: str) -> str:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._fernet.encrypt(value.encode()).decode())

    async def _decrypt(self, value: str) -> str:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._fernet.decrypt(value.encode()).decode())

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def save(self, user_id: str, raw_url: str) -> None:
        """Normalise, validate, encrypt and store a database URL for a user."""
        normalised = to_async_database_url(raw_url.strip())
        try:
            _validate_db_url(normalised)
        except ValueError as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if self._session_factory is None:
            raise RuntimeError("UserDbConnectionService.initialize() must be called before use")
        encrypted = await self._encrypt(normalised)
        now = datetime.utcnow()
        async with self._session_factory() as sess:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = (
                pg_insert(UserDatabaseConnection)
                .values(user_id=user_id, encrypted_url=encrypted, created_at=now, updated_at=now)
                .on_conflict_do_update(
                    index_elements=["user_id"],
                    set_={"encrypted_url": encrypted, "updated_at": now},
                )
            )
            await sess.execute(stmt)
            await sess.commit()
        # Evict stale cached client so next request gets a fresh one
        await self._evict(user_id)
        logger.info("User database URL saved", user_id=user_id)

    async def get_raw(self, user_id: str) -> str | None:
        """Return the decrypted (normalised) URL for a user, or None."""
        if not self._session_factory:
            return None
        try:
            async with self._session_factory() as sess:
                result = await sess.execute(
                    select(UserDatabaseConnection.encrypted_url).where(
                        UserDatabaseConnection.user_id == user_id
                    )
                )
                row = result.scalar_one_or_none()
                if row is None:
                    return None
                return await self._decrypt(row)
        except Exception as exc:
            logger.warning("Failed to get user database URL", user_id=user_id, error=str(exc))
            return None

    async def delete(self, user_id: str) -> None:
        """Remove stored URL and dispose the cached client."""
        if self._session_factory is None:
            raise RuntimeError("UserDbConnectionService.initialize() must be called before use")
        async with self._session_factory() as sess:
            await sess.execute(
                delete(UserDatabaseConnection).where(
                    UserDatabaseConnection.user_id == user_id
                )
            )
            await sess.commit()
        await self._evict(user_id)
        logger.info("User database URL deleted", user_id=user_id)

    async def has_connection(self, user_id: str) -> bool:
        if not self._session_factory:
            return False
        try:
            async with self._session_factory() as sess:
                result = await sess.execute(
                    select(UserDatabaseConnection.id).where(
                        UserDatabaseConnection.user_id == user_id
                    )
                )
                return result.scalar_one_or_none() is not None
        except Exception:
            return False

    # ── Connection cache ──────────────────────────────────────────────────────

    async def get_client(self, user_id: str) -> AsyncDatabaseClient | None:
        """Return a live AsyncDatabaseClient for the user, creating one if needed."""
        cached = self._client_cache.get(user_id)
        if cached is not None:
            return cached
        raw_url = await self.get_raw(user_id)
        if raw_url is None:
            return None
        client = AsyncDatabaseClient(database_url=raw_url)
        evicted = self._client_cache.put(user_id, client)
        if evicted is not None:
            await evicted.dispose()
        return client

    async def _evict(self, user_id: str) -> None:
        client = self._client_cache.pop(user_id)
        if client:
            await client.dispose()
