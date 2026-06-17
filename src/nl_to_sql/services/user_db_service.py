"""UserDbConnectionService — per-user encrypted database URL storage + connection cache."""
from __future__ import annotations

import base64
import hashlib
from datetime import datetime

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import Base, UserDatabaseConnection
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient

logger = structlog.get_logger(__name__)


def _make_fernet(secret_key: str):
    from cryptography.fernet import Fernet
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
        self._engine = None
        self._session_factory = None
        self._client_cache: dict[str, AsyncDatabaseClient] = {}

    async def initialize(self) -> None:
        self._engine = create_async_engine(
            self._database_url,
            pool_pre_ping=False,
            pool_size=2,
            max_overflow=3,
            pool_recycle=300,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("UserDbConnectionService initialized")

    async def dispose(self) -> None:
        for client in self._client_cache.values():
            await client.dispose()
        self._client_cache.clear()
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
        if user_id in self._client_cache:
            return self._client_cache[user_id]
        raw_url = await self.get_raw(user_id)
        if raw_url is None:
            return None
        client = AsyncDatabaseClient(database_url=raw_url)
        self._client_cache[user_id] = client
        return client

    async def _evict(self, user_id: str) -> None:
        client = self._client_cache.pop(user_id, None)
        if client:
            await client.dispose()
