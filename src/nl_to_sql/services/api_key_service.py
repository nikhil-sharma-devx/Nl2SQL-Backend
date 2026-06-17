"""APIKeyService — stores per-user LLM API keys, encrypted at rest."""
import base64
import hashlib
import re
from datetime import datetime

import structlog
from cryptography.fernet import Fernet
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.infrastructure.database.models import Base, UserAPIKey

logger = structlog.get_logger(__name__)


def _to_async_url(url: str) -> str:
    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    url = url.replace("sslmode=", "ssl=")
    url = re.sub(r"[?&]channel_binding=[^&]*", "", url)
    url = re.sub(r"[?&]$", "", url)
    return url


def _make_fernet(secret_key: str) -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(secret_key.encode()).digest())
    return Fernet(key)


class APIKeyService:
    """Stores and retrieves per-user LLM API keys encrypted with Fernet symmetric encryption."""

    def __init__(self, database_url: str, secret_key: str) -> None:
        self._database_url = _to_async_url(database_url)
        self._fernet = _make_fernet(secret_key)
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        self._engine = create_async_engine(
            self._database_url,
            pool_pre_ping=False,
            pool_size=2,
            max_overflow=3,
            pool_recycle=300,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("APIKeyService initialized")

    async def dispose(self) -> None:
        if self._engine:
            await self._engine.dispose()

    async def _encrypt(self, value: str) -> str:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._fernet.encrypt(value.encode()).decode())

    async def _decrypt(self, value: str) -> str:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._fernet.decrypt(value.encode()).decode())

    async def save_key(self, user_id: str, provider: str, api_key: str) -> None:
        """Store (or update) an encrypted API key for a user+provider pair."""
        encrypted = await self._encrypt(api_key)
        now = datetime.utcnow()
        async with self._session_factory() as sess:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = (
                pg_insert(UserAPIKey)
                .values(user_id=user_id, provider=provider, encrypted_key=encrypted, created_at=now, updated_at=now)
                .on_conflict_do_update(
                    constraint="uq_user_api_key_provider",
                    set_={"encrypted_key": encrypted, "updated_at": now},
                )
            )
            await sess.execute(stmt)
            await sess.commit()
        logger.info("API key saved", user_id=user_id, provider=provider)

    async def get_key(self, user_id: str, provider: str) -> str | None:
        """Return the decrypted API key for a user+provider pair, or None."""
        if not self._session_factory:
            return None
        try:
            async with self._session_factory() as sess:
                result = await sess.execute(
                    select(UserAPIKey.encrypted_key).where(
                        UserAPIKey.user_id == user_id,
                        UserAPIKey.provider == provider,
                    )
                )
                row = result.one_or_none()
                if row is None:
                    return None
                return await self._decrypt(row.encrypted_key)
        except Exception as exc:
            logger.warning(
                "Failed to get API key",
                user_id=user_id,
                provider=provider,
                error=str(exc),
            )
            return None

    async def delete_key(self, user_id: str, provider: str) -> None:
        """Remove a stored API key for a user+provider pair."""
        async with self._session_factory() as sess:
            await sess.execute(
                delete(UserAPIKey).where(
                    UserAPIKey.user_id == user_id,
                    UserAPIKey.provider == provider,
                )
            )
            await sess.commit()
        logger.info("API key deleted", user_id=user_id, provider=provider)

    async def has_key(self, user_id: str, provider: str) -> bool:
        """Return True if the user has a key stored for the given provider."""
        if not self._session_factory:
            return False
        try:
            async with self._session_factory() as sess:
                result = await sess.execute(
                    select(UserAPIKey.id).where(
                        UserAPIKey.user_id == user_id,
                        UserAPIKey.provider == provider,
                    )
                )
                return result.scalar_one_or_none() is not None
        except Exception:
            return False
