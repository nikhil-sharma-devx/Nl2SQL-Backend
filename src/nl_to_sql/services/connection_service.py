"""ConnectionService — per-user *multiple* database connections (BYOD).

Owns everything about a user's database connections:

  - CRUD (create / rename+edit / delete / list),
  - encryption of the DSN at rest (reuses the ``api_key_service`` Fernet
    derivation — no second crypto mechanism),
  - ownership validation (cross-user access is reported as *not found*),
  - connectivity testing,
  - default/active selection and resolution,
  - a bounded LRU cache of live :class:`AsyncDatabaseClient` instances keyed by
    ``connection_id`` (pooled, disposed on evict/delete/rename).

A connection whose ``encrypted_url`` is ``NULL`` is the built-in **Server
Default** connection: it resolves to the platform's shared database
(``get_client`` returns ``None`` and callers fall back to the server client).
Every user is guaranteed at least one connection (a Server Default is created on
first access) so the pre-multi-connection experience is preserved.

DSNs are never returned by any method — only a masked ``url_preview``.
"""
from __future__ import annotations

import ipaddress
import socket
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
from uuid import uuid4

import structlog
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nl_to_sql.core.exceptions import (
    ConnectionNotFoundError,
    ConnectionTestError,
    ConnectionValidationError,
)
from nl_to_sql.infrastructure.database.models import Base, UserDatabaseConnection
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url
from nl_to_sql.services.api_key_service import _make_fernet, _to_async_url

logger = structlog.get_logger(__name__)

_CACHE_MAX_SIZE = 100
_MAX_CONNECTIONS_PER_USER = 25

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
    """Raise ValueError if the scheme is not allowed or the host is a private IP (SSRF)."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Database URL scheme '{scheme}' is not allowed.")
    # sqlite is file-based — no host to validate (must be checked before the
    # host requirement below, which only applies to networked databases).
    if scheme == "sqlite+aiosqlite":
        return
    host = parsed.hostname
    if not host:
        raise ValueError("Database URL must include a host.")
    try:
        results = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve database host '{host}': {exc}") from exc
    for *_, sockaddr in results:
        addr = ipaddress.ip_address(sockaddr[0])
        if any(addr in net for net in _PRIVATE_NETS):
            raise ValueError(
                f"Database host '{host}' resolves to a private/internal IP address "
                "and is not allowed."
            )


def _mask_url(url: str) -> str:
    """Replace the password in a connection URL with *** for safe display/logs."""
    import re

    return re.sub(r"(:)[^:@]+(@)", r"\1***\2", url)


def _derive_db_type(url: str) -> str:
    """Best-effort database type from a normalised URL scheme."""
    scheme = urlparse(url).scheme.lower()
    if scheme.startswith("mysql"):
        return "mysql"
    if scheme.startswith("sqlite"):
        return "sqlite"
    return "postgresql"


class _LRUClientCache:
    """Thread-unsafe LRU cache of AsyncDatabaseClient (single event loop — safe)."""

    def __init__(self, maxsize: int) -> None:
        self._cache: OrderedDict[str, AsyncDatabaseClient] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> AsyncDatabaseClient | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: AsyncDatabaseClient) -> AsyncDatabaseClient | None:
        """Insert/update; return the evicted client if the cache was full."""
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
        clients = list(self._cache.values())
        self._cache.clear()
        return clients


@dataclass
class ConnectionInfo:
    """Public view of a connection — never carries the DSN, only a masked preview."""

    connection_id: str
    name: str
    db_type: str
    is_default: bool
    has_dsn: bool
    url_preview: str | None
    created_at: datetime
    updated_at: datetime


class ConnectionService:
    """Manages a user's database connections and their live clients."""

    def __init__(self, database_url: str, secret_key: str) -> None:
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
        logger.info("ConnectionService initialized")

    async def dispose(self) -> None:
        for client in self._client_cache.drain():
            await client.dispose()
        if self._engine:
            await self._engine.dispose()

    # ── Encryption ──────────────────────────────────────────────────────────────

    async def _encrypt(self, value: str) -> str:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._fernet.encrypt(value.encode()).decode()
        )

    async def _decrypt(self, value: str) -> str:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._fernet.decrypt(value.encode()).decode()
        )

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _require_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("ConnectionService.initialize() must be called before use")
        return self._session_factory

    @staticmethod
    def _to_info(row: UserDatabaseConnection) -> ConnectionInfo:
        return ConnectionInfo(
            connection_id=row.connection_id,
            name=row.name,
            db_type=row.db_type,
            is_default=bool(row.is_default),
            has_dsn=row.encrypted_url is not None,
            url_preview=None,  # filled in by list/get/create when a DSN is present
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def _get_owned(
        self, db: AsyncSession, user_id: str, connection_id: str
    ) -> UserDatabaseConnection:
        """Fetch a connection the user owns, or raise ConnectionNotFoundError (404)."""
        row = (
            await db.execute(
                select(UserDatabaseConnection).where(
                    UserDatabaseConnection.connection_id == connection_id,
                    UserDatabaseConnection.user_id == user_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise ConnectionNotFoundError("Connection not found.")
        return row

    async def assert_owned(self, user_id: str, connection_id: str) -> None:
        """Raise ``ConnectionNotFoundError`` unless ``user_id`` owns ``connection_id``.

        Public, read-only ownership check for other services (e.g.
        ``ScheduledQueryService``, ``MetricsService``) that scope their own
        records to a connection but must never confirm existence of a
        connection owned by someone else.
        """
        factory = self._require_factory()
        async with factory() as db:
            await self._get_owned(db, user_id, connection_id)

    def _normalise_and_validate(self, raw_url: str) -> str:
        normalised = to_async_database_url(raw_url.strip())
        try:
            _validate_db_url(normalised)
        except ValueError as exc:
            raise ConnectionValidationError(str(exc)) from exc
        return normalised

    @staticmethod
    async def _test_client(normalised_url: str) -> None:
        """Open a throwaway client and run SELECT 1; raise ConnectionTestError on failure."""
        from sqlalchemy import text

        tmp = AsyncDatabaseClient(database_url=normalised_url)
        try:
            async with tmp.session() as sess:
                await sess.execute(text("SELECT 1"))
        except Exception as exc:
            logger.warning("Connection test failed", error=str(exc))
            raise ConnectionTestError(
                "Could not connect to the database. Check the host and credentials.",
                detail=str(exc),
            ) from exc
        finally:
            await tmp.dispose()

    # ── Ensure a default exists ──────────────────────────────────────────────────

    async def ensure_default_connection(self, user_id: str) -> str:
        """Guarantee the user has ≥1 connection and exactly one default.

        Also self-heals legacy rows missing a ``connection_id``. Returns the
        active (default) connection's ``connection_id``.
        """
        factory = self._require_factory()
        async with factory() as db:
            rows = (
                await db.execute(
                    select(UserDatabaseConnection)
                    .where(UserDatabaseConnection.user_id == user_id)
                    .order_by(UserDatabaseConnection.id)
                )
            ).scalars().all()

            # Backfill any legacy row missing a connection_id (dev/backstop DBs).
            for r in rows:
                if not r.connection_id:
                    r.connection_id = str(uuid4())

            if not rows:
                row = UserDatabaseConnection(
                    connection_id=str(uuid4()),
                    user_id=user_id,
                    name="Server Default",
                    db_type="postgresql",
                    encrypted_url=None,
                    is_default=True,
                )
                db.add(row)
                await db.commit()
                logger.info("Server Default connection created", user_id=user_id)
                return row.connection_id

            defaults = [r for r in rows if r.is_default]
            if not defaults:
                rows[0].is_default = True
                active = rows[0].connection_id
            else:
                active = defaults[0].connection_id
                # Collapse any accidental multi-default state.
                for extra in defaults[1:]:
                    extra.is_default = False
            await db.commit()
            return active

    # ── CRUD ─────────────────────────────────────────────────────────────────────

    async def list_connections(self, user_id: str) -> list[ConnectionInfo]:
        await self.ensure_default_connection(user_id)
        factory = self._require_factory()
        async with factory() as db:
            rows = (
                await db.execute(
                    select(UserDatabaseConnection)
                    .where(UserDatabaseConnection.user_id == user_id)
                    .order_by(UserDatabaseConnection.created_at)
                )
            ).scalars().all()
        infos: list[ConnectionInfo] = []
        for r in rows:
            info = self._to_info(r)
            if r.encrypted_url is not None:
                try:
                    info.url_preview = _mask_url(await self._decrypt(r.encrypted_url))
                except Exception:
                    info.url_preview = None
            infos.append(info)
        return infos

    async def create(
        self, user_id: str, name: str, raw_url: str, db_type: str | None = None
    ) -> ConnectionInfo:
        """Validate + test + encrypt + store a new connection (becomes default if first)."""
        name = name.strip()
        if not name:
            raise ConnectionValidationError("Connection name cannot be empty.")
        normalised = self._normalise_and_validate(raw_url)
        await self._test_client(normalised)
        encrypted = await self._encrypt(normalised)
        resolved_type = db_type or _derive_db_type(normalised)
        now = datetime.utcnow()

        # Make sure the user has a baseline default before we decide is_default.
        await self.ensure_default_connection(user_id)
        factory = self._require_factory()
        async with factory() as db:
            count = (
                await db.execute(
                    select(UserDatabaseConnection).where(
                        UserDatabaseConnection.user_id == user_id
                    )
                )
            ).scalars().all()
            if len(count) >= _MAX_CONNECTIONS_PER_USER:
                raise ConnectionValidationError(
                    f"Connection limit reached (max {_MAX_CONNECTIONS_PER_USER})."
                )
            # The first *personal* (DSN-bearing) connection becomes active, so a
            # user who only had the built-in Server Default immediately uses their
            # own database. Subsequent connections do not steal the active slot.
            make_default = not any(c.encrypted_url is not None for c in count)
            if make_default:
                await db.execute(
                    update(UserDatabaseConnection)
                    .where(UserDatabaseConnection.user_id == user_id)
                    .values(is_default=False)
                )
            row = UserDatabaseConnection(
                connection_id=str(uuid4()),
                user_id=user_id,
                name=name,
                db_type=resolved_type,
                encrypted_url=encrypted,
                is_default=make_default,
                created_at=now,
                updated_at=now,
            )
            db.add(row)
            try:
                await db.commit()
            except IntegrityError as exc:
                await db.rollback()
                raise ConnectionValidationError(
                    f"A connection named '{name}' already exists."
                ) from exc
            await db.refresh(row)
        logger.info(
            "Connection created", user_id=user_id, connection_id=row.connection_id, db_type=resolved_type
        )
        info = self._to_info(row)
        info.url_preview = _mask_url(normalised)
        return info

    async def update(
        self,
        user_id: str,
        connection_id: str,
        name: str | None = None,
        raw_url: str | None = None,
    ) -> ConnectionInfo:
        """Rename and/or replace the DSN of a connection (ownership enforced)."""
        normalised: str | None = None
        if raw_url is not None:
            normalised = self._normalise_and_validate(raw_url)
            await self._test_client(normalised)

        factory = self._require_factory()
        async with factory() as db:
            row = await self._get_owned(db, user_id, connection_id)
            if name is not None:
                cleaned = name.strip()
                if not cleaned:
                    raise ConnectionValidationError("Connection name cannot be empty.")
                row.name = cleaned
            if normalised is not None:
                row.encrypted_url = await self._encrypt(normalised)
                row.db_type = _derive_db_type(normalised)
            row.updated_at = datetime.utcnow()
            try:
                await db.commit()
            except IntegrityError as exc:
                await db.rollback()
                raise ConnectionValidationError(
                    f"A connection named '{name}' already exists."
                ) from exc
            await db.refresh(row)
            info = self._to_info(row)
            if row.encrypted_url is not None:
                info.url_preview = _mask_url(await self._decrypt(row.encrypted_url))

        # A changed DSN invalidates the cached client.
        if normalised is not None:
            await self._evict(connection_id)
        logger.info("Connection updated", user_id=user_id, connection_id=connection_id)
        return info

    async def delete(self, user_id: str, connection_id: str) -> None:
        """Delete a connection, dispose its client, and re-point the default if needed."""
        factory = self._require_factory()
        async with factory() as db:
            row = await self._get_owned(db, user_id, connection_id)
            was_default = bool(row.is_default)
            await db.delete(row)
            await db.flush()
            if was_default:
                nxt = (
                    await db.execute(
                        select(UserDatabaseConnection)
                        .where(UserDatabaseConnection.user_id == user_id)
                        .order_by(UserDatabaseConnection.created_at)
                    )
                ).scalars().first()
                if nxt is not None:
                    nxt.is_default = True
            await db.commit()
        await self._evict(connection_id)
        logger.info("Connection deleted", user_id=user_id, connection_id=connection_id)

    async def set_default(self, user_id: str, connection_id: str) -> ConnectionInfo:
        """Make ``connection_id`` the user's active/default connection."""
        factory = self._require_factory()
        async with factory() as db:
            row = await self._get_owned(db, user_id, connection_id)
            await db.execute(
                update(UserDatabaseConnection)
                .where(UserDatabaseConnection.user_id == user_id)
                .values(is_default=False)
            )
            row.is_default = True
            await db.commit()
            await db.refresh(row)
            info = self._to_info(row)
            if row.encrypted_url is not None:
                info.url_preview = _mask_url(await self._decrypt(row.encrypted_url))
        logger.info("Connection selected", user_id=user_id, connection_id=connection_id)
        return info

    async def test(self, user_id: str, connection_id: str) -> None:
        """Test connectivity of a stored connection (ownership enforced)."""
        factory = self._require_factory()
        async with factory() as db:
            row = await self._get_owned(db, user_id, connection_id)
            encrypted = row.encrypted_url
        if encrypted is None:
            # Server Default — reachability is the platform's responsibility.
            return
        normalised = await self._decrypt(encrypted)
        await self._test_client(normalised)

    # ── Active-connection resolution ──────────────────────────────────────────────

    async def get_active_connection_id(self, user_id: str) -> str:
        """Return the user's active (default) connection id, ensuring one exists."""
        return await self.ensure_default_connection(user_id)

    # ── Legacy /profile/database compatibility helpers ────────────────────────────

    async def get_default_info(self, user_id: str) -> ConnectionInfo:
        """Return the active/default connection (ensuring one exists)."""
        active = await self.ensure_default_connection(user_id)
        factory = self._require_factory()
        async with factory() as db:
            row = (
                await db.execute(
                    select(UserDatabaseConnection).where(
                        UserDatabaseConnection.connection_id == active
                    )
                )
            ).scalar_one()
            info = self._to_info(row)
            if row.encrypted_url is not None:
                info.url_preview = _mask_url(await self._decrypt(row.encrypted_url))
        return info

    async def upsert_default_dsn(self, user_id: str, raw_url: str) -> ConnectionInfo:
        """Set the DSN on the active/default connection (legacy single-DB API)."""
        active = await self.ensure_default_connection(user_id)
        return await self.update(user_id, active, raw_url=raw_url)

    async def clear_default_dsn(self, user_id: str) -> None:
        """Clear the active connection's DSN, reverting it to the Server Default."""
        active = await self.ensure_default_connection(user_id)
        factory = self._require_factory()
        async with factory() as db:
            row = await self._get_owned(db, user_id, active)
            row.encrypted_url = None
            row.updated_at = datetime.utcnow()
            await db.commit()
        await self._evict(active)
        logger.info("Connection DSN cleared (reverted to server default)", user_id=user_id)

    async def get_client(self, connection_id: str) -> AsyncDatabaseClient | None:
        """Return a live client for a connection, or ``None`` for Server Default.

        ``None`` signals the caller to fall back to the platform database client.
        """
        cached = self._client_cache.get(connection_id)
        if cached is not None:
            return cached
        factory = self._require_factory()
        async with factory() as db:
            row = (
                await db.execute(
                    select(UserDatabaseConnection.encrypted_url).where(
                        UserDatabaseConnection.connection_id == connection_id
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            return None  # unknown → caller uses server default
        encrypted = row
        if encrypted is None:
            return None  # Server Default connection
        try:
            raw_url = await self._decrypt(encrypted)
        except Exception as exc:
            logger.warning("Failed to decrypt connection DSN", connection_id=connection_id, error=str(exc))
            return None
        client = AsyncDatabaseClient(database_url=raw_url)
        evicted = self._client_cache.put(connection_id, client)
        if evicted is not None:
            await evicted.dispose()
        return client

    async def _evict(self, connection_id: str) -> None:
        client = self._client_cache.pop(connection_id)
        if client:
            await client.dispose()
