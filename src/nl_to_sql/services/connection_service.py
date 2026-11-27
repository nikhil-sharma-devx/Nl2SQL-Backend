"""ConnectionService — per-user *multiple* database connections (BYOD).

Owns everything about a user's database connections:

  - CRUD (create / rename+edit / delete / list),
  - encryption of the DSN at rest (reuses ``infrastructure.crypto``'s
    versioned-Fernet helper — no second crypto mechanism),
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
from dataclasses import dataclass, field
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

from nl_to_sql.core.access_policy import AccessPolicy
from nl_to_sql.core.exceptions import (
    ConnectionNotFoundError,
    ConnectionTestError,
    ConnectionValidationError,
)
from nl_to_sql.infrastructure.crypto import CURRENT_KDF_VERSION, VersionedFernet
from nl_to_sql.infrastructure.database.models import (
    Base,
    User,
    UserDatabaseConnection,
    Workspace,
    WorkspaceMember,
)
from nl_to_sql.infrastructure.database.policy_client import PolicyEnforcingClient
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.infrastructure.database.url_utils import asyncpg_connect_args, to_async_database_url
from nl_to_sql.services.api_key_service import _to_async_url

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
    """Raise ConnectionValidationError if the scheme is disallowed or the host is a private IP (SSRF)."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ConnectionValidationError(f"Database URL scheme '{scheme}' is not allowed.")
    # sqlite is file-based — no host to validate (must be checked before the
    # host requirement below, which only applies to networked databases).
    if scheme == "sqlite+aiosqlite":
        return
    host = parsed.hostname
    if not host:
        raise ConnectionValidationError("Database URL must include a host.")
    try:
        results = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ConnectionValidationError(f"Cannot resolve database host '{host}': {exc}") from exc
    for *_, sockaddr in results:
        addr = ipaddress.ip_address(sockaddr[0])
        if any(addr in net for net in _PRIVATE_NETS):
            raise ConnectionValidationError(
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


_ClientCacheKey = tuple[str, str]  # (user_id, connection_id)


class _LRUClientCache:
    """Thread-unsafe LRU cache of AsyncDatabaseClient (single event loop — safe).

    Keyed by ``(user_id, connection_id)`` rather than ``connection_id`` alone
    so a cache *hit* can never return another user's client — ownership is
    encoded in the key itself instead of needing a check on every lookup.
    """

    def __init__(self, maxsize: int) -> None:
        self._cache: OrderedDict[_ClientCacheKey, AsyncDatabaseClient] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: _ClientCacheKey) -> AsyncDatabaseClient | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(
        self, key: _ClientCacheKey, value: AsyncDatabaseClient
    ) -> AsyncDatabaseClient | None:
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

    def pop(self, key: _ClientCacheKey) -> AsyncDatabaseClient | None:
        return self._cache.pop(key, None)

    def pop_connection(self, connection_id: str) -> list[AsyncDatabaseClient]:
        """Remove and return every cached client for ``connection_id``, any user."""
        matches = [k for k in self._cache if k[1] == connection_id]
        return [self._cache.pop(k) for k in matches]

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
    # Sharing / access (defaults describe a private, owned connection).
    access: str = "owner"  # "owner" | "admin" | "editor" | "member"
    workspace_id: str | None = None
    workspace_name: str | None = None
    owner_email: str | None = None
    allowed_tables: list[str] | None = None
    restricted_columns: list[str] = field(default_factory=list)


class ConnectionService:
    """Manages a user's database connections and their live clients."""

    def __init__(self, database_url: str, secret_key: str) -> None:
        self._database_url = _to_async_url(database_url)
        self._crypto = VersionedFernet(secret_key)
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
            connect_args=asyncpg_connect_args(self._database_url),
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
        return await self._crypto.encrypt(value)

    async def _decrypt(self, value: str, kdf_version: int = 1) -> str:
        return await self._crypto.decrypt(value, kdf_version)

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
            workspace_id=row.workspace_id,
            allowed_tables=list(row.allowed_tables) if row.allowed_tables is not None else None,
            restricted_columns=list(row.restricted_columns or []),
        )

    @staticmethod
    async def _access(
        db: AsyncSession, user_id: str, connection_id: str
    ) -> tuple[UserDatabaseConnection, str] | None:
        """Return (row, access) if ``user_id`` may USE the connection, else None.

        ``access`` is ``"owner"`` for the owner, otherwise the caller's role in
        the workspace the connection is shared into. This is the only place
        shared access is decided; every read path goes through it.
        """
        row = (
            await db.execute(
                select(UserDatabaseConnection).where(
                    UserDatabaseConnection.connection_id == connection_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        if row.user_id == user_id:
            return row, "owner"
        if row.workspace_id is None or row.encrypted_url is None:
            return None
        role = (
            await db.execute(
                select(WorkspaceMember.role).where(
                    WorkspaceMember.workspace_id == row.workspace_id,
                    WorkspaceMember.user_id == user_id,
                )
            )
        ).scalar_one_or_none()
        return (row, str(role)) if role else None

    async def assert_accessible(self, user_id: str, connection_id: str) -> str:
        """Raise ``ConnectionNotFoundError`` unless the user may use the connection.

        Returns the access level ("owner" or a workspace role).
        """
        factory = self._require_factory()
        async with factory() as db:
            found = await self._access(db, user_id, connection_id)
        if found is None:
            raise ConnectionNotFoundError("Connection not found.")
        return found[1]

    async def get_workspace_id(self, user_id: str, connection_id: str) -> str | None:
        """Workspace the connection is shared into, if the user can access it."""
        factory = self._require_factory()
        async with factory() as db:
            found = await self._access(db, user_id, connection_id)
        return found[0].workspace_id if found else None

    async def _get_owned(
        self,
        db: AsyncSession,
        user_id: str,
        connection_id: str,
    ) -> UserDatabaseConnection:
        """Fetch a connection the user owns, or raise ConnectionNotFoundError (404)."""
        conditions = [
            UserDatabaseConnection.connection_id == connection_id,
            UserDatabaseConnection.user_id == user_id,
        ]
        row = (
            await db.execute(select(UserDatabaseConnection).where(*conditions))
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
        _validate_db_url(normalised)
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
                (
                    await db.execute(
                        select(UserDatabaseConnection)
                        .where(UserDatabaseConnection.user_id == user_id)
                        .order_by(UserDatabaseConnection.id)
                    )
                )
                .scalars()
                .all()
            )

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
        conditions = [UserDatabaseConnection.user_id == user_id]
        async with factory() as db:
            rows = (
                (
                    await db.execute(
                        select(UserDatabaseConnection)
                        .where(*conditions)
                        .order_by(UserDatabaseConnection.created_at)
                    )
                )
                .scalars()
                .all()
            )
        infos: list[ConnectionInfo] = []
        for r in rows:
            info = self._to_info(r)
            if r.encrypted_url is not None:
                try:
                    info.url_preview = _mask_url(await self._decrypt(r.encrypted_url, r.kdf_version))
                except Exception:
                    info.url_preview = None
            infos.append(info)

        # Connections other members shared into workspaces this user belongs to.
        # Never carry a DSN preview — the host is the owner's business.
        async with factory() as db:
            active_shared = (
                await db.execute(select(User.active_shared_connection_id).where(User.id == user_id))
            ).scalar_one_or_none()
            shared_rows = (
                await db.execute(
                    select(UserDatabaseConnection, WorkspaceMember.role, Workspace.name, User.email)
                    .join(Workspace, Workspace.id == UserDatabaseConnection.workspace_id)
                    .join(
                        WorkspaceMember,
                        (WorkspaceMember.workspace_id == UserDatabaseConnection.workspace_id)
                        & (WorkspaceMember.user_id == user_id),
                    )
                    .join(User, User.id == UserDatabaseConnection.user_id)
                    .where(
                        UserDatabaseConnection.user_id != user_id,
                        UserDatabaseConnection.encrypted_url.is_not(None),
                    )
                    .order_by(UserDatabaseConnection.name)
                )
            ).all()
        for r, role, ws_name, owner_email in shared_rows:
            info = self._to_info(r)
            info.access = str(role)
            info.workspace_name = ws_name
            info.owner_email = owner_email
            info.is_default = r.connection_id == active_shared
            infos.append(info)
        if active_shared and any(i.is_default and i.access != "owner" for i in infos):
            for i in infos:
                if i.access == "owner":
                    i.is_default = False
        # Owners also see which workspace each of their connections is in.
        owned_ws = {i.workspace_id for i in infos if i.access == "owner" and i.workspace_id}
        if owned_ws:
            async with factory() as db:
                name_rows = (
                    await db.execute(select(Workspace.id, Workspace.name).where(Workspace.id.in_(owned_ws)))
                ).all()
            ws_names: dict[str, str] = {str(ws_id): str(name) for ws_id, name in name_rows}
            for i in infos:
                if i.access == "owner" and i.workspace_id:
                    i.workspace_name = ws_names.get(i.workspace_id)
        return infos

    async def create(
        self,
        user_id: str,
        name: str,
        raw_url: str,
        db_type: str | None = None,
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
                (
                    await db.execute(
                        select(UserDatabaseConnection).where(
                            UserDatabaseConnection.user_id == user_id
                        )
                    )
                )
                .scalars()
                .all()
            )
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
                kdf_version=CURRENT_KDF_VERSION,
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
            "Connection created",
            user_id=user_id,
            connection_id=row.connection_id,
            db_type=resolved_type,
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
                row.kdf_version = CURRENT_KDF_VERSION
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
                info.url_preview = _mask_url(await self._decrypt(row.encrypted_url, row.kdf_version))

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
                    (
                        await db.execute(
                            select(UserDatabaseConnection)
                            .where(UserDatabaseConnection.user_id == user_id)
                            .order_by(UserDatabaseConnection.created_at)
                        )
                    )
                    .scalars()
                    .first()
                )
                if nxt is not None:
                    nxt.is_default = True
            await db.commit()
        await self._evict(connection_id)
        logger.info("Connection deleted", user_id=user_id, connection_id=connection_id)

    async def set_default(self, user_id: str, connection_id: str) -> ConnectionInfo:
        """Make ``connection_id`` the user's active connection.

        For an owned connection this moves the ``is_default`` flag (and clears
        any shared pointer). For a connection shared with the user through a
        workspace it sets ``users.active_shared_connection_id`` instead — a
        member can never flag another user's row.
        """
        factory = self._require_factory()
        async with factory() as db:
            found = await self._access(db, user_id, connection_id)
            if found is None:
                raise ConnectionNotFoundError("Connection not found.")
            row, access = found
            if access != "owner":
                await db.execute(
                    update(User).where(User.id == user_id).values(active_shared_connection_id=connection_id)
                )
                await db.commit()
                info = self._to_info(row)
                info.access = access
                info.is_default = True
                logger.info("Shared connection selected", user_id=user_id, connection_id=connection_id)
                return info
            await db.execute(
                update(User).where(User.id == user_id).values(active_shared_connection_id=None)
            )
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
                info.url_preview = _mask_url(await self._decrypt(row.encrypted_url, row.kdf_version))
        logger.info("Connection selected", user_id=user_id, connection_id=connection_id)
        return info

    async def test(self, user_id: str, connection_id: str) -> None:
        """Test connectivity of a stored connection (ownership enforced)."""
        factory = self._require_factory()
        async with factory() as db:
            row = await self._get_owned(db, user_id, connection_id)
            encrypted = row.encrypted_url
            kdf_version = row.kdf_version
        if encrypted is None:
            # Server Default — reachability is the platform's responsibility.
            return
        normalised = await self._decrypt(encrypted, kdf_version)
        await self._test_client(normalised)

    # ── Active-connection resolution ──────────────────────────────────────────────

    async def get_active_connection_id(self, user_id: str) -> str:
        """Return the user's active connection id, ensuring one exists.

        A selected shared connection wins while the user can still access it;
        if the share or membership was revoked, the pointer is cleared and the
        user falls back to their own default.
        """
        factory = self._require_factory()
        async with factory() as db:
            shared = (
                await db.execute(select(User.active_shared_connection_id).where(User.id == user_id))
            ).scalar_one_or_none()
            if shared:
                if await self._access(db, user_id, shared) is not None:
                    return shared
                await db.execute(
                    update(User).where(User.id == user_id).values(active_shared_connection_id=None)
                )
                await db.commit()
                logger.info("Stale shared-connection pointer cleared", user_id=user_id)
        return await self.ensure_default_connection(user_id)

    # ── Access policy ───────────────────────────────────────────────────────────

    async def get_policy(self, user_id: str, connection_id: str) -> tuple[AccessPolicy, str]:
        """Return (policy, db_type) for a connection the user may use."""
        factory = self._require_factory()
        async with factory() as db:
            found = await self._access(db, user_id, connection_id)
        if found is None:
            return AccessPolicy(), "postgresql"
        row = found[0]
        return (
            AccessPolicy.from_config(row.allowed_tables, row.restricted_columns),
            row.db_type or "postgresql",
        )

    async def update_policy(
        self,
        user_id: str,
        connection_id: str,
        allowed_tables: list[str] | None,
        restricted_columns: list[str],
    ) -> ConnectionInfo:
        """Set the table allow-list / restricted columns (owner only)."""
        cleaned_tables = (
            sorted({t.strip() for t in allowed_tables if t and t.strip()})
            if allowed_tables is not None
            else None
        )
        cleaned_cols = sorted({c.strip() for c in restricted_columns if c and c.strip()})
        for col in cleaned_cols:
            table, _, column = col.rpartition(".")
            if not table or not column:
                raise ConnectionValidationError(
                    f"Restricted column '{col}' must be written as table.column."
                )
        if cleaned_tables is not None and not cleaned_tables:
            raise ConnectionValidationError(
                "Allow at least one table, or clear the allow-list to allow every table."
            )
        factory = self._require_factory()
        async with factory() as db:
            row = await self._get_owned(db, user_id, connection_id)
            row.allowed_tables = cleaned_tables
            row.restricted_columns = cleaned_cols
            row.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(row)
            info = self._to_info(row)
        logger.info(
            "Connection access policy updated",
            connection_id=connection_id,
            allowed_tables=None if cleaned_tables is None else len(cleaned_tables),
            restricted_columns=len(cleaned_cols),
        )
        return info

    async def get_scoped_client(
        self, user_id: str, connection_id: str, fallback: AsyncDatabaseClient
    ) -> AsyncDatabaseClient:
        """Resolve the client for a connection AND apply its access policy.

        This is the entry point every execution path should use: it falls back
        to the platform client for the Server Default (exactly like
        ``get_client``), then wraps whichever client it got in a
        ``PolicyEnforcingClient`` when the connection has a non-empty policy.
        """
        client = await self.get_client(user_id, connection_id) or fallback
        policy, db_type = await self.get_policy(user_id, connection_id)
        if policy.is_empty:
            return client
        return PolicyEnforcingClient(client, policy, db_type)

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
                info.url_preview = _mask_url(await self._decrypt(row.encrypted_url, row.kdf_version))
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

    async def get_client(self, user_id: str, connection_id: str) -> AsyncDatabaseClient | None:
        """Return a live client for ``user_id``'s connection, or ``None`` for Server Default.

        ``None`` signals the caller to fall back to the platform database
        client — for an unknown connection, a Server Default (no DSN), *or*
        a ``connection_id`` not owned by ``user_id``. Ownership is enforced
        by scoping the lookup query, and the cache key includes ``user_id``
        so a cache hit can never return another user's client.
        """
        # Access is re-checked on EVERY call, before the client cache: a cached
        # client must never outlive a revoked share or a removed membership.
        factory = self._require_factory()
        async with factory() as db:
            found = await self._access(db, user_id, connection_id)
        if found is None:
            return None  # unknown, or not accessible by user_id → caller uses server default
        cache_key = (user_id, connection_id)
        cached = self._client_cache.get(cache_key)
        if cached is not None:
            return cached
        encrypted, kdf_version = found[0].encrypted_url, found[0].kdf_version
        if encrypted is None:
            return None  # Server Default connection
        try:
            raw_url = await self._decrypt(encrypted, kdf_version)
        except Exception as exc:
            logger.warning(
                "Failed to decrypt connection DSN", connection_id=connection_id, error=str(exc)
            )
            return None
        client = AsyncDatabaseClient(database_url=raw_url)
        evicted = self._client_cache.put(cache_key, client)
        if evicted is not None:
            await evicted.dispose()
        return client

    async def _evict(self, connection_id: str) -> None:
        """Drop any cached client for ``connection_id``, regardless of the owning user."""
        for client in self._client_cache.pop_connection(connection_id):
            await client.dispose()
