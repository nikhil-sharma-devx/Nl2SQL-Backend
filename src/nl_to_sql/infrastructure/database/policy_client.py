"""PolicyEnforcingClient — applies a connection's AccessPolicy to every query.

Decorator around :class:`AsyncDatabaseClient`. It is returned wherever a
connection's client is resolved (``ConnectionService.get_scoped_client``), so
every execution path — interactive queries, ``/query/execute``, previews,
saved-query runs, dashboards, scheduled queries — inherits the policy without
any per-route check that could be forgotten.

For each SQL-running method it (1) rejects statements that touch a disallowed
table or restricted column, raising ``AccessPolicyViolationError`` (a
``DatabaseExecutionError``, so the orchestrator's self-correction loop retries
with the reason), and (2) redacts restricted columns in returned rows.
Internal, non-SQL-text methods (sessions, reflection, locks) are delegated.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from nl_to_sql.core.access_policy import AccessPolicy
from nl_to_sql.core.exceptions import AccessPolicyViolationError
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient

logger = structlog.get_logger(__name__)


class PolicyEnforcingClient(AsyncDatabaseClient):
    """An AsyncDatabaseClient that enforces an AccessPolicy.

    Subclasses (rather than duck-types) the real client so existing type
    annotations and ``isinstance`` checks keep working. It deliberately does
    NOT call ``super().__init__`` — it owns no engine; every public method is
    explicitly delegated to ``inner``.
    """

    def __init__(self, inner: AsyncDatabaseClient, policy: AccessPolicy, dialect: str) -> None:
        self._inner = inner
        self.policy = policy
        self._policy_dialect = dialect

    @property
    def inner(self) -> AsyncDatabaseClient:
        return self._inner

    def _enforce(self, sql: str) -> set[str]:
        problems = self.policy.violations(sql, self._policy_dialect)
        if problems:
            logger.warning("Access policy blocked a query", reasons=problems)
            raise AccessPolicyViolationError(" ".join(problems), detail="; ".join(problems))
        return self.policy.referenced_tables(sql, self._policy_dialect)

    # ── SQL-running methods (enforced) ──────────────────────────────────────

    async def execute_sql(self, sql: str) -> list[dict[str, Any]]:
        tables = self._enforce(sql)
        rows = await self._inner.execute_sql(sql)
        return self.policy.redact_rows(rows, tables)

    async def explain(self, sql: str, *, analyze: bool = False) -> dict[str, Any]:
        self._enforce(sql)
        return await self._inner.explain(sql, analyze=analyze)

    async def execute_sql_stream(
        self, sql: str, batch_size: int = 100
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        tables = self._enforce(sql)
        async for batch in self._inner.execute_sql_stream(sql, batch_size=batch_size):
            yield self.policy.redact_rows(batch, tables)

    # ── Delegated (no user SQL text) ────────────────────────────────────────

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self._inner.session() as sess:
            yield sess

    @asynccontextmanager
    async def advisory_xact_lock(self, name: str) -> AsyncGenerator[bool, None]:
        async with self._inner.advisory_xact_lock(name) as acquired:
            yield acquired

    async def health_check(self) -> bool:
        return await self._inner.health_check()

    async def reflect_schema(
        self, schema_name: str = "public", *, schemas: list[str] | None = None
    ) -> dict[str, Any]:
        return await self._inner.reflect_schema(schema_name=schema_name, schemas=schemas)

    async def list_schemas(self) -> list[dict[str, Any]]:
        # Delegate explicitly: this wrapper never runs the base __init__, so the
        # inherited implementation would miss the inner client's state.
        return await self._inner.list_schemas()

    async def switch_engine(
        self,
        new_url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 300,
    ) -> None:
        await self._inner.switch_engine(
            new_url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
        )

    async def dispose(self) -> None:
        # The inner client is owned (and disposed) by ConnectionService's cache
        # or the container; disposing it here would kill a shared pool.
        return None
