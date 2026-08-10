"""MetricsService — governed business metrics, scoped to a DB connection.

Owns the ``metrics`` table (Semantic Layer / Metrics Catalog feature). Reads
are scoped to one ``connection_id`` (any user with access to a connection can
see its certified metrics — metrics are global-to-connection, the same
posture the glossary feature takes for global-to-user); writes are further
scoped to ``user_id`` (only the creator may update/delete/certify their own
metric — this app has no roles system yet, see ``certify``'s docstring).

The session factory is constructor-injected (mirrors ``DashboardService``).
``SQLColumnValidator`` (already a container singleton — reused, not
re-instantiated) and ``SchemaCatalogService`` are also injected, used by
``validate_sql_definition`` to catch hallucinated columns in a metric's SQL
before it can be certified.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.core.exceptions import MetricNotFoundError, MetricValidationError
from nl_to_sql.infrastructure.database.models import Metric
from nl_to_sql.services.schema_catalog_service import SchemaCatalogService
from nl_to_sql.services.sql_column_validator import SQLColumnValidator
from nl_to_sql.services.sql_validator import SQLValidatorService

logger = structlog.get_logger(__name__)

_MAX_METRICS_PER_CONNECTION = 200


@dataclass
class MetricInfo:
    """Public view of a metric."""

    metric_id: str
    connection_id: str
    name: str
    description: str | None
    sql_definition: str
    dimensions: list[str]
    tags: list[str]
    owner: str | None
    certified: bool
    is_builtin: bool
    validation_errors: list[str]
    created_at: datetime
    updated_at: datetime


class MetricsService:
    """Connection-scoped metrics CRUD + SQL validation + certify/uncertify."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        column_validator: SQLColumnValidator,
        schema_catalog_service: SchemaCatalogService,
        sql_validator: SQLValidatorService,
    ) -> None:
        self._session_factory = session_factory
        self._column_validator = column_validator
        self._schema_catalog_service = schema_catalog_service
        self._sql_validator = sql_validator
        self._log = logger.bind(service="Metrics")

    def _enforce_sql_safety(self, sql: str) -> None:
        """Hard-block dangerous/non-SELECT SQL at write time (C6).

        Unlike :meth:`validate_sql_definition` (a soft, hallucinated-column
        check only enforced at certify time), this runs
        :class:`SQLValidatorService` — the same dangerous-function/SELECT-only
        guard the live query path uses — and blocks the write outright. It
        must run at write time because ``preview_metric`` lets any user on
        the connection preview/execute a metric's SQL even before it's
        certified.
        """
        result = self._sql_validator.validate(sql)
        if not result.is_valid:
            raise MetricValidationError(
                "Metric SQL definition rejected by validation.",
                detail="; ".join(result.errors),
            )

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _get_in_connection(
        self, db: AsyncSession, connection_id: str, metric_id: str
    ) -> Metric:
        """Fetch a metric within a connection (read scope — any user with access)."""
        row = (
            await db.execute(
                select(Metric).where(
                    Metric.metric_id == metric_id, Metric.connection_id == connection_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise MetricNotFoundError("Metric not found.")
        return row

    async def _get_owned(
        self,
        db: AsyncSession,
        user_id: str,
        connection_id: str,
        metric_id: str,
    ) -> Metric:
        """Fetch a metric the user created, within the given connection (write scope)."""
        conditions = [
            Metric.metric_id == metric_id,
            Metric.connection_id == connection_id,
            Metric.user_id == user_id,
        ]
        row = (await db.execute(select(Metric).where(*conditions))).scalar_one_or_none()
        if row is None:
            raise MetricNotFoundError("Metric not found.")
        return row

    async def _to_info(
        self, row: Metric, schema_context: dict[str, list[str]] | None = None
    ) -> MetricInfo:
        """Build a :class:`MetricInfo`, reusing a pre-fetched ``schema_context``
        when the caller already has one (avoids re-fetching the catalog per
        row — see ``list_metrics``)."""
        if schema_context is None:
            schema_context = await self._schema_context(row.user_id, row.connection_id)
        errors = self._column_validator.validate(row.sql_definition, schema_context)
        return MetricInfo(
            metric_id=row.metric_id,
            connection_id=row.connection_id,
            name=row.name,
            description=row.description,
            sql_definition=row.sql_definition,
            dimensions=list(row.dimensions or []),
            tags=list(row.tags or []),
            owner=row.owner,
            certified=row.certified,
            is_builtin=row.is_builtin,
            validation_errors=errors,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    # ── SQL validation ───────────────────────────────────────────────────────────

    async def _schema_context(self, user_id: str, connection_id: str) -> dict[str, list[str]]:
        """Fetch the connection's catalog and shape it for column validation.

        The catalog is scoped by ``connection_id`` alone (``user_id`` only
        affects the ``pinned`` flag, unused here) — safe to fetch once and
        reuse across every metric in a connection (see ``list_metrics``).
        """
        try:
            catalog = await self._schema_catalog_service.get_catalog(user_id, connection_id)
        except Exception as exc:
            self._log.warning("metrics: schema catalog unavailable for validation", error=str(exc))
            return {}
        return {
            t["table_name"]: [c["name"] for c in (t.get("columns") or [])]
            for t in catalog.get("tables", [])
        }

    async def validate_sql_definition(
        self, user_id: str, connection_id: str, sql: str
    ) -> list[str]:
        """Static, injection-free validation: flag hallucinated table/column refs.

        Does **not** execute the SQL — that's the separate ``preview`` flow.
        Returns an empty list when the definition is clean.
        """
        schema_context = await self._schema_context(user_id, connection_id)
        return self._column_validator.validate(sql, schema_context)

    # ── CRUD ─────────────────────────────────────────────────────────────────────

    async def list_metrics(
        self,
        user_id: str,
        connection_id: str,
        *,
        search: str | None = None,
        tag: str | None = None,
        certified_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MetricInfo], int]:
        async with self._session_factory() as db:
            stmt = select(Metric).where(Metric.connection_id == connection_id)
            count_stmt = (
                select(func.count())
                .select_from(Metric)
                .where(Metric.connection_id == connection_id)
            )
            if search:
                like = f"%{search.strip()}%"
                stmt = stmt.where(Metric.name.ilike(like) | Metric.description.ilike(like))
                count_stmt = count_stmt.where(
                    Metric.name.ilike(like) | Metric.description.ilike(like)
                )
            if certified_only:
                stmt = stmt.where(Metric.certified.is_(True))
                count_stmt = count_stmt.where(Metric.certified.is_(True))

            total = (await db.execute(count_stmt)).scalar_one()
            rows = (
                (await db.execute(stmt.order_by(Metric.name).limit(limit).offset(offset)))
                .scalars()
                .all()
            )

        if tag:
            rows = [r for r in rows if tag in (r.tags or [])]

        # Medium: fetch the catalog once for the whole page instead of once
        # per row (~200 redundant identical queries near the metrics cap) —
        # every row shares the same connection_id, and the catalog is
        # connection-scoped, not row-scoped.
        schema_context = await self._schema_context(user_id, connection_id)
        infos = [await self._to_info(r, schema_context) for r in rows]
        return infos, int(total)

    async def get(self, user_id: str, connection_id: str, metric_id: str) -> MetricInfo:
        async with self._session_factory() as db:
            row = await self._get_in_connection(db, connection_id, metric_id)
        return await self._to_info(row)

    async def create(
        self,
        user_id: str,
        connection_id: str,
        name: str,
        description: str | None,
        sql_definition: str,
        dimensions: list[str] | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        is_builtin: bool = False,
    ) -> MetricInfo:
        name = name.strip()
        sql_definition = sql_definition.strip()
        if not name:
            raise MetricValidationError("Metric name cannot be empty.")
        if not sql_definition:
            raise MetricValidationError("Metric SQL definition cannot be empty.")
        self._enforce_sql_safety(sql_definition)

        async with self._session_factory() as db:
            count = (
                await db.execute(
                    select(func.count())
                    .select_from(Metric)
                    .where(Metric.connection_id == connection_id)
                )
            ).scalar_one()
            if count >= _MAX_METRICS_PER_CONNECTION:
                raise MetricValidationError(
                    f"Maximum of {_MAX_METRICS_PER_CONNECTION} metrics per connection reached."
                )

            row = Metric(
                user_id=user_id,
                connection_id=connection_id,
                name=name[:200],
                description=description,
                sql_definition=sql_definition,
                dimensions=list(dimensions or []),
                tags=list(tags or []),
                owner=owner,
                is_builtin=is_builtin,
            )
            db.add(row)
            try:
                await db.commit()
            except IntegrityError as exc:
                await db.rollback()
                raise MetricValidationError(
                    f"A metric named '{name}' already exists for this connection."
                ) from exc
            await db.refresh(row)

        self._log.info(
            "metric created", user_id=user_id, connection_id=connection_id, metric_id=row.metric_id
        )
        return await self._to_info(row)

    async def update(
        self,
        user_id: str,
        connection_id: str,
        metric_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        sql_definition: str | None = None,
        dimensions: list[str] | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
    ) -> MetricInfo:
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, connection_id, metric_id)

            if name is not None:
                name = name.strip()
                if not name:
                    raise MetricValidationError("Metric name cannot be empty.")
                row.name = name[:200]
            if description is not None:
                row.description = description
            if sql_definition is not None:
                sql_definition = sql_definition.strip()
                if not sql_definition:
                    raise MetricValidationError("Metric SQL definition cannot be empty.")
                self._enforce_sql_safety(sql_definition)
                row.sql_definition = sql_definition
            if dimensions is not None:
                row.dimensions = list(dimensions)
            if tags is not None:
                row.tags = list(tags)
            if owner is not None:
                row.owner = owner
            row.updated_at = datetime.utcnow()

            try:
                await db.commit()
            except IntegrityError as exc:
                await db.rollback()
                raise MetricValidationError(
                    f"A metric named '{row.name}' already exists for this connection."
                ) from exc
            await db.refresh(row)

        self._log.info("metric updated", user_id=user_id, metric_id=metric_id)
        return await self._to_info(row)

    async def delete(self, user_id: str, connection_id: str, metric_id: str) -> None:
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, connection_id, metric_id)
            await db.delete(row)
            await db.commit()
        self._log.info("metric deleted", user_id=user_id, metric_id=metric_id)

    # ── Certification ────────────────────────────────────────────────────────────

    async def certify(self, user_id: str, connection_id: str, metric_id: str) -> MetricInfo:
        """Mark a metric certified, so it's injected into the SQL-generation prompt.

        V1 permission model: only the metric's own creator may certify it —
        the same ownership check as every other mutation. This app has no
        roles system yet (a real semantic layer would want a distinct
        "steward" role, separate from "anyone who can create metrics") — a
        known governance gap, left as a follow-up.
        """
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, connection_id, metric_id)
            # Belt-and-braces: re-run the hard safety check even though create/update
            # already enforce it — covers metrics persisted before this check existed.
            self._enforce_sql_safety(row.sql_definition)
            errors = await self.validate_sql_definition(user_id, connection_id, row.sql_definition)
            if errors:
                raise MetricValidationError(
                    "Cannot certify a metric with unresolved SQL validation errors.",
                    detail="; ".join(errors),
                )
            row.certified = True
            row.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(row)
        self._log.info("metric certified", user_id=user_id, metric_id=metric_id)
        return await self._to_info(row)

    async def uncertify(self, user_id: str, connection_id: str, metric_id: str) -> MetricInfo:
        async with self._session_factory() as db:
            row = await self._get_owned(db, user_id, connection_id, metric_id)
            row.certified = False
            row.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(row)
        self._log.info("metric uncertified", user_id=user_id, metric_id=metric_id)
        return await self._to_info(row)

    # ── Prompt injection support ───────────────────────────────────────────────────

    async def list_certified_for_prompt(
        self, connection_id: str, *, limit: int = 30
    ) -> list[dict[str, Any]]:
        """Return certified metrics for one connection, for prompt injection.

        No ``user_id`` filter — certified metrics are global-to-connection
        (any user querying this connection should see the same governed
        definitions), mirroring the design note in this module's docstring.
        """
        async with self._session_factory() as db:
            rows = (
                (
                    await db.execute(
                        select(Metric)
                        .where(Metric.connection_id == connection_id, Metric.certified.is_(True))
                        .order_by(Metric.name)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return [
            {
                "name": r.name,
                "description": r.description,
                "sql_definition": r.sql_definition,
                "dimensions": list(r.dimensions or []),
            }
            for r in rows
        ]
