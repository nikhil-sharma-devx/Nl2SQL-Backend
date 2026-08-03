"""DashboardService — per-user dashboards + auto chart recommendation.

Owns the ``dashboards`` + ``dashboard_widgets`` tables (Auto Charting &
Dashboards feature). Every read/write is scoped by ``user_id``; a cross-user
lookup returns ``None`` so the route can translate that into a 404 (never
confirming that another user's dashboard exists).

The session factory is constructor-injected (mirrors ``SchemaCatalogService`` /
``APIKeyService``) — this service never instantiates infrastructure.

``recommend_chart`` is a deterministic, LLM-free heuristic that maps a SQL
result set to the best visualization. It is a pure ``@staticmethod`` so it can
be unit-tested in isolation and reused by the route's ``recommend-chart``
endpoint.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict
from uuid import uuid4

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from nl_to_sql.infrastructure.database.models import Dashboard, DashboardWidget

logger = structlog.get_logger(__name__)


# ── Chart recommendation typing ────────────────────────────────────────────────


class ChartRecommendation(TypedDict, total=False):
    """Result of :meth:`DashboardService.recommend_chart`."""

    chart_type: str
    x_axis: str | None
    y_axis: str | None
    reason: str


# Column name hints that indicate geographic data.
_GEO_NAMES = frozenset(
    {"country", "region", "state", "city", "province", "county", "postcode", "zip"}
)
_LAT_NAMES = frozenset({"lat", "latitude"})
_LON_NAMES = frozenset({"lon", "lng", "long", "longitude"})

# SQL/type-string fragments that indicate a numeric column.
_NUMERIC_TYPES = (
    "int",
    "float",
    "double",
    "real",
    "numeric",
    "decimal",
    "money",
    "serial",
    "number",
)
# Fragments that indicate a temporal column.
_TEMPORAL_TYPES = ("date", "time", "timestamp", "datetime", "year")

# Above this many distinct categories a bar chart stops being useful → table.
_MAX_BAR_CARDINALITY = 50


def _type_str(col: dict[str, Any]) -> str:
    return str(col.get("type") or col.get("data_type") or "").lower()


def _col_name(col: dict[str, Any]) -> str:
    return str(col.get("name") or col.get("column") or "")


def _sample(rows: list[dict[str, Any]], name: str) -> Any:
    for row in rows:
        val = row.get(name)
        if val is not None:
            return val
    return None


def _is_numeric(col: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    type_str = _type_str(col)
    if any(frag in type_str for frag in _NUMERIC_TYPES):
        # bool columns are stored as e.g. "boolean" — excluded by not matching above
        return True
    # Fall back to sniffing a sample value when the type is unknown/blank.
    if not type_str:
        val = _sample(rows, _col_name(col))
        if isinstance(val, bool):
            return False
        if isinstance(val, (int, float)):
            return True
    return False


def _is_temporal(col: dict[str, Any]) -> bool:
    type_str = _type_str(col)
    if any(frag in type_str for frag in _TEMPORAL_TYPES):
        return True
    name = _col_name(col).lower()
    return name in {"date", "day", "month", "year", "week", "timestamp"} or name.endswith(
        ("_date", "_at", "_time")
    )


def _is_geo(col: dict[str, Any]) -> bool:
    name = _col_name(col).lower()
    return name in _GEO_NAMES or name in _LAT_NAMES or name in _LON_NAMES


class DashboardService:
    """Per-user dashboards CRUD + widget management + chart recommendation."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._log = logger.bind(service="Dashboard")

    # ── Chart recommendation (pure, LLM-free) ──────────────────────────────────

    @staticmethod
    def recommend_chart(
        columns: list[dict[str, Any]], rows: list[dict[str, Any]]
    ) -> ChartRecommendation:
        """Recommend the best visualization for a SQL result set.

        Deterministic rules (checked in priority order):
          * geography (lat/lon or country/region/state/city …) → ``map``
          * one numeric column, one row                        → ``kpi``
          * temporal column + numeric column                   → ``line``
          * exactly two numeric columns                        → ``scatter``
          * low-cardinality categorical + numeric              → ``bar``
          * a single numeric column (many rows)                → ``histogram``
          * anything else                                      → ``table``
        """
        cols = [c for c in (columns or []) if _col_name(c)]
        rows = rows or []

        if not cols:
            return {
                "chart_type": "table",
                "x_axis": None,
                "y_axis": None,
                "reason": "No columns to plot; showing a table.",
            }

        numeric = [c for c in cols if _is_numeric(c, rows)]
        temporal = [c for c in cols if _is_temporal(c)]
        geo = [c for c in cols if _is_geo(c)]
        categorical = [
            c for c in cols if c not in numeric and c not in temporal and c not in geo
        ]

        # 1. Geography → map
        if geo:
            lat = next((c for c in cols if _col_name(c).lower() in _LAT_NAMES), None)
            lon = next((c for c in cols if _col_name(c).lower() in _LON_NAMES), None)
            if lat is not None and lon is not None:
                return {
                    "chart_type": "map",
                    "x_axis": _col_name(lon),
                    "y_axis": _col_name(lat),
                    "reason": "Latitude/longitude columns detected; plotting a map.",
                }
            metric = numeric[0] if numeric else None
            return {
                "chart_type": "map",
                "x_axis": _col_name(geo[0]),
                "y_axis": _col_name(metric) if metric else None,
                "reason": "A geographic column was detected; plotting a map.",
            }

        # 2. Single numeric scalar (one column, one row) → KPI
        if len(cols) == 1 and numeric and len(rows) <= 1:
            return {
                "chart_type": "kpi",
                "x_axis": None,
                "y_axis": _col_name(numeric[0]),
                "reason": "A single numeric value; showing a KPI metric.",
            }

        # 3. Time series (temporal + numeric) → line
        if temporal and numeric:
            return {
                "chart_type": "line",
                "x_axis": _col_name(temporal[0]),
                "y_axis": _col_name(numeric[0]),
                "reason": "A date/time column with a numeric measure; plotting a line chart.",
            }

        # 4. Exactly two numeric columns → scatter
        if len(cols) == 2 and len(numeric) == 2:
            return {
                "chart_type": "scatter",
                "x_axis": _col_name(numeric[0]),
                "y_axis": _col_name(numeric[1]),
                "reason": "Two numeric columns; plotting a scatter chart.",
            }

        # 5. Low-cardinality categorical + numeric → bar
        if categorical and numeric:
            cat = categorical[0]
            distinct = len({row.get(_col_name(cat)) for row in rows}) if rows else 0
            if distinct <= _MAX_BAR_CARDINALITY:
                return {
                    "chart_type": "bar",
                    "x_axis": _col_name(cat),
                    "y_axis": _col_name(numeric[0]),
                    "reason": "A categorical column with a numeric measure; plotting a bar chart.",
                }

        # 6. A single numeric column across many rows → histogram
        if len(numeric) == 1 and len(cols) == 1:
            name = _col_name(numeric[0])
            return {
                "chart_type": "histogram",
                "x_axis": name,
                "y_axis": name,
                "reason": "One numeric column; showing its distribution as a histogram.",
            }

        # 7. Fallback → table
        return {
            "chart_type": "table",
            "x_axis": None,
            "y_axis": None,
            "reason": "No clear chart mapping; showing a table.",
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _load_owned(
        self,
        db: AsyncSession,
        user_id: str,
        dashboard_id: str,
    ) -> Dashboard | None:
        """Load a user-owned dashboard with widgets eager-loaded, or ``None``."""
        conditions = [Dashboard.id == dashboard_id, Dashboard.user_id == user_id]
        result = await db.execute(
            select(Dashboard).options(selectinload(Dashboard.widgets)).where(*conditions)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _build_widget(dashboard_id: str, data: dict[str, Any], position: int) -> DashboardWidget:
        return DashboardWidget(
            id=str(uuid4()),
            dashboard_id=dashboard_id,
            title=(data.get("title") or "Untitled")[:200],
            nl_prompt=data.get("nl_prompt"),
            sql=data.get("sql") or "",
            chart_type=(data.get("chart_type") or "table")[:20],
            chart_config=data.get("chart_config"),
            layout=data.get("layout"),
            position=data.get("position") if data.get("position") is not None else position,
        )

    # ── Dashboard CRUD ─────────────────────────────────────────────────────────

    async def create(
        self,
        user_id: str,
        name: str,
        widgets: list[dict[str, Any]] | None = None,
        is_builtin: bool = False,
    ) -> Dashboard:
        """Create a dashboard, optionally seeding it with widgets."""
        dashboard_id = str(uuid4())
        async with self._session_factory() as db:
            dashboard = Dashboard(
                id=dashboard_id,
                user_id=user_id,
                name=name[:200],
                is_builtin=is_builtin,
            )
            db.add(dashboard)
            for idx, widget_data in enumerate(widgets or []):
                db.add(self._build_widget(dashboard_id, widget_data, idx))
            await db.commit()
            loaded = await self._load_owned(db, user_id, dashboard_id)
        assert loaded is not None  # just created within this session
        self._log.info(
            "dashboard created",
            user_id=user_id,
            dashboard_id=dashboard_id,
            widgets=len(widgets or []),
        )
        return loaded

    async def get(self, user_id: str, dashboard_id: str) -> Dashboard | None:
        """Return a user-owned dashboard (with widgets), or ``None``."""
        async with self._session_factory() as db:
            return await self._load_owned(db, user_id, dashboard_id)

    async def list_dashboards(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Dashboard], int]:
        """Return ``(dashboards, total)`` for a user, most-recently-updated first."""
        conditions = [Dashboard.user_id == user_id]
        async with self._session_factory() as db:
            total = (
                await db.execute(
                    select(func.count()).select_from(Dashboard).where(*conditions)
                )
            ).scalar_one()

            result = await db.execute(
                select(Dashboard)
                .options(selectinload(Dashboard.widgets))
                .where(*conditions)
                .order_by(Dashboard.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            items = list(result.scalars().all())
        return items, int(total)

    async def rename(self, user_id: str, dashboard_id: str, name: str) -> Dashboard | None:
        """Rename a user-owned dashboard; ``None`` if not found."""
        async with self._session_factory() as db:
            dashboard = await self._load_owned(db, user_id, dashboard_id)
            if dashboard is None:
                return None
            dashboard.name = name[:200]
            dashboard.updated_at = datetime.utcnow()
            await db.commit()
            return await self._load_owned(db, user_id, dashboard_id)

    async def duplicate(self, user_id: str, dashboard_id: str) -> Dashboard | None:
        """Deep-copy a dashboard (name + all widgets); ``None`` if not found."""
        async with self._session_factory() as db:
            source = await self._load_owned(db, user_id, dashboard_id)
            if source is None:
                return None

            new_id = str(uuid4())
            clone = Dashboard(
                id=new_id,
                user_id=user_id,
                name=f"{source.name} (copy)"[:200],
            )
            db.add(clone)
            for widget in source.widgets:
                db.add(
                    DashboardWidget(
                        id=str(uuid4()),
                        dashboard_id=new_id,
                        title=widget.title,
                        nl_prompt=widget.nl_prompt,
                        sql=widget.sql,
                        chart_type=widget.chart_type,
                        chart_config=widget.chart_config,
                        layout=widget.layout,
                        position=widget.position,
                    )
                )
            await db.commit()
            loaded = await self._load_owned(db, user_id, new_id)
        self._log.info(
            "dashboard duplicated",
            user_id=user_id,
            source_id=dashboard_id,
            new_id=new_id,
        )
        return loaded

    async def delete(self, user_id: str, dashboard_id: str) -> bool:
        """Delete a user-owned dashboard (widgets cascade); False if not found."""
        async with self._session_factory() as db:
            dashboard = await self._load_owned(db, user_id, dashboard_id)
            if dashboard is None:
                return False
            await db.delete(dashboard)
            await db.commit()
        self._log.info("dashboard deleted", user_id=user_id, dashboard_id=dashboard_id)
        return True

    # ── Widget management ──────────────────────────────────────────────────────

    async def add_widget(
        self, user_id: str, dashboard_id: str, data: dict[str, Any]
    ) -> Dashboard | None:
        """Append a widget to a dashboard; ``None`` if dashboard not found."""
        async with self._session_factory() as db:
            dashboard = await self._load_owned(db, user_id, dashboard_id)
            if dashboard is None:
                return None
            next_pos = max((w.position for w in dashboard.widgets), default=-1) + 1
            db.add(self._build_widget(dashboard_id, data, next_pos))
            dashboard.updated_at = datetime.utcnow()
            await db.commit()
            return await self._load_owned(db, user_id, dashboard_id)

    async def update_widget(
        self,
        user_id: str,
        dashboard_id: str,
        widget_id: str,
        updates: dict[str, Any],
    ) -> Dashboard | None:
        """Patch a widget's editable fields; ``None`` if dashboard/widget absent."""
        allowed = {"title", "nl_prompt", "sql", "chart_type", "chart_config", "layout", "position"}
        async with self._session_factory() as db:
            dashboard = await self._load_owned(db, user_id, dashboard_id)
            if dashboard is None:
                return None
            widget = next((w for w in dashboard.widgets if w.id == widget_id), None)
            if widget is None:
                return None
            for field, value in updates.items():
                if field in allowed and value is not None:
                    setattr(widget, field, value)
            dashboard.updated_at = datetime.utcnow()
            await db.commit()
            return await self._load_owned(db, user_id, dashboard_id)

    async def delete_widget(
        self, user_id: str, dashboard_id: str, widget_id: str
    ) -> Dashboard | None:
        """Remove a widget; ``None`` if dashboard/widget not found."""
        async with self._session_factory() as db:
            dashboard = await self._load_owned(db, user_id, dashboard_id)
            if dashboard is None:
                return None
            widget = next((w for w in dashboard.widgets if w.id == widget_id), None)
            if widget is None:
                return None
            await db.delete(widget)
            dashboard.updated_at = datetime.utcnow()
            await db.commit()
            return await self._load_owned(db, user_id, dashboard_id)

    async def reorder_widgets(
        self, user_id: str, dashboard_id: str, ordered_ids: list[str]
    ) -> Dashboard | None:
        """Set widget ``position`` from the given id order; ``None`` if not found."""
        async with self._session_factory() as db:
            dashboard = await self._load_owned(db, user_id, dashboard_id)
            if dashboard is None:
                return None
            order = {wid: idx for idx, wid in enumerate(ordered_ids)}
            for widget in dashboard.widgets:
                if widget.id in order:
                    widget.position = order[widget.id]
            dashboard.updated_at = datetime.utcnow()
            await db.commit()
            return await self._load_owned(db, user_id, dashboard_id)
