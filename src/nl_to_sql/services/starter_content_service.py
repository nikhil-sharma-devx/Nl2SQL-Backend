"""StarterContentService — seeds real, working example content on first visit.

Seeds each user's Templates / Metrics / Scheduled Queries / Dashboards with 3
built-in examples the first time they hit each screen, so a brand-new user
sees the feature in action instead of a bare "No X yet" empty state. All seed
SQL is grounded in the shared "Server Default" demo database's ``customer``
and ``sales_order`` tables — the only tables in that dataset with real,
non-placeholder columns — so every seeded template/metric/widget runs
successfully and returns a meaningful result, not a stand-in.

Idempotency is a one-way gate via markers stored in
``OnboardingState.completed_items`` (the same JSON list column
``api/routes/tutorial.py`` uses for the user-facing onboarding checklist —
reused here rather than adding a new table). This is deliberately distinct
from a naive "seed if the list is empty" check: once a domain's
``"seeded_<domain>"`` marker is present, seeding never runs again for that
user, even if they delete every built-in row — matching "built-ins behave
exactly like user-created items" (a deleted item must never silently
reappear). The extra markers coexist with real onboarding-checklist items in
the same list but never affect onboarding progress, since that calculation
intersects against a fixed whitelist (``ONBOARDING_ITEMS``) these marker
strings aren't part of.

Seed rows are created through each domain's own service (``MetricsService``,
``ScheduledQueryService``, ``DashboardService``) so they go through the exact
same validation/caps/business rules as a user-created row — this service
never bypasses that. ``QueryTemplate`` has no dedicated service yet (its CRUD
lives inline in ``api/routes/query_templates.py``); this service inserts rows
directly via its own injected session factory, matching that route's existing
pattern rather than introducing a new architectural layer for it.

Called from the hot list-read path of four screens (see the four routes'
``list_*`` handlers) — so every public method here is best-effort: failures
are logged and swallowed, never raised, since a bug in starter-content
seeding must never turn a user's real list-fetch into a 500 (exactly the
failure mode a prior bug in ``ConnectionService.ensure_default_connection``
already caused once, unguarded, on this same kind of call site).
"""
from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.infrastructure.database.models import OnboardingState, QueryTemplate
from nl_to_sql.services.dashboard_service import DashboardService
from nl_to_sql.services.metrics_service import MetricsService
from nl_to_sql.services.scheduled_query_service import ScheduledQueryService

logger = structlog.get_logger(__name__)

MARKER_TEMPLATES = "seeded_templates"
MARKER_METRICS = "seeded_metrics"
MARKER_SCHEDULES = "seeded_schedules"
MARKER_DASHBOARDS = "seeded_dashboards"

# ── Seed content — grounded in customer / sales_order's real columns ─────────

_TEMPLATE_SEED: list[dict] = [
    {
        "name": "Customer Retention Analysis",
        "description": (
            "Monthly customer signups vs. how many remain active — a quick "
            "pulse-check on retention."
        ),
        "template_nl": "Show monthly customer signups and how many of them are still active.",
        "template_sql": (
            "SELECT DATE_TRUNC('month', created_at) AS signup_month, "
            "COUNT(*) AS total_customers, "
            "COUNT(*) FILTER (WHERE status = 'ACTIVE') AS active_customers "
            "FROM customer GROUP BY 1 ORDER BY 1"
        ),
        "parameters": [],
        "tags": ["example", "customers"],
    },
    {
        "name": "High-Value Customer Report",
        "description": "Customers above a minimum annual income, ranked by credit score.",
        "template_nl": (
            "List customers with annual income above {{min_income}}, ordered by credit score."
        ),
        "template_sql": (
            "SELECT customer_id, full_name, email, annual_income, credit_score "
            "FROM customer WHERE annual_income >= {{min_income}} "
            "ORDER BY credit_score DESC LIMIT 100"
        ),
        "parameters": [
            {
                "name": "min_income",
                "type": "number",
                "description": "Minimum annual income threshold",
                "default": "75000",
            }
        ],
        "tags": ["example", "customers"],
    },
    {
        "name": "Monthly Customer Signups",
        "description": "Count of new customers created each calendar month.",
        "template_nl": "Show the number of new customers created each month.",
        "template_sql": (
            "SELECT DATE_TRUNC('month', created_at) AS month, COUNT(*) AS new_customers "
            "FROM customer GROUP BY 1 ORDER BY 1"
        ),
        "parameters": [],
        "tags": ["example", "growth"],
    },
]

_METRIC_SEED: list[dict] = [
    {
        "name": "Active Customers",
        "description": "Count of customers currently marked active.",
        "sql_definition": "SELECT COUNT(*) FROM customer WHERE status = 'ACTIVE'",
        "dimensions": ["status"],
    },
    {
        "name": "Average Credit Score",
        "description": "Mean credit score across all customers.",
        "sql_definition": "SELECT AVG(credit_score) FROM customer",
        "dimensions": ["credit_score"],
    },
    {
        "name": "Total Customers",
        "description": "Total number of customers on record — the headline top-line count.",
        "sql_definition": "SELECT COUNT(*) FROM customer",
        "dimensions": [],
    },
]

# schedule_text accepts a literal 5-field cron expression, passed straight
# through by ``cron_utils.parse_schedule_text`` (see that module's docstring).
_SCHEDULE_SEED: list[dict] = [
    {
        "name": "Daily Active Customer Report",
        "nl_prompt": "How many active customers do we have today?",
        "schedule_text": "0 9 * * *",
        "notify_condition": "always",
    },
    {
        "name": "Weekly Customer Growth Report",
        "nl_prompt": "Show new customer signups from the past 7 days.",
        "schedule_text": "0 9 * * 1",
        "notify_condition": "always",
    },
    {
        "name": "Low Credit Score Monitor",
        "nl_prompt": "List customers with a credit score below 600.",
        "schedule_text": "0 * * * *",
        "notify_condition": "on_results",
    },
]

_DASHBOARD_SEED: list[dict] = [
    {
        "name": "Executive Overview",
        "widgets": [
            {
                "title": "Total Customers",
                "nl_prompt": "How many customers do we have in total?",
                "sql": "SELECT COUNT(*) AS total_customers FROM customer",
                "chart_type": "kpi",
                "chart_config": {"type": "kpi", "x_axis": None, "y_axis": "total_customers"},
                "layout": {"x": 0, "y": 0, "w": 6, "h": 4},
            },
            {
                "title": "Customers by Status",
                "nl_prompt": "Break down customers by status.",
                "sql": "SELECT status, COUNT(*) AS customer_count FROM customer GROUP BY status",
                "chart_type": "bar",
                "chart_config": {"type": "bar", "x_axis": "status", "y_axis": "customer_count"},
                "layout": {"x": 6, "y": 0, "w": 6, "h": 4},
            },
        ],
    },
    {
        "name": "Customer Analytics",
        "widgets": [
            {
                "title": "Credit Score Distribution",
                "nl_prompt": "Break customers down into credit score bands.",
                "sql": (
                    "SELECT CASE "
                    "WHEN credit_score < 580 THEN 'Poor' "
                    "WHEN credit_score < 670 THEN 'Fair' "
                    "WHEN credit_score < 740 THEN 'Good' "
                    "WHEN credit_score < 800 THEN 'Very Good' "
                    "ELSE 'Excellent' END AS score_band, "
                    "COUNT(*) AS customer_count FROM customer GROUP BY 1 ORDER BY 1"
                ),
                "chart_type": "bar",
                "chart_config": {"type": "bar", "x_axis": "score_band", "y_axis": "customer_count"},
                "layout": {"x": 0, "y": 0, "w": 6, "h": 4},
            },
            {
                "title": "Income Bracket Breakdown",
                "nl_prompt": "Break customers down into annual income brackets.",
                "sql": (
                    "SELECT CASE "
                    "WHEN annual_income < 30000 THEN 'Under 30k' "
                    "WHEN annual_income < 60000 THEN '30k-60k' "
                    "WHEN annual_income < 100000 THEN '60k-100k' "
                    "ELSE '100k+' END AS income_bracket, "
                    "COUNT(*) AS customer_count FROM customer GROUP BY 1 ORDER BY 1"
                ),
                "chart_type": "bar",
                "chart_config": {
                    "type": "bar", "x_axis": "income_bracket", "y_axis": "customer_count"
                },
                "layout": {"x": 6, "y": 0, "w": 6, "h": 4},
            },
        ],
    },
    {
        "name": "Sales Performance",
        "widgets": [
            {
                "title": "Orders by Status",
                "nl_prompt": "Break down sales orders by status.",
                "sql": "SELECT status, COUNT(*) AS order_count FROM sales_order GROUP BY status",
                "chart_type": "bar",
                "chart_config": {"type": "bar", "x_axis": "status", "y_axis": "order_count"},
                "layout": {"x": 0, "y": 0, "w": 6, "h": 4},
            },
            {
                "title": "Orders Created Over Time",
                "nl_prompt": "Show sales orders created per month.",
                "sql": (
                    "SELECT DATE_TRUNC('month', created_at) AS month, "
                    "COUNT(*) AS order_count FROM sales_order GROUP BY 1 ORDER BY 1"
                ),
                "chart_type": "line",
                "chart_config": {"type": "line", "x_axis": "month", "y_axis": "order_count"},
                "layout": {"x": 6, "y": 0, "w": 6, "h": 4},
            },
        ],
    },
]


class StarterContentService:
    """Seeds starter Templates/Metrics/Schedules/Dashboards, once per user."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        metrics_service: MetricsService,
        scheduled_query_service: ScheduledQueryService,
        dashboard_service: DashboardService,
    ) -> None:
        self._session_factory = session_factory
        self._metrics_service = metrics_service
        self._scheduled_query_service = scheduled_query_service
        self._dashboard_service = dashboard_service
        self._log = logger.bind(service="StarterContent")

    # ── Marker bookkeeping (shared OnboardingState.completed_items) ─────────────

    async def _has_marker(self, user_id: str, marker: str) -> bool:
        async with self._session_factory() as db:
            row = (
                await db.execute(select(OnboardingState).where(OnboardingState.user_id == user_id))
            ).scalar_one_or_none()
        return marker in set((row.completed_items or []) if row else [])

    async def _set_marker(self, user_id: str, marker: str) -> None:
        async with self._session_factory() as db:
            row = (
                await db.execute(select(OnboardingState).where(OnboardingState.user_id == user_id))
            ).scalar_one_or_none()
            if row is None:
                db.add(OnboardingState(user_id=user_id, completed_items=[marker]))
            else:
                existing = set(row.completed_items or [])
                existing.add(marker)
                row.completed_items = list(existing)
            await db.commit()

    # ── Per-domain seeding (each best-effort: log + swallow, never raise) ───────

    async def ensure_templates_seeded(self, user_id: str) -> None:
        try:
            if await self._has_marker(user_id, MARKER_TEMPLATES):
                return
            async with self._session_factory() as db:
                for item in _TEMPLATE_SEED:
                    db.add(
                        QueryTemplate(
                            user_id=user_id,
                            name=item["name"],
                            description=item["description"],
                            template_nl=item["template_nl"],
                            template_sql=item["template_sql"],
                            parameters=item["parameters"],
                            tags=item["tags"],
                            is_builtin=True,
                        )
                    )
                await db.commit()
            await self._set_marker(user_id, MARKER_TEMPLATES)
            self._log.info("seeded starter templates", user_id=user_id, count=len(_TEMPLATE_SEED))
        except Exception as exc:
            self._log.warning("starter template seeding failed", user_id=user_id, error=str(exc))

    async def ensure_metrics_seeded(self, user_id: str, connection_id: str) -> None:
        try:
            if await self._has_marker(user_id, MARKER_METRICS):
                return
            for item in _METRIC_SEED:
                info = await self._metrics_service.create(
                    user_id,
                    connection_id,
                    item["name"],
                    item["description"],
                    item["sql_definition"],
                    dimensions=item["dimensions"],
                    tags=["example"],
                    is_builtin=True,
                )
                await self._metrics_service.certify(user_id, connection_id, info.metric_id)
            await self._set_marker(user_id, MARKER_METRICS)
            self._log.info(
                "seeded starter metrics", user_id=user_id, connection_id=connection_id,
                count=len(_METRIC_SEED),
            )
        except Exception as exc:
            self._log.warning("starter metric seeding failed", user_id=user_id, error=str(exc))

    async def ensure_schedules_seeded(self, user_id: str, connection_id: str) -> None:
        try:
            if await self._has_marker(user_id, MARKER_SCHEDULES):
                return
            for item in _SCHEDULE_SEED:
                # is_paused=True is passed directly into create() (one atomic
                # INSERT) rather than create-then-pause as two transactions —
                # the latter leaves a real window where the row briefly exists
                # unpaused and the background worker can pick it up and fire
                # it once before the follow-up pause() lands.
                await self._scheduled_query_service.create(
                    user_id,
                    connection_id,
                    item["name"],
                    item["nl_prompt"],
                    item["schedule_text"],
                    notify_condition=item["notify_condition"],
                    is_builtin=True,
                    is_paused=True,
                )
            await self._set_marker(user_id, MARKER_SCHEDULES)
            self._log.info(
                "seeded starter schedules", user_id=user_id, connection_id=connection_id,
                count=len(_SCHEDULE_SEED),
            )
        except Exception as exc:
            self._log.warning("starter schedule seeding failed", user_id=user_id, error=str(exc))

    async def ensure_dashboards_seeded(self, user_id: str) -> None:
        try:
            if await self._has_marker(user_id, MARKER_DASHBOARDS):
                return
            for item in _DASHBOARD_SEED:
                await self._dashboard_service.create(
                    user_id, item["name"], item["widgets"], is_builtin=True
                )
            await self._set_marker(user_id, MARKER_DASHBOARDS)
            self._log.info("seeded starter dashboards", user_id=user_id, count=len(_DASHBOARD_SEED))
        except Exception as exc:
            self._log.warning("starter dashboard seeding failed", user_id=user_id, error=str(exc))
