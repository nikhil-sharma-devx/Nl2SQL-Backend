"""Unit tests for DashboardService.

Covers the deterministic ``recommend_chart`` heuristic (all six mappings + the
table fallback) and the per-user CRUD surface (create / list / rename /
duplicate deep-copy / delete / widget ops / cross-user → not found). DB access
is mocked via an AsyncMock session factory (matching the repo's route tests).
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from nl_to_sql.infrastructure.database.models import Dashboard, DashboardWidget
from nl_to_sql.services.dashboard_service import DashboardService

USER = "user-1"
OTHER = "user-2"
DT = datetime(2024, 1, 1, 12, 0, 0)


# ── recommend_chart ─────────────────────────────────────────────────────────────


class TestRecommendChart:
    def test_line_for_temporal_plus_numeric(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "day", "type": "date"}, {"name": "revenue", "type": "float"}],
            [{"day": "2024-01-01", "revenue": 10}, {"day": "2024-01-02", "revenue": 20}],
        )
        assert rec["chart_type"] == "line"
        assert rec["x_axis"] == "day"
        assert rec["y_axis"] == "revenue"

    def test_bar_for_categorical_plus_numeric(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "category", "type": "text"}, {"name": "count", "type": "integer"}],
            [{"category": "A", "count": 3}, {"category": "B", "count": 5}],
        )
        assert rec["chart_type"] == "bar"
        assert rec["x_axis"] == "category"
        assert rec["y_axis"] == "count"

    def test_histogram_for_single_numeric_many_rows(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "age", "type": "integer"}],
            [{"age": 20}, {"age": 21}, {"age": 22}],
        )
        assert rec["chart_type"] == "histogram"
        assert rec["y_axis"] == "age"

    def test_kpi_for_single_scalar(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "total", "type": "integer"}], [{"total": 42}]
        )
        assert rec["chart_type"] == "kpi"
        assert rec["y_axis"] == "total"

    def test_map_for_lat_lon(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "latitude", "type": "float"}, {"name": "longitude", "type": "float"}],
            [{"latitude": 40.7, "longitude": -74.0}],
        )
        assert rec["chart_type"] == "map"
        assert rec["x_axis"] == "longitude"
        assert rec["y_axis"] == "latitude"

    def test_map_for_place_name(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "country", "type": "text"}, {"name": "sales", "type": "float"}],
            [{"country": "US", "sales": 100}],
        )
        assert rec["chart_type"] == "map"
        assert rec["x_axis"] == "country"

    def test_scatter_for_two_numeric(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "height", "type": "float"}, {"name": "weight", "type": "float"}],
            [{"height": 1.8, "weight": 80}, {"height": 1.6, "weight": 60}],
        )
        assert rec["chart_type"] == "scatter"
        assert rec["x_axis"] == "height"
        assert rec["y_axis"] == "weight"

    def test_table_fallback_all_strings(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "name", "type": "text"}, {"name": "description", "type": "text"}],
            [{"name": "alice", "description": "hello"}],
        )
        assert rec["chart_type"] == "table"

    def test_table_for_empty_columns(self) -> None:
        assert DashboardService.recommend_chart([], [])["chart_type"] == "table"

    def test_numeric_inferred_from_sample_when_type_missing(self) -> None:
        rec = DashboardService.recommend_chart(
            [{"name": "category"}, {"name": "revenue"}],
            [{"category": "A", "revenue": 100}],
        )
        assert rec["chart_type"] == "bar"
        assert rec["y_axis"] == "revenue"


# ── Service CRUD (mocked session) ────────────────────────────────────────────────


def _mock_session() -> AsyncMock:
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.delete = AsyncMock()
    return db


def _result_scalar(value: object) -> MagicMock:
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    return r


def _result_scalar_one(value: object) -> MagicMock:
    r = MagicMock()
    r.scalar_one = MagicMock(return_value=value)
    return r


def _result_scalars(items: list) -> MagicMock:
    r = MagicMock()
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=items)))
    return r


def _make_dashboard(
    dashboard_id: str = "dash-1", user_id: str = USER, widgets: list | None = None
) -> Dashboard:
    d = Dashboard()
    d.id = dashboard_id
    d.user_id = user_id
    d.name = "My Dashboard"
    d.created_at = DT
    d.updated_at = DT
    d.widgets = widgets or []
    return d


def _make_widget(
    widget_id: str, dashboard_id: str = "dash-1", position: int = 0
) -> DashboardWidget:
    w = DashboardWidget()
    w.id = widget_id
    w.dashboard_id = dashboard_id
    w.title = f"Widget {widget_id}"
    w.nl_prompt = "show data"
    w.sql = "SELECT 1"
    w.chart_type = "bar"
    w.chart_config = {"type": "bar", "x_axis": "a", "y_axis": "b"}
    w.layout = {"x": 0, "y": 0, "w": 6, "h": 4}
    w.position = position
    w.created_at = DT
    return w


def _service_with(db: AsyncMock) -> DashboardService:
    factory = MagicMock(return_value=db)
    validator = MagicMock()
    validator.validate.return_value = MagicMock(is_valid=True, errors=[])
    return DashboardService(session_factory=factory, sql_validator=validator)


class TestDashboardServiceCrud:
    async def test_create_returns_loaded_dashboard(self) -> None:
        db = _mock_session()
        created = _make_dashboard()
        db.execute = AsyncMock(return_value=_result_scalar(created))
        svc = _service_with(db)

        result = await svc.create(USER, "My Dashboard", widgets=[{"title": "W", "sql": "SELECT 1"}])

        assert result is created
        db.add.assert_called()  # dashboard + widget added
        db.commit.assert_awaited_once()

    async def test_get_returns_none_cross_user(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))
        svc = _service_with(db)

        assert await svc.get(OTHER, "dash-1") is None

    async def test_list_returns_items_and_total(self) -> None:
        db = _mock_session()
        d = _make_dashboard()
        db.execute = AsyncMock(side_effect=[_result_scalar_one(1), _result_scalars([d])])
        svc = _service_with(db)

        items, total = await svc.list_dashboards(USER, limit=20, offset=0)
        assert total == 1
        assert items == [d]

    async def test_rename_not_found_returns_none(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))
        svc = _service_with(db)

        assert await svc.rename(USER, "missing", "New") is None
        db.commit.assert_not_called()

    async def test_rename_updates_name(self) -> None:
        db = _mock_session()
        d = _make_dashboard()
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with(db)

        result = await svc.rename(USER, "dash-1", "Renamed")
        assert result is d
        assert d.name == "Renamed"
        db.commit.assert_awaited_once()

    async def test_duplicate_deep_copies_widgets(self) -> None:
        db = _mock_session()
        source = _make_dashboard(
            widgets=[_make_widget("w1", position=0), _make_widget("w2", position=1)]
        )
        # First load = source; second load = the new dashboard (return source stand-in)
        db.execute = AsyncMock(side_effect=[_result_scalar(source), _result_scalar(source)])
        svc = _service_with(db)

        await svc.duplicate(USER, "dash-1")

        added = db.add.call_args_list
        # 1 cloned dashboard + 2 cloned widgets
        assert len(added) == 3
        clone_dashboard = added[0].args[0]
        assert isinstance(clone_dashboard, Dashboard)
        assert clone_dashboard.id != "dash-1"
        assert clone_dashboard.name == "My Dashboard (copy)"
        clone_widgets = [c.args[0] for c in added[1:]]
        assert all(isinstance(w, DashboardWidget) for w in clone_widgets)
        assert {w.sql for w in clone_widgets} == {"SELECT 1"}
        # widget ids are freshly minted (not the originals)
        assert {w.id for w in clone_widgets}.isdisjoint({"w1", "w2"})
        assert all(w.dashboard_id == clone_dashboard.id for w in clone_widgets)

    async def test_duplicate_cross_user_returns_none(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))
        svc = _service_with(db)

        assert await svc.duplicate(OTHER, "dash-1") is None
        db.add.assert_not_called()

    async def test_delete_found(self) -> None:
        db = _mock_session()
        d = _make_dashboard()
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with(db)

        assert await svc.delete(USER, "dash-1") is True
        db.delete.assert_awaited_once_with(d)
        db.commit.assert_awaited_once()

    async def test_delete_not_found(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))
        svc = _service_with(db)

        assert await svc.delete(OTHER, "dash-1") is False
        db.delete.assert_not_called()

    async def test_add_widget_appends_with_next_position(self) -> None:
        db = _mock_session()
        d = _make_dashboard(widgets=[_make_widget("w1", position=0)])
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with(db)

        await svc.add_widget(USER, "dash-1", {"title": "New", "sql": "SELECT 2"})
        added = db.add.call_args_list[0].args[0]
        assert isinstance(added, DashboardWidget)
        assert added.position == 1  # next after existing position 0

    async def test_update_widget_missing_widget_returns_none(self) -> None:
        db = _mock_session()
        d = _make_dashboard(widgets=[_make_widget("w1")])
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with(db)

        assert await svc.update_widget(USER, "dash-1", "nope", {"title": "X"}) is None

    async def test_update_widget_sets_fields(self) -> None:
        db = _mock_session()
        widget = _make_widget("w1")
        d = _make_dashboard(widgets=[widget])
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with(db)

        await svc.update_widget(USER, "dash-1", "w1", {"title": "Renamed", "chart_type": "line"})
        assert widget.title == "Renamed"
        assert widget.chart_type == "line"

    async def test_reorder_widgets_sets_positions(self) -> None:
        db = _mock_session()
        w1, w2 = _make_widget("w1", position=0), _make_widget("w2", position=1)
        d = _make_dashboard(widgets=[w1, w2])
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with(db)

        await svc.reorder_widgets(USER, "dash-1", ["w2", "w1"])
        assert w2.position == 0
        assert w1.position == 1


# ── C5: widget SQL is validated against SQLValidatorService ─────────────────


def _service_with_real_validator(db: AsyncMock) -> DashboardService:
    from nl_to_sql.services.sql_validator import SQLValidatorService

    factory = MagicMock(return_value=db)
    return DashboardService(session_factory=factory, sql_validator=SQLValidatorService())


class TestWidgetSqlValidation:
    async def test_create_rejects_dangerous_function(self) -> None:
        from nl_to_sql.core.exceptions import SQLValidationError

        db = _mock_session()
        svc = _service_with_real_validator(db)

        with pytest.raises(SQLValidationError):
            await svc.create(USER, "Dash", widgets=[{"sql": "SELECT pg_sleep(300)"}])
        db.add.assert_not_called()

    async def test_create_rejects_non_select(self) -> None:
        from nl_to_sql.core.exceptions import SQLValidationError

        db = _mock_session()
        svc = _service_with_real_validator(db)

        with pytest.raises(SQLValidationError):
            await svc.create(USER, "Dash", widgets=[{"sql": "DELETE FROM users"}])

    async def test_create_allows_empty_sql_draft(self) -> None:
        db = _mock_session()
        created = _make_dashboard()
        db.execute = AsyncMock(return_value=_result_scalar(created))
        svc = _service_with_real_validator(db)

        result = await svc.create(USER, "Dash", widgets=[{"title": "Draft", "sql": ""}])
        assert result is created

    async def test_add_widget_rejects_dblink(self) -> None:
        from nl_to_sql.core.exceptions import SQLValidationError

        db = _mock_session()
        d = _make_dashboard(widgets=[])
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with_real_validator(db)

        with pytest.raises(SQLValidationError):
            await svc.add_widget(
                USER, "dash-1", {"sql": "SELECT dblink_connect('host=evil.example')"}
            )

    async def test_update_widget_rejects_dangerous_sql(self) -> None:
        from nl_to_sql.core.exceptions import SQLValidationError

        db = _mock_session()
        widget = _make_widget("w1")
        d = _make_dashboard(widgets=[widget])
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with_real_validator(db)

        with pytest.raises(SQLValidationError):
            await svc.update_widget(
                USER, "dash-1", "w1", {"sql": "SELECT pg_read_file('/etc/passwd')"}
            )

    async def test_update_widget_without_sql_field_untouched(self) -> None:
        db = _mock_session()
        widget = _make_widget("w1")
        d = _make_dashboard(widgets=[widget])
        db.execute = AsyncMock(return_value=_result_scalar(d))
        svc = _service_with_real_validator(db)

        # No 'sql' key at all — must not trigger validation or raise.
        await svc.update_widget(USER, "dash-1", "w1", {"title": "Renamed"})
        assert widget.title == "Renamed"
