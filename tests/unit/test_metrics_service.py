"""Unit tests for MetricsService.

Covers CRUD, read-scope (any user in a connection) vs write-scope (creator
only) enforcement, duplicate-name rejection, SQL validation via a stubbed
SQLColumnValidator, and certify/uncertify (including the "cannot certify with
validation errors" guard). DB access is mocked via an AsyncMock session
factory (matching test_dashboard_service.py's pattern).
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from nl_to_sql.core.exceptions import MetricNotFoundError, MetricValidationError
from nl_to_sql.infrastructure.database.models import Metric
from nl_to_sql.services.metrics_service import MetricsService

USER = "user-1"
OTHER = "user-2"
CONN = "conn-1"
DT = datetime(2026, 1, 1, 12, 0, 0)


def _mock_session() -> AsyncMock:
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock()
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


def _metric(**overrides) -> Metric:
    defaults = {
        "metric_id": "m1",
        "user_id": USER,
        "connection_id": CONN,
        "name": "Net Revenue",
        "description": "Total revenue minus refunds",
        "sql_definition": "SELECT SUM(amount) FROM orders",
        "dimensions": ["region"],
        "tags": ["finance"],
        "owner": "Finance Team",
        "certified": False,
        "created_at": DT,
        "updated_at": DT,
    }
    defaults.update(overrides)
    return Metric(**defaults)


def _service_with(
    db: AsyncMock,
    *,
    validator: MagicMock | None = None,
    catalog_service: MagicMock | None = None,
    sql_validator: MagicMock | None = None,
) -> MetricsService:
    factory = MagicMock(return_value=db)
    column_validator = validator or MagicMock(validate=MagicMock(return_value=[]))
    schema_catalog_service = catalog_service or MagicMock(
        get_catalog=AsyncMock(return_value={"tables": []})
    )
    if sql_validator is None:
        sql_validator = MagicMock()
        sql_validator.validate.return_value = MagicMock(is_valid=True, errors=[])
    return MetricsService(
        session_factory=factory,
        column_validator=column_validator,
        schema_catalog_service=schema_catalog_service,
        sql_validator=sql_validator,
    )


class TestReadScope:
    async def test_get_by_any_user_in_connection_succeeds(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(_metric()))
        svc = _service_with(db)
        info = await svc.get(OTHER, CONN, "m1")  # OTHER didn't create it
        assert info.metric_id == "m1"

    async def test_get_missing_raises_not_found(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))
        svc = _service_with(db)
        with pytest.raises(MetricNotFoundError):
            await svc.get(USER, CONN, "missing")


class TestWriteScope:
    async def test_update_by_non_creator_raises_not_found(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))  # not owned by OTHER
        svc = _service_with(db)
        with pytest.raises(MetricNotFoundError):
            await svc.update(OTHER, CONN, "m1", name="New name")

    async def test_delete_by_creator_succeeds(self) -> None:
        db = _mock_session()
        row = _metric()
        db.execute = AsyncMock(return_value=_result_scalar(row))
        svc = _service_with(db)
        await svc.delete(USER, CONN, "m1")
        db.delete.assert_awaited_once_with(row)


class TestCreate:
    async def test_rejects_empty_name(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar_one(0))
        svc = _service_with(db)
        with pytest.raises(MetricValidationError):
            await svc.create(USER, CONN, "  ", "desc", "SELECT 1")

    async def test_rejects_empty_sql(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar_one(0))
        svc = _service_with(db)
        with pytest.raises(MetricValidationError):
            await svc.create(USER, CONN, "Net Revenue", "desc", "   ")

    async def test_duplicate_name_maps_to_validation_error(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar_one(0))
        db.commit = AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("dup")))
        svc = _service_with(db)
        with pytest.raises(MetricValidationError):
            await svc.create(USER, CONN, "Net Revenue", "desc", "SELECT 1")
        db.rollback.assert_awaited_once()

    async def test_create_success(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar_one(0))
        svc = _service_with(db)
        info = await svc.create(USER, CONN, "Net Revenue", "desc", "SELECT 1")
        assert info.name == "Net Revenue"
        db.add.assert_called_once()


class TestSqlValidation:
    async def test_flags_hallucinated_column(self) -> None:
        db = _mock_session()
        validator = MagicMock(validate=MagicMock(return_value=["Unknown column: orders.bogus"]))
        catalog_service = MagicMock(
            get_catalog=AsyncMock(
                return_value={"tables": [{"table_name": "orders", "columns": [{"name": "amount"}]}]}
            )
        )
        svc = _service_with(db, validator=validator, catalog_service=catalog_service)
        errors = await svc.validate_sql_definition(USER, CONN, "SELECT bogus FROM orders")
        assert errors == ["Unknown column: orders.bogus"]
        catalog_service.get_catalog.assert_awaited_once_with(USER, CONN)

    async def test_catalog_failure_degrades_to_no_errors(self) -> None:
        db = _mock_session()
        catalog_service = MagicMock(get_catalog=AsyncMock(side_effect=RuntimeError("boom")))
        svc = _service_with(db, catalog_service=catalog_service)
        errors = await svc.validate_sql_definition(USER, CONN, "SELECT 1")
        assert errors == []


class TestCertify:
    async def test_certify_success(self) -> None:
        db = _mock_session()
        row = _metric(certified=False)
        db.execute = AsyncMock(return_value=_result_scalar(row))
        svc = _service_with(db)
        info = await svc.certify(USER, CONN, "m1")
        assert info.certified is True

    async def test_certify_blocked_by_validation_errors(self) -> None:
        db = _mock_session()
        row = _metric(certified=False)
        db.execute = AsyncMock(return_value=_result_scalar(row))
        validator = MagicMock(validate=MagicMock(return_value=["Unknown column: x"]))
        svc = _service_with(db, validator=validator)
        with pytest.raises(MetricValidationError):
            await svc.certify(USER, CONN, "m1")

    async def test_uncertify_success(self) -> None:
        db = _mock_session()
        row = _metric(certified=True)
        db.execute = AsyncMock(return_value=_result_scalar(row))
        svc = _service_with(db)
        info = await svc.uncertify(USER, CONN, "m1")
        assert info.certified is False

    async def test_certify_by_non_creator_raises_not_found(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(None))
        svc = _service_with(db)
        with pytest.raises(MetricNotFoundError):
            await svc.certify(OTHER, CONN, "m1")


class TestListMetricsCatalogFetchedOnce:
    """Medium finding: list_metrics re-fetched the schema catalog once per
    row instead of once per call — ~200 redundant identical queries near
    the metrics cap."""

    async def test_get_catalog_called_once_regardless_of_row_count(self) -> None:
        db = _mock_session()
        rows = [_metric(metric_id=f"m{i}") for i in range(5)]
        db.execute = AsyncMock(
            side_effect=[_result_scalar_one(5), _result_scalars(rows)]
        )
        catalog_service = MagicMock(
            get_catalog=AsyncMock(return_value={"tables": []})
        )
        svc = _service_with(db, catalog_service=catalog_service)

        infos, total = await svc.list_metrics(USER, CONN)

        assert total == 5
        assert len(infos) == 5
        catalog_service.get_catalog.assert_awaited_once()


class TestListForPrompt:
    async def test_only_certified_returned(self) -> None:
        db = _mock_session()
        certified = _metric(metric_id="m1", certified=True)
        db.execute = AsyncMock(return_value=_result_scalars([certified]))
        svc = _service_with(db)
        results = await svc.list_certified_for_prompt(CONN)
        assert len(results) == 1
        assert results[0]["name"] == "Net Revenue"


# ── C6: sql_definition is validated against SQLValidatorService ─────────────


def _service_with_real_sql_validator(db: AsyncMock) -> MetricsService:
    from nl_to_sql.services.sql_validator import SQLValidatorService

    factory = MagicMock(return_value=db)
    column_validator = MagicMock(validate=MagicMock(return_value=[]))
    schema_catalog_service = MagicMock(get_catalog=AsyncMock(return_value={"tables": []}))
    return MetricsService(
        session_factory=factory,
        column_validator=column_validator,
        schema_catalog_service=schema_catalog_service,
        sql_validator=SQLValidatorService(),
    )


class TestSqlSafetyEnforcement:
    async def test_create_rejects_dangerous_function(self) -> None:
        db = _mock_session()
        svc = _service_with_real_sql_validator(db)

        with pytest.raises(MetricValidationError):
            await svc.create(
                USER, CONN, "Evil", None, "SELECT pg_read_file('/etc/passwd')"
            )
        db.add.assert_not_called()

    async def test_create_rejects_non_select(self) -> None:
        db = _mock_session()
        svc = _service_with_real_sql_validator(db)

        with pytest.raises(MetricValidationError):
            await svc.create(USER, CONN, "Evil", None, "DELETE FROM orders")

    async def test_create_allows_clean_sql(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar_one(0))
        svc = _service_with_real_sql_validator(db)

        info = await svc.create(USER, CONN, "Net Revenue", None, "SELECT SUM(amount) FROM orders")
        assert info.name == "Net Revenue"

    async def test_update_rejects_dangerous_sql(self) -> None:
        db = _mock_session()
        db.execute = AsyncMock(return_value=_result_scalar(_metric()))
        svc = _service_with_real_sql_validator(db)

        with pytest.raises(MetricValidationError):
            await svc.update(
                USER, CONN, "m1", sql_definition="SELECT dblink_connect('host=evil.example')"
            )

    async def test_certify_rechecks_sql_safety_for_legacy_metrics(self) -> None:
        """A metric persisted before this check existed must not be certifiable."""
        db = _mock_session()
        malicious = _metric(sql_definition="SELECT pg_sleep(300)")
        db.execute = AsyncMock(return_value=_result_scalar(malicious))
        svc = _service_with_real_sql_validator(db)

        with pytest.raises(MetricValidationError):
            await svc.certify(USER, CONN, "m1")
