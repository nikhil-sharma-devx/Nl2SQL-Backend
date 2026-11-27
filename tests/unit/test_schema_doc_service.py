"""Unit tests for SchemaDocService — RAG-powered schema explanations.

Covers:
  - structured explanation returned (LLM enrichment + FK-derived facts)
  - cache hit avoids re-calling the LLM
  - LLM-failure fallback still returns schema-derived parts
  - unknown table / unknown column raise TableNotFoundError
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nl_to_sql.core.exceptions import TableNotFoundError
from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.core.models.sql_result import LLMResponse
from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache
from nl_to_sql.services.schema_doc_service import SchemaDocService

USER_ID = "user-doc-1"
CONNECTION_ID = "conn-doc-1"

_LLM_JSON = (
    '{"description": "Orders placed by customers.", '
    '"business_meaning": "Each row is a purchase.", '
    '"example_usage": "Count orders per customer.", '
    '"example_sql": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;"}'
)


def _catalog_data() -> dict[str, Any]:
    """A catalog with FK links in both directions around ``orders``."""
    return {
        "database_name": "shop",
        "dialect": "postgresql",
        "source": "reflected",
        "last_synced_at": None,
        "tables": [
            {
                "id": 1,
                "schema_name": "public",
                "table_name": "customers",
                "source": "reflected",
                "columns": [
                    {"name": "customer_id", "data_type": "INTEGER", "primary_key": True,
                     "foreign_key": None, "description": None, "nullable": False},
                    {"name": "email", "data_type": "TEXT", "primary_key": False,
                     "foreign_key": None, "description": None, "nullable": True},
                ],
                "description": None,
                "pinned": False,
                "is_new": False,
                "last_seen_at": None,
            },
            {
                "id": 2,
                "schema_name": "public",
                "table_name": "orders",
                "source": "reflected",
                "columns": [
                    {"name": "order_id", "data_type": "INTEGER", "primary_key": True,
                     "foreign_key": None, "description": None, "nullable": False},
                    {"name": "customer_id", "data_type": "INTEGER", "primary_key": False,
                     "foreign_key": "customers.customer_id", "description": None, "nullable": True},
                    {"name": "total", "data_type": "REAL", "primary_key": False,
                     "foreign_key": None, "description": "Order total in USD.", "nullable": True},
                ],
                "description": "Customer orders.",
                "pinned": False,
                "is_new": False,
                "last_seen_at": None,
            },
            {
                "id": 3,
                "schema_name": "public",
                "table_name": "order_items",
                "source": "reflected",
                "columns": [
                    {"name": "item_id", "data_type": "INTEGER", "primary_key": True,
                     "foreign_key": None, "description": None, "nullable": False},
                    {"name": "order_id", "data_type": "INTEGER", "primary_key": False,
                     "foreign_key": "orders.order_id", "description": None, "nullable": True},
                ],
                "description": None,
                "pinned": False,
                "is_new": False,
                "last_seen_at": None,
            },
        ],
    }


def _make_service(
    *,
    llm: MagicMock | None = None,
    cache: InMemoryCache | None = None,
    catalog: MagicMock | None = None,
    vector_store: MagicMock | None = None,
) -> SchemaDocService:
    catalog = catalog or MagicMock()
    if not hasattr(catalog.get_catalog, "assert_awaited"):
        catalog.get_catalog = AsyncMock(return_value=_catalog_data())

    vector_store = vector_store or MagicMock()
    vector_store.get_chunks_by_table_names = AsyncMock(
        return_value=[
            SchemaChunk(
                chunk_id="public.orders",
                table_name="orders",
                content="Table: orders — customer purchases.",
            )
        ]
    )

    if llm is None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(content=_LLM_JSON, prompt_tokens=10, completion_tokens=20)
        )

    return SchemaDocService(
        catalog=catalog,
        vector_store=vector_store,
        llm_provider=llm,
        cache=cache or InMemoryCache(default_ttl=60),
        per_user_isolation=True,
        cache_ttl_seconds=60,
    )


class TestExplainTable:

    async def test_returns_structured_explanation(self) -> None:
        svc = _make_service()
        result = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders")

        assert result.table == "orders"
        assert result.column is None
        assert result.description == "Orders placed by customers."
        assert result.business_meaning == "Each row is a purchase."
        assert result.example_usage
        assert "orders" in result.example_sql
        assert result.llm_generated is True
        assert result.cached is False

    async def test_relationships_derived_both_directions(self) -> None:
        svc = _make_service()
        result = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders")

        # Outgoing: orders.customer_id → customers.customer_id
        assert any("orders.customer_id → customers.customer_id" in r for r in result.relationships)
        # Incoming: order_items.order_id → orders.order_id
        assert any("order_items.order_id → orders.order_id" in r for r in result.relationships)

    async def test_common_joins_column_level(self) -> None:
        svc = _make_service()
        result = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders")

        assert any(
            "orders JOIN customers ON orders.customer_id = customers.customer_id" in j
            for j in result.common_joins
        )
        assert any(
            "orders JOIN order_items ON orders.order_id = order_items.order_id" in j
            for j in result.common_joins
        )

    async def test_column_focus(self) -> None:
        svc = _make_service()
        result = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders", column="total")
        assert result.column == "total"


class TestCaching:

    async def test_cache_hit_avoids_second_llm_call(self) -> None:
        cache = InMemoryCache(default_ttl=60)
        svc = _make_service(cache=cache)

        first = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders")
        assert first.cached is False
        assert svc._llm.complete.await_count == 1  # type: ignore[attr-defined]

        second = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders")
        assert second.cached is True
        assert svc._llm.complete.await_count == 1  # type: ignore[attr-defined]
        assert second.description == first.description

    async def test_cache_scoped_per_connection(self) -> None:
        """An explanation depends only on the connection's schema, so it is
        cached per connection: two connections never share an answer, while
        members of one (shared) connection reuse it."""
        cache = InMemoryCache(default_ttl=60)
        svc = _make_service(cache=cache)

        await svc.explain(connection_id="conn-a", user_id="user-a", table="orders")
        await svc.explain(connection_id="conn-b", user_id="user-b", table="orders")
        assert svc._llm.complete.await_count == 2  # type: ignore[attr-defined]

        await svc.explain(connection_id="conn-a", user_id="user-c", table="orders")
        assert svc._llm.complete.await_count == 2  # type: ignore[attr-defined]


class TestFallback:

    async def test_llm_failure_returns_derived_parts(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("provider down"))
        svc = _make_service(llm=llm)

        result = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders")

        assert result.llm_generated is False
        # Derived, factual parts are still present.
        assert result.relationships
        assert result.common_joins
        assert result.example_sql == "SELECT * FROM orders LIMIT 10;"
        # Falls back to the catalog table description.
        assert result.description == "Customer orders."

    async def test_unparseable_llm_output_falls_back(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content="not json at all", prompt_tokens=1, completion_tokens=1
            )
        )
        svc = _make_service(llm=llm)

        result = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders")
        assert result.llm_generated is False
        assert result.relationships

    async def test_vector_store_failure_is_non_fatal(self) -> None:
        vs = MagicMock()
        vs.get_chunks_by_table_names = AsyncMock(side_effect=Exception("qdrant down"))
        svc = _make_service(vector_store=vs)

        result = await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders")
        # Still produces an explanation (LLM enrichment ran without chunk context).
        assert result.table == "orders"


class TestNotFound:

    async def test_unknown_table_raises(self) -> None:
        svc = _make_service()
        with pytest.raises(TableNotFoundError):
            await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="does_not_exist")

    async def test_unknown_column_raises(self) -> None:
        svc = _make_service()
        with pytest.raises(TableNotFoundError):
            await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="orders", column="nope")

    async def test_llm_not_called_for_unknown_table(self) -> None:
        svc = _make_service()
        with pytest.raises(TableNotFoundError):
            await svc.explain(connection_id=CONNECTION_ID, user_id=USER_ID, table="ghost")
        svc._llm.complete.assert_not_awaited()  # type: ignore[attr-defined]
