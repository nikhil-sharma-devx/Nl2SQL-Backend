"""Pytest configuration and shared fixtures."""
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from nl_to_sql.api.app import create_app
from nl_to_sql.core.models.schema import (
    ColumnInfo,
    SchemaChunk,
    SchemaMetadata,
    TableInfo,
)
from nl_to_sql.core.models.sql_result import LLMResponse
from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_schema() -> SchemaMetadata:
    """Minimal e-commerce schema for tests."""
    return SchemaMetadata(
        database_name="test_db",
        dialect="postgresql",
        tables=[
            TableInfo(
                name="customers",
                columns=[
                    ColumnInfo(name="customer_id", data_type="INTEGER", primary_key=True),
                    ColumnInfo(name="email", data_type="TEXT"),
                    ColumnInfo(name="country", data_type="TEXT"),
                ],
            ),
            TableInfo(
                name="orders",
                columns=[
                    ColumnInfo(name="order_id", data_type="INTEGER", primary_key=True),
                    ColumnInfo(name="customer_id", data_type="INTEGER",
                               foreign_key="customers.customer_id"),
                    ColumnInfo(name="total_amount", data_type="REAL"),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_chunks() -> list[SchemaChunk]:
    """Pre-built schema chunks with dummy embeddings."""
    dummy_embedding = [0.1] * 1536
    return [
        SchemaChunk(
            chunk_id="main.customers",
            table_name="customers",
            schema_name="main",
            content="Table: main.customers\nColumns:\n  - customer_id (INTEGER) [PRIMARY KEY]",
            embedding=dummy_embedding,
        ),
        SchemaChunk(
            chunk_id="main.orders",
            table_name="orders",
            schema_name="main",
            content="Table: main.orders\nColumns:\n  - order_id (INTEGER) [PRIMARY KEY]",
            embedding=dummy_embedding,
        ),
    ]


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Mock LLM provider that returns a canned SQL response."""
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="SELECT customer_id, email FROM customers LIMIT 10;",
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder that returns dummy vectors."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
    embedder.dimensions = 1536
    return embedder


@pytest.fixture
def mock_vector_store(sample_chunks: list[SchemaChunk]) -> MagicMock:
    """Mock vector store that returns predefined chunks."""
    store = MagicMock()
    store.similarity_search = AsyncMock(return_value=sample_chunks)
    store.upsert = AsyncMock()
    store.count = AsyncMock(return_value=5)
    store.delete_collection = AsyncMock()
    store.health_check = AsyncMock(return_value=True)
    return store


@pytest.fixture
def in_memory_cache() -> InMemoryCache:
    return InMemoryCache(default_ttl=60)


@pytest.fixture
def app() -> Any:
    """FastAPI test application (no real LLM or vector store needed)."""
    return create_app()


@pytest.fixture
def sync_client(app: Any) -> TestClient:
    return TestClient(app)


@pytest.fixture
async def async_client(app: Any) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
