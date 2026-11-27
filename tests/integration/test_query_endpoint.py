"""Integration tests for the /api/v1/query endpoint."""
from unittest.mock import patch

import pytest
from httpx import AsyncClient

from nl_to_sql.core.models.schema import SchemaChunk

VALID_SQL = "SELECT customer_id, email FROM customers LIMIT 10;"


@pytest.fixture
def patched_app(sample_chunks: list[SchemaChunk]) -> object:
    """App with mocked LLM/embedder/vector store for integration tests."""
    with (
        patch(
            "nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"
        )
    ):
        from nl_to_sql.api.app import create_app
        app = create_app()
        yield app


class TestQueryEndpoint:

    async def test_health_returns_200(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    async def test_query_requires_auth(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/query", json={})
        assert response.status_code == 401  # Auth required before body validation

    async def test_query_question_too_short_requires_auth(self, async_client: AsyncClient) -> None:
        response = await async_client.post("/api/v1/query", json={"question": "hi"})
        assert response.status_code == 401  # Auth required

    async def test_openapi_schema_accessible(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/v1/query" in schema["paths"]

    async def test_docs_accessible(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/docs")
        assert response.status_code == 200
