"""Unit tests for SchemaRetriever."""
from unittest.mock import MagicMock

import pytest

from nl_to_sql.core.exceptions import EmptySchemaError
from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.services.schema_retriever import SchemaRetriever


class TestSchemaRetriever:

    @pytest.fixture
    def retriever(
        self, mock_embedder: MagicMock, mock_vector_store: MagicMock
    ) -> SchemaRetriever:
        return SchemaRetriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            top_k=3,
        )

    async def test_retrieve_returns_chunks(
        self, retriever: SchemaRetriever, sample_chunks: list[SchemaChunk]
    ) -> None:
        chunks = await retriever.retrieve("Show me all customers")
        assert len(chunks) == len(sample_chunks)

    async def test_retrieve_raises_on_empty_store(
        self, mock_embedder: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.count.return_value = 0
        retriever = SchemaRetriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )
        with pytest.raises(EmptySchemaError):
            await retriever.retrieve("Count customers")

    def test_build_schema_context_formats_correctly(
        self, retriever: SchemaRetriever, sample_chunks: list[SchemaChunk]
    ) -> None:
        context = retriever.build_schema_context(sample_chunks)
        assert "RELEVANT DATABASE SCHEMA" in context
        assert "customers" in context
        assert "orders" in context

    def test_build_schema_context_empty(self, retriever: SchemaRetriever) -> None:
        context = retriever.build_schema_context([])
        assert "No relevant schema" in context
