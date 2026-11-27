"""Unit tests for ChromaVectorStore async safety (H8).

Every method called the synchronous ``chromadb`` client directly with no
``asyncio.to_thread``, so any deployment with ``vector_store_provider !=
"qdrant"`` serialized all retrieval/ingestion behind blocking I/O. The fix
offloads every blocking ``self._collection``/``self._client`` call.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.infrastructure.vector_store.chroma_store import ChromaVectorStore


def _store() -> ChromaVectorStore:
    with patch("nl_to_sql.infrastructure.vector_store.chroma_store.chromadb.PersistentClient"):
        store = ChromaVectorStore(persist_dir="/tmp/does-not-matter")
    store._collection = MagicMock()
    store._client = MagicMock()
    return store


def _chunk() -> SchemaChunk:
    return SchemaChunk(
        chunk_id="t1",
        table_name="orders",
        schema_name="public",
        content="Table orders",
        embedding=[0.1, 0.2],
    )


class TestChromaStoreAsyncSafety:
    async def test_upsert_offloads_to_thread(self) -> None:
        store = _store()
        with patch(
            "nl_to_sql.infrastructure.vector_store.chroma_store.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as mock_to_thread:
            await store.upsert([_chunk()])
        mock_to_thread.assert_called_once()
        assert mock_to_thread.call_args.args[0] == store._collection.upsert

    async def test_similarity_search_offloads_to_thread(self) -> None:
        store = _store()
        store._collection.count.return_value = 5
        store._collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"table_name": "orders", "schema_name": "public"}]],
            "ids": [["t1"]],
        }
        with patch(
            "nl_to_sql.infrastructure.vector_store.chroma_store.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as mock_to_thread:
            result = await store.similarity_search(query_embedding=[0.1, 0.2], top_k=3)
        assert len(result) == 1
        assert result[0].table_name == "orders"
        # count() and query() are both offloaded.
        offloaded_callables = {c.args[0] for c in mock_to_thread.call_args_list}
        assert store._collection.count in offloaded_callables
        assert store._collection.query in offloaded_callables

    async def test_count_offloads_to_thread(self) -> None:
        store = _store()
        store._collection.count.return_value = 42
        with patch(
            "nl_to_sql.infrastructure.vector_store.chroma_store.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as mock_to_thread:
            result = await store.count()
        assert result == 42
        mock_to_thread.assert_called_once_with(store._collection.count)

    async def test_health_check_offloads_to_thread(self) -> None:
        store = _store()
        with patch(
            "nl_to_sql.infrastructure.vector_store.chroma_store.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as mock_to_thread:
            result = await store.health_check()
        assert result is True
        mock_to_thread.assert_called_once_with(store._collection.count)

    async def test_delete_collection_offloads_to_thread(self) -> None:
        store = _store()
        store._collection.get.return_value = {"ids": ["a", "b"]}
        store._collection.count.return_value = 0
        with patch(
            "nl_to_sql.infrastructure.vector_store.chroma_store.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as mock_to_thread:
            await store.delete_collection()
        offloaded_callables = {c.args[0] for c in mock_to_thread.call_args_list}
        assert store._collection.get in offloaded_callables
        assert store._collection.delete in offloaded_callables
        assert store._collection.count in offloaded_callables

    async def test_get_chunks_by_table_names_offloads_to_thread(self) -> None:
        store = _store()
        store._collection.get.return_value = {
            "documents": ["doc1"],
            "metadatas": [{"table_name": "orders", "schema_name": "public"}],
            "ids": ["t1"],
        }
        with patch(
            "nl_to_sql.infrastructure.vector_store.chroma_store.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as mock_to_thread:
            result = await store.get_chunks_by_table_names(["orders"])
        assert len(result) == 1
        mock_to_thread.assert_called_once()
        assert mock_to_thread.call_args.args[0] == store._collection.get

    async def test_get_all_table_names_offloads_to_thread(self) -> None:
        store = _store()
        store._collection.get.return_value = {
            "metadatas": [{"table_name": "orders"}, {"table_name": "customers"}]
        }
        with patch(
            "nl_to_sql.infrastructure.vector_store.chroma_store.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as mock_to_thread:
            result = await store.get_all_table_names()
        assert result == ["customers", "orders"]
        mock_to_thread.assert_called_once()
