"""Unit tests for SemanticCache tenant isolation (C7).

``get_semantic`` used to only filter candidate hits by ``connection_id`` when
one was provided (`if connection_id: ...`); an unscoped caller (e.g. a
connection-resolution failure that fell back to ``connection_id=None``) fell
through unfiltered and could receive another tenant's cached answer. The fix
always filters, including when ``connection_id`` is ``None`` (matching only
other ``None``-tagged/shared entries).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.infrastructure.cache.semantic_cache import SemanticCache


def _chunk(connection_id: str | None, response: str = '{"sql": "SELECT 1"}') -> SchemaChunk:
    return SchemaChunk(
        chunk_id="semantic:x",
        table_name="_semantic_cache",
        schema_name="_cache",
        content="some question",
        metadata={
            "original_question": "some question",
            "response": response,
            "cached_at": 0.0,
            "ttl": 0,  # 0 disables the TTL-expiry check
            "type": "semantic_cache",
            "connection_id": connection_id,
            "distance": 0.0,  # similarity == 1.0
        },
    )


def _cache(similar_chunks: list[SchemaChunk]) -> SemanticCache:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2])
    vector_store = MagicMock()
    vector_store.similarity_search = AsyncMock(return_value=similar_chunks)
    exact_cache = MagicMock()
    return SemanticCache(embedder=embedder, vector_store=vector_store, exact_cache=exact_cache)


class TestSemanticCacheIsolation:
    async def test_unscoped_caller_does_not_see_other_tenants_entry(self) -> None:
        """The exact C7 exploit: a resolution failure (connection_id=None)
        must not surface tenant A's cached answer."""
        cache = _cache([_chunk(connection_id="tenant-a")])
        result = await cache.get_semantic("some question", connection_id=None)
        assert result is None

    async def test_unscoped_caller_sees_shared_entry(self) -> None:
        cache = _cache([_chunk(connection_id=None)])
        result = await cache.get_semantic("some question", connection_id=None)
        assert result == {"sql": "SELECT 1"}

    async def test_scoped_caller_does_not_see_other_tenants_entry(self) -> None:
        cache = _cache([_chunk(connection_id="tenant-b")])
        result = await cache.get_semantic("some question", connection_id="tenant-a")
        assert result is None

    async def test_scoped_caller_sees_own_entry(self) -> None:
        cache = _cache([_chunk(connection_id="tenant-a")])
        result = await cache.get_semantic("some question", connection_id="tenant-a")
        assert result == {"sql": "SELECT 1"}
