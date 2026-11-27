"""Unit tests for Chunker (H1: infinite-loop guard on fixed-size chunking).

``_chunk_fixed_size``'s window advances by ``chunk_size - chunk_overlap``
each iteration. When ``chunk_overlap >= chunk_size`` that's <= 0 and the
synchronous ``while`` loop never terminates, freezing the whole asyncio
event loop. The fix clamps ``chunk_overlap`` at construction time so the
advance is always >= 1.
"""
from __future__ import annotations

from nl_to_sql.core.models.schema import SchemaChunk
from nl_to_sql.rag.ingestion.chunker import Chunker


def _long_doc(length: int = 250) -> SchemaChunk:
    return SchemaChunk(
        chunk_id="t1",
        table_name="orders",
        schema_name="public",
        content="x" * length,
    )


class TestChunkOverlapGuard:
    def test_overlap_equal_to_size_is_clamped(self) -> None:
        # chunk_size == chunk_overlap would advance by 0 — must terminate.
        chunker = Chunker(strategy="fixed", chunk_size=100, chunk_overlap=100)
        result = chunker.chunk([_long_doc(250)])
        assert len(result) > 0

    def test_overlap_greater_than_size_is_clamped(self) -> None:
        # Would previously advance by a NEGATIVE amount — freezes forever.
        chunker = Chunker(strategy="fixed", chunk_size=50, chunk_overlap=500)
        result = chunker.chunk([_long_doc(250)])
        assert len(result) > 0
        assert len(result) < 1000  # sanity bound — must not have looped indefinitely

    def test_normal_overlap_still_works(self) -> None:
        chunker = Chunker(strategy="fixed", chunk_size=100, chunk_overlap=20)
        result = chunker.chunk([_long_doc(250)])
        assert len(result) == 4  # ceil((250 - 20) / 80) windows, matches unclamped behavior
        assert chunker._chunk_overlap == 20  # untouched when valid
