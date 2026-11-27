"""Unit test for the shared BM25 tokenizer (Medium finding).

Regression test: bm25_indexer.py and bm25_retriever.py each defined an
identical ``_tokenise()`` regex independently, linked only by a comment
("must match ingestion") — a future edit to one without the other would
silently desync dense/sparse chunk IDs. Both now import
``infrastructure.bm25_store.tokenize_for_bm25``.
"""
from __future__ import annotations

from nl_to_sql.infrastructure.bm25_store import tokenize_for_bm25
from nl_to_sql.rag.ingestion.bm25_indexer import tokenize_for_bm25 as indexer_tokenize
from nl_to_sql.rag.retrieval.bm25_retriever import tokenize_for_bm25 as retriever_tokenize


class TestSharedTokenizer:
    def test_ingestion_and_retrieval_use_the_same_function(self) -> None:
        assert indexer_tokenize is tokenize_for_bm25
        assert retriever_tokenize is tokenize_for_bm25

    def test_tokenizes_identifiers_case_insensitively(self) -> None:
        assert tokenize_for_bm25("Table Orders_2024") == ["table", "orders_2024"]

    def test_strips_punctuation(self) -> None:
        assert tokenize_for_bm25("customer_id (INT) [FK]") == ["customer_id", "int", "fk"]
