"""Retrieval sub-package — online per-query pipeline.

Runs on every user query. Embeds the user's question, searches both the
vector store and BM25 index, reranks the results, and returns the most
relevant schema chunks formatted as LLM context.
"""
from nl_to_sql.rag.retrieval.bm25_retriever import BM25Retriever
from nl_to_sql.rag.retrieval.context_builder import ContextBuilder
from nl_to_sql.rag.retrieval.fk_extractor import FKRelationshipExtractor
from nl_to_sql.rag.retrieval.query_embedder import QueryEmbedder
from nl_to_sql.rag.retrieval.reranker import Reranker
from nl_to_sql.rag.retrieval.retrieval_chain import RetrievalChain
from nl_to_sql.rag.retrieval.table_selector import TableSelectorService
from nl_to_sql.rag.retrieval.vector_retriever import VectorRetriever

__all__ = [
    "BM25Retriever",
    "ContextBuilder",
    "FKRelationshipExtractor",
    "QueryEmbedder",
    "Reranker",
    "RetrievalChain",
    "TableSelectorService",
    "VectorRetriever",
]
