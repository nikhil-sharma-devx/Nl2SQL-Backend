"""Ingestion sub-package — offline batch pipeline.

Runs when the database schema changes. Reads the schema, converts it into
text documents, embeds them, and writes them to the vector store and BM25
sparse index.
"""
from nl_to_sql.rag.ingestion.pipeline import IngestionPipeline
from nl_to_sql.rag.ingestion.schema_loader import SchemaLoader
from nl_to_sql.rag.ingestion.doc_builder import DocBuilder
from nl_to_sql.rag.ingestion.chunker import Chunker
from nl_to_sql.rag.ingestion.embedder import IngestionEmbedder
from nl_to_sql.rag.ingestion.bm25_indexer import BM25Indexer
from nl_to_sql.rag.ingestion.vector_writer import VectorWriter

__all__ = [
    "IngestionPipeline",
    "SchemaLoader",
    "DocBuilder",
    "Chunker",
    "IngestionEmbedder",
    "BM25Indexer",
    "VectorWriter",
]
