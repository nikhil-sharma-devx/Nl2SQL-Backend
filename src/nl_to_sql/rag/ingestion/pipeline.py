"""Ingestion pipeline — orchestrates the full offline indexing flow.

Entry point: load → build docs → chunk → embed → BM25 index → vector write.
This is the single function you call to trigger a full re-index.
"""
import structlog

from nl_to_sql.core.exceptions import SchemaIngestionError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaMetadata
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.rag.ingestion.chunker import Chunker
from nl_to_sql.rag.ingestion.doc_builder import DocBuilder
from nl_to_sql.rag.ingestion.embedder import IngestionEmbedder
from nl_to_sql.rag.ingestion.schema_loader import SchemaLoader
from nl_to_sql.rag.ingestion.vector_writer import VectorWriter

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """Orchestrates the full schema ingestion pipeline.

    Pipeline steps (in order):
      1. schema_loader  — Reflect schema from the live database.
      2. doc_builder    — Convert tables into text chunks.
      3. chunker        — Split large documents (configurable strategy).
      4. embedder       — Compute dense embeddings for each chunk.
      5. bm25_indexer   — Build BM25 sparse index (optional).
      6. vector_writer  — Upsert into the vector database.

    SOLID:
      S — Only orchestrates; delegates all work to specialised modules.
      O — New steps can be added without modifying existing ones.
      D — Depends on abstractions (IEmbedder, IVectorStore).
    """

    def __init__(
        self,
        db_client: AsyncDatabaseClient,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        bm25_store: object | None = None,
        chunk_strategy: str = "table",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_batch_size: int = 32,
        bm25_enabled: bool = False,
    ) -> None:
        self._schema_loader = SchemaLoader(db_client)
        self._doc_builder = DocBuilder()
        self._chunker = Chunker(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._embedder = IngestionEmbedder(
            embedder=embedder,
            batch_size=embedding_batch_size,
        )
        self._vector_writer = VectorWriter(vector_store)
        self._bm25_enabled = bm25_enabled
        self._bm25_store = bm25_store

        # Lazy init BM25 indexer only when needed
        self._bm25_indexer = None

    async def run(
        self,
        schema_name: str = "public",
        reset: bool = True,
    ) -> int:
        """Run the full ingestion pipeline from database reflection.

        Args:
            schema_name: Database schema to reflect.
            reset: Clear vector store before ingesting.

        Returns:
            Number of chunks ingested.
        """
        log = logger.bind(schema=schema_name)
        log.info("Starting ingestion pipeline")

        # Step 1: Load schema
        schema = await self._schema_loader.load(schema_name)
        return await self.run_from_schema(schema, reset=reset)

    async def run_from_schema(
        self,
        schema: SchemaMetadata,
        reset: bool = True,
    ) -> int:
        """Run the ingestion pipeline from a pre-loaded SchemaMetadata.

        Args:
            schema: Pre-loaded schema metadata.
            reset: Clear vector store before ingesting.

        Returns:
            Number of chunks ingested.
        """
        log = logger.bind(
            database=schema.database_name,
            table_count=len(schema.tables),
        )

        # Step 2: Build documents
        log.info("Step 2: Building document chunks")
        documents = self._doc_builder.build_chunks(schema.tables)

        # Step 3: Chunk
        log.info("Step 3: Running chunker")
        chunks = self._chunker.chunk(documents)

        # Step 4: Embed
        log.info("Step 4: Embedding chunks")
        embedded_chunks = await self._embedder.embed_chunks(chunks)

        # Step 5: BM25 index (optional)
        if self._bm25_enabled and self._bm25_store is not None:
            log.info("Step 5: Building BM25 index")
            try:
                from nl_to_sql.rag.ingestion.bm25_indexer import BM25Indexer
                if self._bm25_indexer is None:
                    self._bm25_indexer = BM25Indexer(self._bm25_store)
                self._bm25_indexer.build_index(chunks)
            except Exception as exc:
                log.warning("BM25 indexing failed — skipping", error=str(exc))
        else:
            log.debug("BM25 indexing skipped (disabled or no store)")

        # Step 6: Write to vector store
        log.info("Step 6: Writing to vector store")
        count = await self._vector_writer.write(
            embedded_chunks, schema=schema, reset=reset
        )

        log.info(
            "Ingestion pipeline complete",
            chunks_ingested=count,
            tables=schema.table_names,
        )
        return count
