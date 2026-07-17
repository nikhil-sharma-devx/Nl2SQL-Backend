"""Schema ingestion service — parses DB schema and populates the vector store.

NOTE: This module is kept for backward compatibility. New code should prefer
the ``rag.ingestion.pipeline.IngestionPipeline`` which provides the full
six-step ingestion flow (load → build → chunk → embed → BM25 → vector write).

The ``build_schema_from_dict`` static method delegates to
``rag.ingestion.schema_loader.SchemaLoader.build_schema_from_dict``.
"""
from typing import Any

import structlog

from nl_to_sql.core.exceptions import SchemaIngestionError
from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import ColumnInfo, SchemaChunk, SchemaMetadata, TableInfo
from nl_to_sql.rag.ingestion.chunker import build_column_child_chunks
from nl_to_sql.rag.ingestion.doc_builder import DocBuilder  # noqa: F401
from nl_to_sql.rag.ingestion.schema_loader import SchemaLoader  # noqa: F401
from nl_to_sql.rag.ingestion.table_describer import TableDescriber

logger = structlog.get_logger(__name__)


class SchemaIngestionService:
    """Orchestrates loading a database schema into the vector store.

    Flow:
      1. Accept SchemaMetadata (parsed tables + columns).
      2. Convert each table into a readable text chunk.
      3. Embed all chunks in one batch.
      4. Upsert into the vector store.

    SOLID:
      S — Only handles the ingestion pipeline step.
      D — Depends on IEmbedder and IVectorStore abstractions.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        per_user_isolation: bool = False,
        describer: TableDescriber | None = None,
        descriptions_enabled: bool = False,
        parent_child_enabled: bool = False,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._per_user_isolation = per_user_isolation
        # P1 — optional LLM description enrichment at ingest time.
        self._describer = describer
        self._descriptions_enabled = descriptions_enabled
        # P4 — optional parent-child (column-level) chunking.
        self._parent_child_enabled = parent_child_enabled

    def apply_runtime_flags(
        self, *, descriptions_enabled: bool, parent_child_enabled: bool
    ) -> None:
        """Refresh the P1/P4 ingest toggles from live settings before a re-ingest.

        This service is captured by long-lived singletons (the schema catalog and
        monitor), so its construction-time flags would otherwise ignore runtime
        changes made via ``PUT /config/rag``. Callers refresh these immediately
        before ``ingest`` so the "re-ingest to take effect" contract holds without
        a server restart — mirrors how ``get_request_orchestrator`` refreshes the
        retriever's runtime flags.
        """
        self._descriptions_enabled = descriptions_enabled
        self._parent_child_enabled = parent_child_enabled

    async def ingest(
        self, schema: SchemaMetadata, reset: bool = False, user_id: str | None = None
    ) -> int:
        """Ingest a SchemaMetadata object into the vector store.

        Args:
            schema: Parsed schema with all tables and columns.
            reset: If True, clears existing chunks before ingesting.
            user_id: When provided (per-user isolation), each chunk is tagged
                with this owner and its id is namespaced so it never collides
                with another user's chunks. ``reset`` then clears only this
                user's chunks (via ``delete_by_user`` if the store supports it)
                instead of the whole collection. ``None`` preserves the shared
                single-collection behaviour.

        Returns:
            Number of chunks ingested.

        Raises:
            SchemaIngestionError: On embedding or vector store errors.
        """
        log = logger.bind(database=schema.database_name, dialect=schema.dialect, user_id=user_id)
        log.info("Starting schema ingestion", table_count=len(schema.tables))

        if reset:
            if user_id is not None and hasattr(self._vector_store, "delete_by_user"):
                log.info("Resetting user's schema chunks")
                await self._vector_store.delete_by_user(user_id)
            elif user_id is None:
                # A shared re-ingest must not destroy per-user chunks when
                # isolation is active — delete only the shared/un-tagged chunks.
                if self._per_user_isolation and hasattr(self._vector_store, "delete_shared"):
                    log.info("Resetting shared schema chunks (preserving per-user)")
                    await self._vector_store.delete_shared()
                else:
                    log.info("Resetting vector store collection")
                    await self._vector_store.delete_collection()

        # P1 — enrich tables with LLM-generated NL descriptions before chunking
        # so the description is embedded alongside the DDL.
        if self._descriptions_enabled and self._describer is not None:
            try:
                await self._describer.enrich(schema.tables)
            except Exception as exc:
                log.warning("Table description enrichment failed — continuing", error=str(exc))

        chunks = [self._table_to_chunk(table) for table in schema.tables]
        if user_id is not None:
            for chunk in chunks:
                chunk.chunk_id = f"{user_id}:{chunk.chunk_id}"

        # P4 — derive column-level child chunks from each (namespaced) parent so
        # fine-grained retrieval hits still resolve to the full table DDL.
        if self._parent_child_enabled:
            child_chunks: list[SchemaChunk] = []
            for parent in chunks:
                child_chunks.extend(build_column_child_chunks(parent))
            chunks.extend(child_chunks)
            log.info("Parent-child chunking applied", children=len(child_chunks))

        try:
            texts = [c.content for c in chunks]
            embeddings = await self._embedder.embed_batch(texts)
        except Exception as exc:
            raise SchemaIngestionError(
                f"Failed to embed schema chunks: {exc}"
            ) from exc

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding

        try:
            await self._vector_store.upsert(chunks, user_id=user_id)

            # Store the hash in the vector store (global hash — only meaningful
            # for the shared-collection path; skipped for per-user ingestion).
            if user_id is None and hasattr(self._vector_store, 'update_schema_hash'):
                schema_hash = self.compute_schema_hash(schema)
                self._vector_store.update_schema_hash(schema_hash)
                log.info("Schema hash stored for change detection", hash=schema_hash[:16])
        except Exception as exc:
            raise SchemaIngestionError(
                f"Failed to upsert chunks into vector store: {type(exc).__name__}: {exc}"
            ) from exc

        log.info("Schema ingestion complete", chunks_ingested=len(chunks))
        return len(chunks)

    @staticmethod
    def _table_to_chunk(table: TableInfo) -> SchemaChunk:
        """Convert a TableInfo into a descriptive text SchemaChunk.

        The text representation is human-readable and designed to give the LLM
        precise context about each table's structure and relationships.
        """
        lines: list[str] = [
            # Use unqualified name for default schemas so the LLM generates
            # clean SQL without schema prefix (e.g. FROM categories, not FROM public.categories)
            f"Table: {table.name}",
        ]
        if table.description:
            lines.append(f"Description: {table.description}")

        lines.append("Columns:")
        for col in table.columns:
            parts = [f"  - {col.name} ({col.data_type})"]
            if col.primary_key:
                parts.append("[PRIMARY KEY]")
            if col.foreign_key:
                parts.append(f"[FK → {col.foreign_key}]")
            if not col.nullable:
                parts.append("[NOT NULL]")
            if col.description:
                parts.append(f"— {col.description}")
            lines.append(" ".join(parts))

        pk_cols = [c.name for c in table.columns if c.primary_key]
        if pk_cols:
            lines.append(f"Primary Key: ({', '.join(pk_cols)})")

        fk_cols = [(c.name, c.foreign_key) for c in table.columns if c.foreign_key]
        if fk_cols:
            for col_name, fk_ref in fk_cols:
                lines.append(f"Foreign Key: {col_name} → {fk_ref}")

        content = "\n".join(lines)
        return SchemaChunk(
            chunk_id=f"{table.schema_name}.{table.name}",
            table_name=table.name,
            schema_name=table.schema_name,
            content=content,
        )

    @staticmethod
    def compute_schema_hash(schema: SchemaMetadata) -> str:
        """Compute a deterministic hash for a schema to detect changes."""
        import hashlib
        import json

        # Sort tables and columns to ensure deterministic hashing
        sorted_tables = sorted(schema.tables, key=lambda t: t.name)

        schema_dict = {
            "database_name": schema.database_name,
            "dialect": schema.dialect,
            "tables": [
                {
                    "name": t.name,
                    "schema_name": t.schema_name,
                    "description": t.description,
                    "columns": [
                        {
                            "name": c.name,
                            "data_type": c.data_type,
                            "primary_key": c.primary_key,
                            "foreign_key": c.foreign_key,
                            "nullable": c.nullable
                        }
                        for c in sorted(t.columns, key=lambda c: c.name)
                    ]
                }
                for t in sorted_tables
            ]
        }
        return hashlib.sha256(
            json.dumps(schema_dict, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def build_schema_from_dict(raw: dict[str, Any]) -> SchemaMetadata:
        """Construct a SchemaMetadata from a plain Python dict.

        Useful for loading schema from a JSON config file or DB reflection.

        Args:
            raw: Dict with keys: database_name, dialect, tables (list).

        Returns:
            A SchemaMetadata instance.
        """
        tables: list[TableInfo] = []
        for t in raw.get("tables", []):
            columns = [ColumnInfo(**c) for c in t.get("columns", [])]
            tables.append(
                TableInfo(
                    name=t["name"],
                    schema_name=t.get("schema_name", "public"),
                    columns=columns,
                    description=t.get("description"),
                )
            )
        return SchemaMetadata(
            database_name=raw["database_name"],
            dialect=raw["dialect"],
            tables=tables,
        )
