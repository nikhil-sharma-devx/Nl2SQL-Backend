"""Vector writer — upserts embedded chunks into the vector database.

Takes the embedded chunks from the embedder step and writes them to the
vector store, attaching metadata (table name, column list, database ID).
"""
import hashlib
import json

import structlog

from nl_to_sql.core.exceptions import SchemaIngestionError
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk, SchemaMetadata

logger = structlog.get_logger(__name__)


class VectorWriter:
    """Writes embedded schema chunks to the vector database.

    SOLID:
      S — Only handles writing to the vector store.
      D — Depends on IVectorStore abstraction.
    """

    def __init__(
        self, vector_store: IVectorStore, per_user_isolation: bool = False
    ) -> None:
        self._vector_store = vector_store
        self._per_user_isolation = per_user_isolation

    async def write(
        self,
        chunks: list[SchemaChunk],
        schema: SchemaMetadata | None = None,
        reset: bool = False,
    ) -> int:
        """Upsert embedded chunks into the vector database.

        Args:
            chunks: Embedded schema chunks to store.
            schema: Optional schema metadata for hash computation.
            reset: If True, clears existing chunks before writing.

        Returns:
            Number of chunks written.

        Raises:
            SchemaIngestionError: On vector store write errors.
        """
        log = logger.bind(chunk_count=len(chunks), reset=reset)

        if reset:
            # Preserve per-user chunks on a shared re-ingest when isolation is on.
            if self._per_user_isolation and hasattr(self._vector_store, "delete_shared"):
                log.info("Resetting shared vector chunks (preserving per-user)")
                await self._vector_store.delete_shared()
            else:
                log.info("Resetting vector store collection")
                await self._vector_store.delete_collection()

        try:
            await self._vector_store.upsert(chunks)
        except Exception as exc:
            raise SchemaIngestionError(
                f"Failed to upsert chunks into vector store: {exc}"
            ) from exc

        # Store schema hash for change detection
        if schema is not None:
            self._store_schema_hash(schema)

        log.info("Chunks written to vector store", count=len(chunks))
        return len(chunks)

    def _store_schema_hash(self, schema: SchemaMetadata) -> None:
        """Compute and store a hash of the schema for change detection."""
        schema_dict = {
            "database_name": schema.database_name,
            "dialect": schema.dialect,
            "tables": [
                {
                    "name": t.name,
                    "schema_name": t.schema_name,
                    "columns": [
                        {"name": c.name, "data_type": c.data_type}
                        for c in t.columns
                    ],
                }
                for t in schema.tables
            ],
        }
        schema_hash = hashlib.sha256(
            json.dumps(schema_dict, sort_keys=True).encode()
        ).hexdigest()

        if hasattr(self._vector_store, "update_schema_hash"):
            self._vector_store.update_schema_hash(schema_hash)
            logger.info(
                "Schema hash stored for change detection", hash=schema_hash[:16]
            )
