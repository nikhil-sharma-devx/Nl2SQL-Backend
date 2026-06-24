"""ChromaDB vector store — implements IVectorStore."""
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings

from nl_to_sql.core.exceptions import VectorStoreError
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


class ChromaVectorStore(IVectorStore):  # type: ignore[misc]
    """Persistent ChromaDB backed vector store for schema chunks.

    Stores and retrieves SchemaChunk objects using cosine similarity.

    SOLID:
      S — Only handles vector storage/retrieval.
      O — Other backends (FAISS, Pinecone) add functionality without changing
          this or its callers.
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "schema_chunks",
    ) -> None:
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialised",
            collection=collection_name,
            persist_dir=persist_dir,
        )

    def get_schema_hash(self) -> str | None:
        """Get the stored schema hash from a special metadata chunk.

        Returns:
            The schema hash if stored, None otherwise.
        """
        try:
            # Try to get the special schema_hash chunk
            result = self._collection.get(ids=["_schema_hash"])
            metadatas = result.get("metadatas")
            if result and metadatas and len(metadatas) > 0:
                return str(metadatas[0].get("hash")) if metadatas[0].get("hash") else None
            return None
        except Exception:
            return None

    def update_schema_hash(self, schema_hash: str) -> None:
        """Store the schema hash as a special metadata chunk.

        Args:
            schema_hash: The SHA256 hash of the current schema.
        """
        try:
            # Upsert a special chunk with the schema hash
            self._collection.upsert(
                ids=["_schema_hash"],
                documents=["Schema metadata hash"],
                metadatas=[{"hash": schema_hash, "type": "schema_hash"}]
            )
            logger.info("Stored schema hash in collection")
        except Exception as exc:
            logger.warning("Failed to store schema hash", error=str(exc))

    async def upsert(self, chunks: list[SchemaChunk]) -> None:
        """Upsert schema chunks into ChromaDB."""
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding is not None]
        metadatas = [
            {"table_name": c.table_name, "schema_name": c.schema_name, **c.metadata}
            for c in chunks
        ]
        try:
            # Retry up to 3 times in case of transient collection issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self._collection.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,  # type: ignore[arg-type]
                    )
                    logger.info("Upserted schema chunks", count=len(chunks))
                    return  # Success, exit retry loop
                except Exception as exc:
                    if "does not exist" in str(exc) and attempt < max_retries - 1:
                        # Collection doesn't exist, recreate it and retry
                        logger.warning(
                            "Collection not found during upsert, recreating...",
                            attempt=attempt + 1,
                            collection=self._collection_name
                        )
                        await self.delete_collection()
                        continue  # Retry
                    else:
                        # Max retries exceeded or different error
                        raise
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB upsert failed: {exc}") from exc

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SchemaChunk]:
        """Return top-k most similar schema chunks."""
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],  # type: ignore[arg-type]
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if "does not exist" in str(exc):
                # Collection might not exist, try to recreate it
                logger.warning(
                    "Collection not found during query, attempting to recreate",
                    collection=self._collection_name
                )
                try:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
                    logger.info("Collection recreated during query", collection=self._collection_name)
                    # Return empty results since collection is empty
                    return []
                except Exception as recreate_exc:
                    raise VectorStoreError(
                        f"ChromaDB query failed and collection recreation failed: {recreate_exc}"
                    ) from recreate_exc
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        chunks: list[SchemaChunk] = []
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        ids = results.get("ids") or [[]]

        for doc, meta, chunk_id in zip(documents[0], metadatas[0], ids[0], strict=True):
            chunks.append(
                SchemaChunk(
                    chunk_id=chunk_id,
                    table_name=meta.get("table_name", ""),
                    schema_name=meta.get("schema_name", "public"),
                    content=doc,
                    metadata={k: v for k, v in meta.items()
                               if k not in ("table_name", "schema_name")},
                )
            )
        return chunks

    async def delete_collection(self) -> None:
        """Clear all data from the collection without deleting it (preserves UUID)."""
        try:
            # Get all IDs in the collection
            all_data = self._collection.get()

            if all_data and all_data.get("ids"):
                # Delete all entries by their IDs
                self._collection.delete(ids=all_data["ids"])
                logger.info(
                    "Cleared all entries from collection",
                    collection=self._collection_name,
                    deleted_count=len(all_data["ids"])
                )
            else:
                logger.info(
                    "Collection is already empty",
                    collection=self._collection_name
                )

            # Verify collection is empty
            count = self._collection.count()
            logger.info(
                "Collection cleared successfully",
                collection=self._collection_name,
                remaining_count=count
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to clear collection: {exc}") from exc

    async def count(self) -> int:
        """Return the number of stored chunks."""
        try:
            return self._collection.count()
        except Exception as exc:
            # Collection might not exist or be corrupted
            logger.warning(
                "Collection count failed, attempting to recreate collection",
                collection=self._collection_name,
                error=str(exc)
            )
            # Try to recreate the collection
            try:
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "Collection recreated successfully",
                    collection=self._collection_name
                )
                return self._collection.count()
            except Exception as recreate_exc:
                raise VectorStoreError(
                    f"ChromaDB collection '{self._collection_name}' does not exist and cannot be created. "
                    f"Please run 'python scripts/ingest_schema.py' to initialize the schema. "
                    f"Original error: {exc}"
                ) from recreate_exc

    async def health_check(self) -> bool:
        """Verify ChromaDB is functional."""
        try:
            self._collection.count()
            return True
        except Exception:
            return False

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[SchemaChunk]:
        """Hybrid search — falls back to vector search for ChromaDB.

        ChromaDB doesn't support native BM25, so we use pure vector search.
        For true hybrid search, use QdrantVectorStore.

        Args:
            query_text: The original query text (unused in ChromaDB).
            query_embedding: The query vector for semantic search.
            top_k: Number of results to return.
            alpha: Weight parameter (unused in fallback).

        Returns:
            Ordered list of SchemaChunk from vector search.
        """
        # Fall back to pure vector search
        return await self.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    async def get_chunks_by_table_names(
        self,
        table_names: list[str],
    ) -> list[SchemaChunk]:
        """Fetch schema chunks for the specified table names via metadata filter.

        Uses ChromaDB's ``$in`` where-filter so no re-embedding is needed.

        Args:
            table_names: Exact table names to fetch.

        Returns:
            List of matching SchemaChunk objects.
        """
        if not table_names:
            return []
        try:
            where_filter: dict[str, Any] = (
                {"table_name": {"$in": table_names}}
                if len(table_names) > 1
                else {"table_name": table_names[0]}
            )
            result = self._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB get_chunks_by_table_names failed: {exc}"
            ) from exc

        chunks: list[SchemaChunk] = []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        ids = result.get("ids") or []

        for doc, meta, chunk_id in zip(documents, metadatas, ids, strict=True):
            # Skip the internal schema-hash sentinel record
            if chunk_id == "_schema_hash":
                continue
            chunks.append(
                SchemaChunk(
                    chunk_id=chunk_id,
                    table_name=meta.get("table_name", ""),
                    schema_name=meta.get("schema_name", "public"),
                    content=doc,
                    metadata={
                        k: v
                        for k, v in meta.items()
                        if k not in ("table_name", "schema_name")
                    },
                )
            )
        return chunks

    async def get_all_table_names(self) -> list[str]:
        """Return every unique table name currently stored in ChromaDB.

        Performs a full metadata scan — acceptable because the schema chunk
        count is always small (one chunk per table).

        Returns:
            Sorted list of unique table name strings.
        """
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB get_all_table_names failed: {exc}"
            ) from exc

        table_names: set[str] = set()
        for meta in result.get("metadatas") or []:
            name = meta.get("table_name", "")
            # Skip the internal hash sentinel and empty names
            if name and meta.get("type") != "schema_hash":
                table_names.add(str(name))
        return sorted(table_names)
