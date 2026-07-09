"""Qdrant vector store — native hybrid search (dense + sparse BM42)."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from nl_to_sql.core.exceptions import VectorStoreError
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)

# Reserved UUID for the internal schema-hash sentinel point
_SCHEMA_HASH_UUID = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))


def _exc_detail(exc: Exception) -> str:
    """Return a non-empty string describing exc, even for qdrant ResponseHandlingException."""
    # ResponseHandlingException wraps the real cause in .source
    source = getattr(exc, "source", None)
    if source is not None:
        return f"{type(exc).__name__}({type(source).__name__}: {source})"
    msg = str(exc)
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _to_uuid(chunk_id: str) -> str:
    """Deterministic UUID from an arbitrary chunk_id string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _no_hash_filter(user_id: str | None = None) -> Filter:
    """Base filter that excludes the hash sentinel, optionally scoped to a user."""
    must = []
    if user_id is not None:
        must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    return Filter(
        must=must or None,
        must_not=[FieldCondition(key="type", match=MatchValue(value="schema_hash"))],
    )


def _payload_to_chunk(payload: dict[str, Any]) -> SchemaChunk:
    return SchemaChunk(
        chunk_id=payload.get("chunk_id", ""),
        table_name=payload.get("table_name", ""),
        schema_name=payload.get("schema_name", "public"),
        content=payload.get("content", ""),
        metadata={
            k: v
            for k, v in payload.items()
            if k not in {"chunk_id", "table_name", "schema_name", "content", "type"}
        },
    )


class QdrantVectorStore(IVectorStore):  # type: ignore[misc]
    """Qdrant-backed vector store with native hybrid search.

    Each document is stored with two named vectors:
      - "dense":  cosine embedding (passed in via SchemaChunk.embedding — provider-agnostic)
      - "sparse": BM42 sparse encoding of SchemaChunk.content (computed via fastembed)

    Hybrid retrieval fires both searches in a single Qdrant Query API call and
    fuses results with RRF (Reciprocal Rank Fusion), replacing both ChromaDB
    (dense-only) and the BM25 pickle index.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "schema_chunks",
        dimensions: int = 384,
    ) -> None:
        client_kwargs: dict[str, Any] = {
            "url": url,
            "timeout": 30,  # seconds — prevents stale-connection hangs
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = AsyncQdrantClient(**client_kwargs)
        self._collection_name = collection_name
        self._dimensions = dimensions
        self._sparse_model: Any = None
        self._schema_hash: str | None = None
        self._initialized = False
        logger.info("QdrantVectorStore created", url=url, collection=collection_name)

    # ── Sparse model ──────────────────────────────────────────────────────────

    def _get_sparse_model(self) -> Any:
        if self._sparse_model is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is required for Qdrant sparse encoding. "
                    "Install with: pip install 'qdrant-client[fastembed]'"
                ) from exc
            self._sparse_model = SparseTextEmbedding(
                model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
            )
        return self._sparse_model

    def _embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        model = self._get_sparse_model()
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in model.embed(texts)
        ]

    def _embed_sparse_query(self, text: str) -> SparseVector:
        model = self._get_sparse_model()
        e = next(iter(model.query_embed(text)))
        return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

    # ── Lazy initialization ───────────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        exists = await self._client.collection_exists(self._collection_name)
        if not exists:
            try:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self._dimensions,
                            distance=Distance.COSINE,
                            hnsw_config=HnswConfigDiff(on_disk=False),
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                        ),
                    },
                )
                logger.info("Qdrant collection created", collection=self._collection_name)
            except Exception as exc:
                # With multiple workers (WEB_CONCURRENCY > 1) another worker may have
                # created the collection between our collection_exists() check and here.
                # Treat "already exists" / 409 as success; re-raise everything else.
                if "already exists" in str(exc).lower():
                    logger.debug(
                        "Collection created by concurrent worker — continuing",
                        collection=self._collection_name,
                    )
                else:
                    raise
        else:
            # Load persisted schema hash into memory cache
            try:
                records = await self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=[_SCHEMA_HASH_UUID],
                    with_payload=True,
                )
                if records:
                    self._schema_hash = (records[0].payload or {}).get("hash")
            except Exception:
                pass

        # Ensure payload indexes exist (idempotent — safe to call on existing collections)
        for field in ("table_name", "type", "user_id"):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        self._initialized = True

    async def _persist_schema_hash(self, schema_hash: str) -> None:
        await self._ensure_initialized()
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=_SCHEMA_HASH_UUID,
                    vector={
                        "dense": [0.0] * self._dimensions,
                        "sparse": SparseVector(indices=[0], values=[0.001]),
                    },
                    payload={"hash": schema_hash, "type": "schema_hash"},
                )
            ],
        )

    # ── IVectorStore ──────────────────────────────────────────────────────────

    async def upsert(self, chunks: list[SchemaChunk], user_id: str | None = None) -> None:
        await self._ensure_initialized()
        valid = [c for c in chunks if c.embedding is not None]
        if not valid:
            return

        texts = [c.content for c in valid]
        try:
            sparse_vecs = await asyncio.to_thread(self._embed_sparse, texts)
        except Exception as exc:
            raise VectorStoreError(
                f"Sparse (BM42) embedding failed — the fastembed ONNX model may not be "
                f"downloaded yet. Run the server once with internet access to auto-download. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

        points = [
            PointStruct(
                id=_to_uuid(c.chunk_id),
                vector={"dense": c.embedding, "sparse": sv},
                payload={
                    "chunk_id": c.chunk_id,
                    "table_name": c.table_name,
                    "schema_name": c.schema_name,
                    "content": c.content,
                    "type": "chunk",
                    **({"user_id": user_id} if user_id is not None else {}),
                    **c.metadata,
                },
            )
            for c, sv in zip(valid, sparse_vecs, strict=True)
        ]

        try:
            batch = 100
            for i in range(0, len(points), batch):
                await self._client.upsert(
                    collection_name=self._collection_name,
                    points=points[i : i + batch],
                )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {_exc_detail(exc)}") from exc
        logger.debug("Qdrant upsert", n=len(points))

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        await self._ensure_initialized()
        no_hash = _no_hash_filter(user_id)
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            using="dense",
            query_filter=no_hash,
            limit=top_k,
            with_payload=True,
        )
        return [_payload_to_chunk(p.payload or {}) for p in response.points]

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Dense + BM42 sparse search fused with Qdrant's RRF."""
        await self._ensure_initialized()
        no_hash = _no_hash_filter(user_id)
        sparse_q = await asyncio.to_thread(self._embed_sparse_query, query_text)
        response = await self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=top_k * 2,
                    filter=no_hash,
                ),
                Prefetch(
                    query=sparse_q,
                    using="sparse",
                    limit=top_k * 2,
                    filter=no_hash,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=no_hash,
            limit=top_k,
            with_payload=True,
        )
        return [_payload_to_chunk(p.payload or {}) for p in response.points]

    async def delete_collection(self) -> None:
        try:
            await self._client.delete_collection(self._collection_name)
            logger.info("Qdrant collection deleted", collection=self._collection_name)
        except Exception as exc:
            logger.warning("Qdrant delete_collection failed", error=_exc_detail(exc))
        # Reset state so the next upsert/search triggers lazy re-initialization.
        # Do NOT call _ensure_initialized() here — it would cause a 409 race when
        # multiple workers call delete_collection() + upsert() concurrently.
        self._initialized = False
        self._schema_hash = None

    async def count(self, user_id: str | None = None) -> int:
        await self._ensure_initialized()
        result = await self._client.count(
            collection_name=self._collection_name,
            count_filter=_no_hash_filter(user_id),
            exact=True,
        )
        return result.count

    async def delete_by_user(self, user_id: str) -> None:
        """Delete all of a user's chunks (per-user reset). Best-effort."""
        await self._ensure_initialized()
        try:
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
            )
            logger.info("Qdrant deleted user chunks", user_id=user_id)
        except Exception as exc:
            logger.warning("Qdrant delete_by_user failed", error=_exc_detail(exc))

    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def get_chunks_by_table_names(
        self, table_names: list[str], user_id: str | None = None
    ) -> list[SchemaChunk]:
        await self._ensure_initialized()
        if not table_names:
            return []
        match = (
            MatchAny(any=table_names)
            if len(table_names) > 1
            else MatchValue(value=table_names[0])
        )
        must = [FieldCondition(key="table_name", match=match)]
        if user_id is not None:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        f = Filter(must=must)
        chunks: list[SchemaChunk] = []
        offset = None
        while True:
            records, next_offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=f,
                with_payload=True,
                limit=500,
                offset=offset,
            )
            chunks.extend(_payload_to_chunk(r.payload or {}) for r in records)
            if next_offset is None:
                break
            offset = next_offset
        return chunks

    async def get_all_table_names(self, user_id: str | None = None) -> list[str]:
        await self._ensure_initialized()
        no_hash = _no_hash_filter(user_id)
        names: set[str] = set()
        offset = None
        while True:
            records, next_offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=no_hash,
                with_payload=["table_name"],
                limit=500,
                offset=offset,
            )
            for r in records:
                name = (r.payload or {}).get("table_name", "")
                if name:
                    names.add(name)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(names)

    # ── Schema hash (duck-typed, not in IVectorStore) ─────────────────────────

    def get_schema_hash(self) -> str | None:
        return self._schema_hash

    def update_schema_hash(self, schema_hash: str) -> None:
        self._schema_hash = schema_hash
        try:
            loop = asyncio.get_running_loop()
            _task = loop.create_task(self._persist_schema_hash(schema_hash))
            _task.add_done_callback(lambda t: None)  # prevent GC
        except RuntimeError:
            # No running event loop — hash lives in memory only; next restart re-ingests
            pass
