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
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
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


def _user_scope_should(connection_id: str | None) -> list[Any] | None:
    """OR-conditions scoping reads to a user's own chunks *plus* shared ones.

    When per-user isolation is active the caller passes the authenticated
    ``connection_id``; a point is then visible if it is tagged with that user
    (``connection_id == X``) **or** is un-tagged/shared (the ``connection_id`` payload is
    missing — e.g. the default database schema ingested globally at startup).
    Returns ``None`` when no user scoping is requested (shared-only behaviour).
    """
    if connection_id is None:
        return None
    return [
        FieldCondition(key="connection_id", match=MatchValue(value=connection_id)),
        IsEmptyCondition(is_empty=PayloadField(key="connection_id")),
    ]


def _no_hash_filter(
    connection_id: str | None = None,
    reserved_types: tuple[str, ...] = ("schema_hash",),
) -> Filter:
    """Base read filter: excludes reserved non-schema point types, optionally
    scoped to a user.

    ``reserved_types`` are point ``type``s that must never surface in a schema
    read/count — always the ``schema_hash`` sentinel, plus ``semantic_cache`` on
    the schema collection so any legacy cache points written there (before the
    cache got its own collection) stay invisible to retrieval and the chunk
    count. The cache's *own* store keeps the default (``schema_hash`` only) so
    its ``semantic_cache`` points remain searchable.

    With a ``connection_id`` the read matches the user's own chunks OR shared
    (un-tagged) chunks via ``should`` (at-least-one-must-match), so users on
    the default shared database still retrieve its schema.
    """
    type_match = (
        MatchAny(any=list(reserved_types))
        if len(reserved_types) > 1
        else MatchValue(value=reserved_types[0])
    )
    return Filter(
        should=_user_scope_should(connection_id),
        must_not=[FieldCondition(key="type", match=type_match)],
    )


def _dedupe_prefer_own(items: list[Any], requesting_user: str | None) -> list[Any]:
    """Collapse points sharing a ``(schema_name, table_name)`` to a single one,
    preferring the requesting user's own (``connection_id``-tagged) point over the
    shared (un-tagged) copy — "own-else-shared" visibility.

    With own-OR-shared reads a user who has re-embedded their own copy of a table
    also matches the shared copy, so both would be returned — wasting retrieval
    slots and double-weighting the table in fusion/reranking. This keeps the
    user's own chunk when present, else the shared one, preserving original order
    (i.e. rank). No-op when unscoped (``requesting_user is None``).

    Column-level child chunks (P4, ``is_child``) are passed through untouched:
    they share a ``table_name`` but are distinct columns and must not collapse.
    Operates on any object exposing a ``.payload`` dict (query points / scroll
    records).
    """
    if requesting_user is None:
        return items
    kept: list[Any] = []
    index_by_table: dict[tuple[str, str], int] = {}
    for item in items:
        payload = item.payload or {}
        if payload.get("is_child"):
            kept.append(item)
            continue
        table = payload.get("table_name", "")
        if not table:
            kept.append(item)
            continue
        key = (payload.get("schema_name", "public"), table)
        existing_idx = index_by_table.get(key)
        if existing_idx is None:
            index_by_table[key] = len(kept)
            kept.append(item)
        elif payload.get("connection_id") == requesting_user:
            existing_payload = kept[existing_idx].payload or {}
            if existing_payload.get("connection_id") != requesting_user:
                # Replace the shared duplicate with the user's own copy, keeping
                # the earlier (higher-ranked) position.
                kept[existing_idx] = item
    return kept


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
        non_schema_types: tuple[str, ...] = ("schema_hash",),
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
        # Point ``type``s excluded from every schema read/count. The schema store
        # adds ``semantic_cache`` here so stray cache points never leak into
        # retrieval; the cache's own store keeps only ``schema_hash``.
        self._reserved_types = non_schema_types
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
        for field in ("table_name", "type", "connection_id"):
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

    async def upsert(self, chunks: list[SchemaChunk], connection_id: str | None = None) -> None:
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
                    **({"connection_id": connection_id} if connection_id is not None else {}),
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
        connection_id: str | None = None,
    ) -> list[SchemaChunk]:
        await self._ensure_initialized()
        no_hash = _no_hash_filter(connection_id, self._reserved_types)
        # Over-fetch when user-scoped so own-else-shared dedup still yields top_k
        # (a table has at most a shared + an own copy, so 2x is sufficient).
        fetch_k = top_k * 2 if connection_id is not None else top_k
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            using="dense",
            query_filter=no_hash,
            limit=fetch_k,
            with_payload=True,
        )
        points = _dedupe_prefer_own(list(response.points), connection_id)[:top_k]
        return [_payload_to_chunk(p.payload or {}) for p in points]

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
        connection_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Dense + BM42 sparse search fused with Qdrant's RRF."""
        await self._ensure_initialized()
        no_hash = _no_hash_filter(connection_id, self._reserved_types)
        sparse_q = await asyncio.to_thread(self._embed_sparse_query, query_text)
        # Over-fetch when user-scoped so own-else-shared dedup still yields top_k.
        fetch_k = top_k * 2 if connection_id is not None else top_k
        response = await self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=fetch_k * 2,
                    filter=no_hash,
                ),
                Prefetch(
                    query=sparse_q,
                    using="sparse",
                    limit=fetch_k * 2,
                    filter=no_hash,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=no_hash,
            limit=fetch_k,
            with_payload=True,
        )
        points = _dedupe_prefer_own(list(response.points), connection_id)[:top_k]
        return [_payload_to_chunk(p.payload or {}) for p in points]

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

    async def count(self, connection_id: str | None = None) -> int:
        await self._ensure_initialized()
        result = await self._client.count(
            collection_name=self._collection_name,
            count_filter=_no_hash_filter(connection_id, self._reserved_types),
            exact=True,
        )
        return result.count

    async def delete_by_connection(self, connection_id: str) -> None:
        """Delete all of a user's chunks (per-user reset). Best-effort."""
        await self._ensure_initialized()
        try:
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="connection_id", match=MatchValue(value=connection_id))]
                ),
            )
            logger.info("Qdrant deleted user chunks", connection_id=connection_id)
        except Exception as exc:
            logger.warning("Qdrant delete_by_connection failed", error=_exc_detail(exc))

    async def delete_shared(self) -> None:
        """Delete only shared/un-tagged chunks, preserving per-user chunks.

        Used for the shared re-ingest (startup / schema-monitor / legacy refresh)
        when per-user isolation is active: dropping the whole collection would
        wipe every user's uploaded schema, so we delete just the un-tagged
        ``chunk`` points (``connection_id`` payload missing) and keep the hash
        sentinel and all per-user points intact. Best-effort.
        """
        await self._ensure_initialized()
        try:
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[IsEmptyCondition(is_empty=PayloadField(key="connection_id"))],
                    must_not=[
                        FieldCondition(key="type", match=MatchValue(value="schema_hash"))
                    ],
                ),
            )
            logger.info("Qdrant deleted shared (un-tagged) chunks")
        except Exception as exc:
            logger.warning("Qdrant delete_shared failed", error=_exc_detail(exc))

    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def get_chunks_by_table_names(
        self, table_names: list[str], connection_id: str | None = None
    ) -> list[SchemaChunk]:
        await self._ensure_initialized()
        if not table_names:
            return []
        match = (
            MatchAny(any=table_names)
            if len(table_names) > 1
            else MatchValue(value=table_names[0])
        )
        f = Filter(
            must=[FieldCondition(key="table_name", match=match)],
            should=_user_scope_should(connection_id),
        )
        all_records: list[Any] = []
        offset = None
        while True:
            records, next_offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=f,
                with_payload=True,
                limit=500,
                offset=offset,
            )
            all_records.extend(records)
            if next_offset is None:
                break
            offset = next_offset
        # Own-else-shared: a user with their own copy of a table would otherwise
        # get both it and the shared copy for the same table.
        deduped = _dedupe_prefer_own(all_records, connection_id)
        return [_payload_to_chunk(r.payload or {}) for r in deduped]

    async def get_all_table_names(self, connection_id: str | None = None) -> list[str]:
        await self._ensure_initialized()
        no_hash = _no_hash_filter(connection_id, self._reserved_types)
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
