"""Example store — P2: vector index of successful NL→SQL pairs for few-shot.

Successful (thumbs-up / high-score) natural-language → SQL pairs are embedded
and stored in a dedicated Qdrant collection. At query time the most *similar*
past pairs are retrieved and injected into the SQL-generation prompt as concrete
few-shot examples, which is far more useful than the most *recent* pairs.

The store is defensive: every operation is wrapped so a Qdrant outage (or a dev
environment without Qdrant) degrades gracefully to "no examples" rather than
breaking the query pipeline.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.infrastructure.vector_store.tenant_scope import own_or_shared_should

logger = structlog.get_logger(__name__)


class ExampleStore:
    """Dedicated Qdrant collection of NL→SQL examples for few-shot retrieval.

    SOLID:
      S — Only stores/retrieves example pairs.
      D — Depends on the IEmbedder abstraction for vectors.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "query_examples",
        dimensions: int = 384,
    ) -> None:
        client_kwargs: dict[str, Any] = {"url": url, "timeout": 30}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = AsyncQdrantClient(**client_kwargs)
        self._embedder = embedder
        self._collection_name = collection_name
        self._dimensions = dimensions
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        exists = await self._client.collection_exists(self._collection_name)
        if not exists:
            try:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self._dimensions, distance=Distance.COSINE),
                )
                logger.info("Example collection created", collection=self._collection_name)
            except Exception as exc:
                # Another worker may have created it concurrently — tolerate that.
                if "already exists" not in str(exc).lower():
                    raise
        self._initialized = True

    async def index_example(
        self, question: str, sql: str, connection_id: str | None = None
    ) -> None:
        """Embed and upsert a single NL→SQL example (best-effort)."""
        question = (question or "").strip()
        sql = (sql or "").strip()
        if not question or not sql:
            return
        try:
            await self._ensure_initialized()
            embedding = await self._embedder.embed(question)
            # Deterministic id so re-submitting the same pair updates in place.
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{connection_id or ''}:{question}:{sql}")
            )
            payload: dict[str, Any] = {"question": question, "sql": sql}
            if connection_id is not None:
                payload["connection_id"] = connection_id
            await self._client.upsert(
                collection_name=self._collection_name,
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
            )
            logger.debug("Indexed few-shot example", question=question[:60])
        except Exception as exc:
            logger.warning("Failed to index few-shot example — skipping", error=str(exc))

    async def search(
        self, question: str, top_k: int = 3, connection_id: str | None = None
    ) -> list[dict[str, str]]:
        """Return the top-k most similar past examples as {question, sql} dicts."""
        question = (question or "").strip()
        if not question or top_k <= 0:
            return []
        try:
            await self._ensure_initialized()
            embedding = await self._embedder.embed(question)
            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=embedding,
                query_filter=Filter(should=own_or_shared_should(connection_id)),
                limit=top_k,
                with_payload=True,
            )
            examples: list[dict[str, str]] = []
            for point in response.points:
                payload = point.payload or {}
                q = payload.get("question", "")
                sql = payload.get("sql", "")
                if q and sql:
                    examples.append({"question": q, "sql": sql})
            return examples
        except Exception as exc:
            logger.warning("Few-shot example search failed — returning none", error=str(exc))
            return []

    async def count(self, connection_id: str | None = None) -> int:
        """Return how many examples are stored (best-effort; 0 on error)."""
        try:
            await self._ensure_initialized()
            result = await self._client.count(
                collection_name=self._collection_name,
                count_filter=Filter(should=own_or_shared_should(connection_id)),
                exact=True,
            )
            return int(result.count)
        except Exception as exc:
            logger.warning("Example count failed", error=str(exc))
            return 0

    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False
