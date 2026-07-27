"""RAG infrastructure factories — embedder and vector store selection."""
from nl_to_sql.config.settings import Settings
from nl_to_sql.infrastructure.embeddings.huggingface_embedder import HuggingFaceEmbedder
from nl_to_sql.infrastructure.vector_store.chroma_store import ChromaVectorStore


def build_vector_store(settings: Settings) -> object:
    """Factory: choose the schema vector store from settings."""
    if settings.vector_store_provider == "qdrant":
        from nl_to_sql.infrastructure.vector_store.qdrant_store import QdrantVectorStore
        return QdrantVectorStore(
            url=settings.qdrant_url.strip(),
            api_key=settings.qdrant_api_key.strip() or None,
            collection_name=settings.qdrant_collection_name.strip(),
            dimensions=settings.embedding_dimensions,
            # Keep any legacy semantic-cache points (written here before the cache
            # got its own collection) out of schema retrieval and the chunk count.
            non_schema_types=("schema_hash", "semantic_cache"),
        )
    if settings.vector_store_provider == "faiss":
        from nl_to_sql.infrastructure.vector_store.faiss_store import FAISSVectorStore
        return FAISSVectorStore(dimensions=settings.embedding_dimensions)
    # Default: ChromaDB
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


def build_semantic_cache_store(settings: Settings) -> object:
    """Factory: a *dedicated* vector store for the semantic cache.

    The semantic cache must NOT share the schema vector store. Sharing it writes
    cache entries into the schema collection (inflating the chunk count and
    leaking cache points into schema retrieval) and lets ``SemanticCache.clear()``
    — which calls ``delete_collection()`` — wipe the entire schema index. This
    store is bound to ``semantic_cache_collection`` instead of the schema one and
    keeps the default reserved types so its ``semantic_cache`` points stay
    searchable.
    """
    if settings.vector_store_provider == "qdrant":
        from nl_to_sql.infrastructure.vector_store.qdrant_store import QdrantVectorStore
        return QdrantVectorStore(
            url=settings.qdrant_url.strip(),
            api_key=settings.qdrant_api_key.strip() or None,
            collection_name=settings.semantic_cache_collection.strip(),
            dimensions=settings.embedding_dimensions,
        )
    if settings.vector_store_provider == "faiss":
        from nl_to_sql.infrastructure.vector_store.faiss_store import FAISSVectorStore
        return FAISSVectorStore(dimensions=settings.embedding_dimensions)
    # Default: ChromaDB
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.semantic_cache_collection,
    )


def build_embedder(settings: Settings) -> object:
    """Factory: choose the embedder from settings."""
    if settings.embedding_provider == "gemini":
        from nl_to_sql.infrastructure.embeddings.gemini_embedder import GeminiEmbedder
        return GeminiEmbedder(
            api_key=settings.resolved_gemini_embedding_api_key,
            model=settings.gemini_embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    return HuggingFaceEmbedder(
        model=settings.huggingface_model,
        dimensions=settings.embedding_dimensions,
    )
