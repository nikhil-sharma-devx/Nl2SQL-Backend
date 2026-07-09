"""Composition-root factories.

Each factory module builds one family of infrastructure objects from Settings.
Extracted from `config/container.py` so the container is a thin wiring layer
and the heavy provider imports live next to the code that selects them.
"""
from nl_to_sql.config.factories.cache_factory import build_cache
from nl_to_sql.config.factories.llm_factory import build_llm_provider, create_llm_provider
from nl_to_sql.config.factories.rag_factory import build_embedder, build_vector_store

__all__ = [
    "build_cache",
    "build_embedder",
    "build_llm_provider",
    "build_vector_store",
    "create_llm_provider",
]
