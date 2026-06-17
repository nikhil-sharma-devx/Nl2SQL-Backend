"""Core interfaces package."""
from nl_to_sql.core.interfaces.i_cache import ICache
from nl_to_sql.core.interfaces.i_embedder import IEmbedder
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.interfaces.i_sql_validator import ISQLValidator
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore

__all__ = [
    "ICache",
    "IEmbedder",
    "ILLMProvider",
    "ISQLValidator",
    "IVectorStore",
]
