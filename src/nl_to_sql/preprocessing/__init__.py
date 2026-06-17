"""Preprocessing package — query transformation and analysis.

This package handles query preprocessing before the main RAG pipeline:
  - Classification: Determine query intent and complexity
  - Expansion: Generate alternative query formulations
  - Rewriting: Clarify ambiguous queries

Architecture:
  - User query → preprocessing/ → rag/retrieval/ → pipelines/ → SQL

Note: Components remain in services/ to avoid circular imports with the
orchestrator. Import them from services/ or use this package's lazy imports.
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "QueryClassifier":
        from nl_to_sql.services.query_classifier import QueryClassifier
        return QueryClassifier
    elif name == "QueryExpander":
        from nl_to_sql.services.query_expander import QueryExpander
        return QueryExpander
    elif name == "QueryRewriter":
        from nl_to_sql.services.query_rewriter import QueryRewriter
        return QueryRewriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "QueryClassifier",  # nl_to_sql.services.query_classifier
    "QueryExpander",  # nl_to_sql.services.query_expander
    "QueryRewriter",  # nl_to_sql.services.query_rewriter
]
