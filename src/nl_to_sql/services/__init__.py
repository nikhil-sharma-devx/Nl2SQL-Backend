"""Services package — Business logic and features.

This package contains business logic that is NOT part of the core RAG pipeline:
  - Analytics: Query metrics and insights
  - Chat sessions: Conversation management
  - Feedback: User feedback collection
  - Fine-tuning: Model improvement
  - Query history: Historical query tracking
  - Training data: Dataset management
  - SQL explanation: Natural language SQL descriptions
  - Prompt management: Prompt templates and versions
  - TTL management: Cache expiration policies
  - Schema monitoring: Background schema change detection

Core RAG components have been moved to:
  - rag/ingestion/: Offline schema ingestion pipeline
  - rag/retrieval/: Online retrieval pipeline
  - pipelines/: Query orchestration (orchestrator, generator, validator)
  - preprocessing/: Query preprocessing (classifier, expander, rewriter)
"""

# Backward compatibility — re-export from new locations
from nl_to_sql.services.reranker import CrossEncoderReranker
from nl_to_sql.services.schema_retriever import SchemaRetriever

__all__ = [
    "CrossEncoderReranker",
    "SchemaRetriever",
]
