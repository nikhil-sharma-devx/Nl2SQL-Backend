"""Pipelines package — orchestration layer for NL2SQL workflows.

This package coordinates the various components (RAG retrieval, SQL generation,
validation, execution) into complete workflows.

Architecture:
  - rag/ (retrieval components)
  - pipelines/ (orchestration - this package)
  - preprocessing/ (query transformation)
  - services/ (business logic)

Note: Pipeline components remain in services/ package to avoid circular imports,
but are conceptually part of the pipeline layer. Import them from services/:

  from nl_to_sql.services.query_orchestrator import QueryOrchestrator
  from nl_to_sql.services.sql_generator import SQLGeneratorService
  from nl_to_sql.services.sql_validator import SQLValidatorService
"""

# No imports here to avoid circular dependencies
# See docstring for correct import paths

__all__ = [
    "QueryOrchestrator",  # Import from nl_to_sql.services.query_orchestrator
    "SQLGeneratorService",  # Import from nl_to_sql.services.sql_generator
    "SQLValidatorService",  # Import from nl_to_sql.services.sql_validator
]
