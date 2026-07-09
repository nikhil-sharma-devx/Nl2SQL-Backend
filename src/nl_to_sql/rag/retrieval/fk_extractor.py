"""FK Relationship Extractor â€” Expands retrieved tables via foreign keys.

When the retriever returns a set of tables, this service:
1. Extracts all foreign key relationships from those tables
2. Identifies related tables that should also be included
3. Fetches schema chunks for those related tables
4. Returns the expanded set of tables

This prevents hallucination when the LLM needs to JOIN to related tables
(like products â†’ categories) but the retriever only returned products.

SOLID:
  S â€” Only handles FK relationship expansion
  D â€” Depends on vector store and schema metadata
"""
from typing import Any

import structlog

from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.core.models.schema import SchemaChunk, SchemaMetadata

logger = structlog.get_logger(__name__)


class FKRelationshipExtractor:
    """Extracts and follows foreign key relationships to expand schema context."""

    def __init__(
        self,
        vector_store: IVectorStore,
        schema_metadata: SchemaMetadata | None = None,
    ) -> None:
        """Initialize with vector store and optional schema metadata.

        Args:
            vector_store: Vector store to fetch schema chunks.
            schema_metadata: Full schema metadata with FK relationships.
                           If None, will extract FKs from chunk content.
        """
        self._vector_store = vector_store
        self._schema_metadata = schema_metadata
        self._fk_map: dict[str, set[str]] = {}  # table -> set of related tables

        # Build FK map from schema metadata if available
        if schema_metadata:
            self._build_fk_map_from_metadata(schema_metadata)

    def _build_fk_map_from_metadata(self, metadata: SchemaMetadata) -> None:
        """Build a map of table -> related tables via foreign keys.

        Example:
            {
                "products": {"categories"},
                "orders": {"customers", "order_items"},
                "order_items": {"orders", "products"},
            }
        """
        for table in metadata.tables:
            related_tables: set[str] = set()

            for col in table.columns:
                if col.foreign_key:
                    # Extract referenced table from "table.column" format
                    ref_table = col.foreign_key.split(".")[0]
                    related_tables.add(ref_table)

            self._fk_map[table.name] = related_tables

        logger.info(
            "Built FK relationship map",
            table_count=len(self._fk_map),
            relationships={k: list(v) for k, v in self._fk_map.items()},
        )

    def _extract_fk_from_chunks(self, chunks: list[SchemaChunk]) -> dict[str, set[str]]:
        """Extract FK relationships from chunk content.

        Parses chunk text to find FK patterns like "[FK â†’ categories.category_id]"

        Args:
            chunks: Schema chunks to analyze.

        Returns:
            Dict mapping table_name -> set of related table names.
        """
        import re

        fk_map: dict[str, set[str]] = {}

        for chunk in chunks:
            table_name = chunk.table_name
            related_tables: set[str] = set()

            # Find all FK references in chunk content
            fk_pattern = r'\[FK\s*â†’\s*([a-zA-Z_][a-zA-Z0-9_]*)\.'
            matches = re.findall(fk_pattern, chunk.content)

            for ref_table in matches:
                related_tables.add(ref_table)

            fk_map[table_name] = related_tables

        return fk_map

    async def expand_tables(
        self,
        initial_chunks: list[SchemaChunk],
        max_expansion: int = 3,
        user_id: str | None = None,
    ) -> list[SchemaChunk]:
        """Expand initial table set by following FK relationships.

        Args:
            initial_chunks: Initially retrieved schema chunks.
            max_expansion: Maximum number of tables to add via FK expansion.
            user_id: When provided, restrict FK-chunk fetches to this user.

        Returns:
            Expanded list of schema chunks including related tables.
        """
        if not initial_chunks:
            return initial_chunks

        # Get FK map from metadata or extract from chunks
        if self._fk_map:
            fk_map = self._fk_map
        else:
            fk_map = self._extract_fk_from_chunks(initial_chunks)

        # Get initially retrieved table names
        initial_tables = {chunk.table_name for chunk in initial_chunks}
        logger.debug(
            "Starting FK expansion",
            initial_tables=list(initial_tables),
        )

        # Find all related tables via FKs
        related_tables: set[str] = set()
        for table in initial_tables:
            if table in fk_map:
                related_tables.update(fk_map[table])

        # Remove already retrieved tables
        new_tables = related_tables - initial_tables

        if not new_tables:
            logger.debug("No FK expansion needed")
            return initial_chunks

        # Limit expansion to avoid retrieving too many tables
        tables_to_fetch = list(new_tables)[:max_expansion]
        logger.info(
            "Expanding schema via FK relationships",
            initial_tables=list(initial_tables),
            new_tables=tables_to_fetch,
        )

        # Fetch schema chunks for related tables
        try:
            additional_chunks = await self._vector_store.get_chunks_by_table_names(
                tables_to_fetch, user_id=user_id
            )

            # Combine and return (initial chunks first, then related)
            expanded = initial_chunks + additional_chunks

            logger.info(
                "FK expansion complete",
                total_tables=len(expanded),
                tables=[c.table_name for c in expanded],
            )

            return expanded  # type: ignore[no-any-return]

        except Exception as exc:
            logger.warning(
                "FK expansion failed, returning initial chunks",
                error=str(exc),
            )
            return initial_chunks

    async def expand_and_retrieve(
        self,
        question: str,
        retrieval_chain: Any,
        max_expansion: int = 3,
    ) -> list[SchemaChunk]:
        """Convenience method: retrieve + expand in one call.

        Args:
            question: User's natural language question.
            retrieval_chain: RetrievalChain instance for initial retrieval.
            max_expansion: Max tables to add via FK expansion.

        Returns:
            Expanded schema chunks.
        """
        # Step 1: Initial retrieval
        initial_chunks = await retrieval_chain.retrieve(question)

        # Step 2: Expand via FK relationships
        expanded_chunks = await self.expand_tables(initial_chunks, max_expansion)

        return expanded_chunks
