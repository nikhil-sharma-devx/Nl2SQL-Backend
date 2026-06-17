"""Document builder — converts raw schema objects into human-readable text.

Each document typically represents one table with its columns described in
natural language so the embedding model has rich context.
"""
import structlog

from nl_to_sql.core.models.schema import SchemaChunk, TableInfo

logger = structlog.get_logger(__name__)


class DocBuilder:
    """Converts TableInfo objects into human-readable SchemaChunk documents.

    SOLID:
      S — Only converts structured schema into text documents.
    """

    @staticmethod
    def build_chunks(tables: list[TableInfo]) -> list[SchemaChunk]:
        """Convert a list of TableInfo objects into SchemaChunk documents.

        Args:
            tables: List of table metadata objects.

        Returns:
            List of SchemaChunk, one per table.
        """
        chunks = [DocBuilder._table_to_chunk(table) for table in tables]
        logger.info("Built document chunks from schema", chunk_count=len(chunks))
        return chunks

    @staticmethod
    def _table_to_chunk(table: TableInfo) -> SchemaChunk:
        """Convert a TableInfo into a descriptive text SchemaChunk.

        The text representation is human-readable and designed to give the LLM
        precise context about each table's structure and relationships.
        """
        lines: list[str] = [
            # Use unqualified name for default schemas so the LLM generates
            # clean SQL without schema prefix (e.g. FROM categories, not
            # FROM public.categories)
            f"Table: {table.name}",
        ]
        if table.description:
            lines.append(f"Description: {table.description}")

        lines.append("Columns:")
        for col in table.columns:
            parts = [f"  - {col.name} ({col.data_type})"]
            if col.primary_key:
                parts.append("[PRIMARY KEY]")
            if col.foreign_key:
                parts.append(f"[FK → {col.foreign_key}]")
            if not col.nullable:
                parts.append("[NOT NULL]")
            if col.description:
                parts.append(f"— {col.description}")
            lines.append(" ".join(parts))

        pk_cols = [c.name for c in table.columns if c.primary_key]
        if pk_cols:
            lines.append(f"Primary Key: ({', '.join(pk_cols)})")

        fk_cols = [(c.name, c.foreign_key) for c in table.columns if c.foreign_key]
        if fk_cols:
            for col_name, fk_ref in fk_cols:
                lines.append(f"Foreign Key: {col_name} → {fk_ref}")

        content = "\n".join(lines)
        return SchemaChunk(
            chunk_id=f"{table.schema_name}.{table.name}",
            table_name=table.name,
            schema_name=table.schema_name,
            content=content,
        )
