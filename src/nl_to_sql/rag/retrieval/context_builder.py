"""Context builder — formats reranked chunks into LLM-consumable context.

Takes the top schema chunks and produces a structured prompt section with
table names, column definitions, data types, and key relationships.
"""
import structlog

from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


def dedupe_parent_chunks(chunks: list[SchemaChunk]) -> list[SchemaChunk]:
    """Collapse parent-child chunks back to their parent for LLM context (P4).

    Column-level child chunks (``metadata.is_child``) sharpen retrieval but the
    LLM needs the full parent table DDL. When a parent chunk for a table is
    present we drop that table's child chunks; child chunks whose parent is
    absent are kept so no table is lost. A no-op when parent-child chunking is
    not in use (no ``is_child`` metadata).
    """
    parent_tables = {
        c.table_name for c in chunks if not c.metadata.get("is_child")
    }
    result: list[SchemaChunk] = []
    seen_ids: set[str] = set()
    for c in chunks:
        if c.metadata.get("is_child") and c.table_name in parent_tables:
            continue
        if c.chunk_id in seen_ids:
            continue
        seen_ids.add(c.chunk_id)
        result.append(c)
    return result


class ContextBuilder:
    """Formats schema chunks into a structured prompt section for the LLM.

    SOLID:
      S — Only formats context; does not retrieve or rerank.
    """

    @staticmethod
    def build(chunks: list[SchemaChunk]) -> str:
        """Concatenate chunk contents into a single schema context block.

        Args:
            chunks: Reranked schema chunks (most relevant first).

        Returns:
            A formatted multi-table schema description for LLM injection.
        """
        if not chunks:
            return "No relevant schema context was found."

        chunks = dedupe_parent_chunks(chunks)
        sections = "\n\n---\n\n".join(c.to_text() for c in chunks)
        context = f"RELEVANT DATABASE SCHEMA:\n\n{sections}"

        logger.debug(
            "Schema context built",
            chunk_count=len(chunks),
            context_length=len(context),
        )
        return context
