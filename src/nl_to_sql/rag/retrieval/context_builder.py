"""Context builder — formats reranked chunks into LLM-consumable context.

Takes the top schema chunks and produces a structured prompt section with
table names, column definitions, data types, and key relationships.
"""
import structlog

from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)


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

        sections = "\n\n---\n\n".join(c.to_text() for c in chunks)
        context = f"RELEVANT DATABASE SCHEMA:\n\n{sections}"

        logger.debug(
            "Schema context built",
            chunk_count=len(chunks),
            context_length=len(context),
        )
        return context
