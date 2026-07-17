"""Chunker — splits large documents into overlapping chunks.

The chunking strategy is controlled here and can be changed without touching
anything else in the pipeline. Supports table-level (one chunk per table),
fixed-size with overlap, sentence-aware, and parent-child strategies.
"""
import re

import structlog

from nl_to_sql.core.models.schema import SchemaChunk

logger = structlog.get_logger(__name__)

# Matches a column definition line in a table chunk, e.g. "  - customer_id (INT) [FK ...]"
_COLUMN_LINE_RE = re.compile(r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def build_column_child_chunks(parent: SchemaChunk) -> list[SchemaChunk]:
    """Derive fine-grained column-level child chunks from a table chunk (P4).

    Each child carries a ``parent_id`` pointing at the parent table chunk so the
    retrieval layer can hit a precise column chunk yet return the full parent
    table DDL to the LLM. Children reuse the parent's ``table_name`` so table
    candidate-extraction downstream is unaffected.
    """
    children: list[SchemaChunk] = []
    for line in parent.content.split("\n"):
        match = _COLUMN_LINE_RE.match(line)
        if not match:
            continue
        column = match.group(1)
        children.append(
            SchemaChunk(
                chunk_id=f"{parent.chunk_id}#col={column}",
                table_name=parent.table_name,
                schema_name=parent.schema_name,
                content=f"Table {parent.table_name} column {line.strip().lstrip('- ').strip()}",
                metadata={
                    **parent.metadata,
                    "parent_id": parent.chunk_id,
                    "is_child": True,
                    "column": column,
                },
            )
        )
    return children


class Chunker:
    """Splits schema documents into chunks of a configured size.

    SOLID:
      S — Only responsible for chunking; does not embed or store.
    """

    def __init__(
        self,
        strategy: str = "table",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialise the chunker.

        Args:
            strategy: Chunking strategy — 'table' (one chunk per table),
                      'fixed' (fixed-size windows), or 'sentence' (sentence-aware).
            chunk_size: Maximum characters per chunk (for 'fixed' and 'sentence').
            chunk_overlap: Overlap between consecutive chunks (for 'fixed').
        """
        self._strategy = strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, documents: list[SchemaChunk]) -> list[SchemaChunk]:
        """Split documents into chunks using the configured strategy.

        Args:
            documents: List of SchemaChunk documents (one per table).

        Returns:
            List of SchemaChunk, potentially more than input if splitting occurred.
        """
        if self._strategy == "table":
            return self._chunk_by_table(documents)
        elif self._strategy == "fixed":
            return self._chunk_fixed_size(documents)
        elif self._strategy == "sentence":
            return self._chunk_sentence_aware(documents)
        elif self._strategy == "parent_child":
            return self._chunk_parent_child(documents)
        else:
            logger.warning(
                "Unknown chunking strategy, falling back to table-level",
                strategy=self._strategy,
            )
            return self._chunk_by_table(documents)

    def _chunk_by_table(self, documents: list[SchemaChunk]) -> list[SchemaChunk]:
        """Table-level chunking — one chunk per table (pass-through).

        This is the default and simplest strategy. Each table is already
        a natural semantic unit for schema context.
        """
        logger.info("Using table-level chunking", chunk_count=len(documents))
        return documents

    def _chunk_parent_child(self, documents: list[SchemaChunk]) -> list[SchemaChunk]:
        """Parent-child chunking (P4) — keep the parent table chunk and add one
        fine-grained child chunk per column.

        Column-level children improve retrieval precision; the parent table chunk
        is retained so the LLM still receives full table DDL for generation (the
        retrieval layer dedupes children back to their parent).
        """
        result: list[SchemaChunk] = []
        for doc in documents:
            result.append(doc)
            result.extend(build_column_child_chunks(doc))
        logger.info(
            "Parent-child chunking complete",
            parent_count=len(documents),
            total_count=len(result),
        )
        return result

    def _chunk_fixed_size(self, documents: list[SchemaChunk]) -> list[SchemaChunk]:
        """Fixed-size chunking with overlap.

        Splits documents that exceed chunk_size into windows of chunk_size
        characters with chunk_overlap overlap between consecutive windows.
        """
        result: list[SchemaChunk] = []
        for doc in documents:
            content = doc.content
            if len(content) <= self._chunk_size:
                result.append(doc)
                continue

            # Split into overlapping windows
            start = 0
            part_idx = 0
            while start < len(content):
                end = start + self._chunk_size
                chunk_content = content[start:end]
                result.append(
                    SchemaChunk(
                        chunk_id=f"{doc.chunk_id}__part{part_idx}",
                        table_name=doc.table_name,
                        schema_name=doc.schema_name,
                        content=chunk_content,
                        metadata={**doc.metadata, "part": part_idx},
                    )
                )
                start += self._chunk_size - self._chunk_overlap
                part_idx += 1

        logger.info(
            "Fixed-size chunking complete",
            input_count=len(documents),
            output_count=len(result),
        )
        return result

    def _chunk_sentence_aware(self, documents: list[SchemaChunk]) -> list[SchemaChunk]:
        """Sentence-aware chunking — splits on newlines respecting sentence boundaries.

        Tries to keep logical groupings (column definitions) together.
        """
        result: list[SchemaChunk] = []
        for doc in documents:
            content = doc.content
            if len(content) <= self._chunk_size:
                result.append(doc)
                continue

            # Split on newlines (each line is typically a column definition)
            lines = content.split("\n")
            current_chunk: list[str] = []
            current_size = 0
            part_idx = 0

            for line in lines:
                line_size = len(line) + 1  # +1 for newline
                if current_size + line_size > self._chunk_size and current_chunk:
                    # Emit current chunk
                    result.append(
                        SchemaChunk(
                            chunk_id=f"{doc.chunk_id}__part{part_idx}",
                            table_name=doc.table_name,
                            schema_name=doc.schema_name,
                            content="\n".join(current_chunk),
                            metadata={**doc.metadata, "part": part_idx},
                        )
                    )
                    current_chunk = []
                    current_size = 0
                    part_idx += 1

                current_chunk.append(line)
                current_size += line_size

            # Emit remaining
            if current_chunk:
                result.append(
                    SchemaChunk(
                        chunk_id=f"{doc.chunk_id}__part{part_idx}",
                        table_name=doc.table_name,
                        schema_name=doc.schema_name,
                        content="\n".join(current_chunk),
                        metadata={**doc.metadata, "part": part_idx},
                    )
                )

        logger.info(
            "Sentence-aware chunking complete",
            input_count=len(documents),
            output_count=len(result),
        )
        return result
