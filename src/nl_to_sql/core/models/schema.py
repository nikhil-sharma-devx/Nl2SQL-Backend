"""Pydantic models for database schema representation."""
from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """Metadata for a single database column."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None  # e.g. "orders.customer_id"
    description: str | None = None


class TableInfo(BaseModel):
    """Metadata for a single database table."""

    name: str
    schema_name: str = "public"
    columns: list[ColumnInfo]
    description: str | None = None


class SchemaChunk(BaseModel):
    """A text chunk derived from schema info that lives in the vector store.

    Each chunk corresponds to one table's schema description, making retrieval
    granular and meaningful.
    """

    chunk_id: str = Field(..., description="Unique ID — typically '<schema>.<table>'.")
    table_name: str
    schema_name: str = "public"
    content: str = Field(
        ...,
        description="Human-readable text description of the table and its columns. "
        "This is what gets embedded and stored in the vector store.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Dense vector representation (populated after embedding).",
    )
    metadata: dict = Field(default_factory=dict)

    def to_text(self) -> str:
        """Return the chunk content used for LLM prompt injection."""
        return self.content


class SchemaMetadata(BaseModel):
    """Top-level wrapper for the full database schema."""

    database_name: str
    dialect: str
    tables: list[TableInfo]

    @property
    def table_names(self) -> list[str]:
        """Return sorted list of all table names."""
        return sorted(t.name for t in self.tables)
