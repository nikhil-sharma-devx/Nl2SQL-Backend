"""Schema loader — connects to the target database and introspects its structure.

Reads table names, column names, data types, primary keys, foreign keys,
and any available column comments from the live database using
information_schema queries.
"""
import structlog

from nl_to_sql.core.exceptions import SchemaIngestionError
from nl_to_sql.core.models.schema import ColumnInfo, SchemaMetadata, TableInfo
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient

logger = structlog.get_logger(__name__)


class SchemaLoader:
    """Loads database schema by reflecting the live relational database.

    SOLID:
      S — Only responsible for loading raw schema structure from a database.
      D — Depends on AsyncDatabaseClient abstraction.
    """

    def __init__(self, db_client: AsyncDatabaseClient) -> None:
        self._db_client = db_client

    async def load(self, schema_name: str = "public") -> SchemaMetadata:
        """Reflect database schema and return a SchemaMetadata object.

        Args:
            schema_name: The database schema to reflect (default: 'public').

        Returns:
            SchemaMetadata with all tables and columns.

        Raises:
            SchemaIngestionError: If reflection fails.
        """
        log = logger.bind(schema=schema_name)
        log.info("Loading schema from live database")

        try:
            schema_dict = await self._db_client.reflect_schema(schema_name=schema_name)
        except Exception as exc:
            raise SchemaIngestionError(
                f"Failed to reflect database schema: {exc}", detail=str(exc)
            ) from exc

        schema_metadata = self.build_schema_from_dict(schema_dict)
        log.info(
            "Schema loaded successfully",
            table_count=len(schema_metadata.tables),
            tables=schema_metadata.table_names,
        )
        return schema_metadata

    @staticmethod
    def build_schema_from_dict(raw: dict) -> SchemaMetadata:
        """Construct a SchemaMetadata from a plain Python dict.

        Useful for loading schema from a JSON config file or DB reflection.

        Args:
            raw: Dict with keys: database_name, dialect, tables (list).

        Returns:
            A SchemaMetadata instance.
        """
        tables: list[TableInfo] = []
        for t in raw.get("tables", []):
            columns = [ColumnInfo(**c) for c in t.get("columns", [])]
            tables.append(
                TableInfo(
                    name=t["name"],
                    schema_name=t.get("schema_name", "public"),
                    columns=columns,
                    description=t.get("description"),
                )
            )
        return SchemaMetadata(
            database_name=raw["database_name"],
            dialect=raw["dialect"],
            tables=tables,
        )
