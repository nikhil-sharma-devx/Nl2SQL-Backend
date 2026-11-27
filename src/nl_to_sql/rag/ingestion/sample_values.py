"""Sample-value profiling for low-cardinality columns.

Most real databases have no column comments, so the model has to guess the
literal to put in a WHERE clause (``status = 'cancelled'`` vs ``'canceled'`` vs
``'CANCELLED'``). Indexing the distinct values of small categorical columns
removes that guess — usually the single biggest accuracy win for filters.

Privacy and cost guards (this is the only ingestion step that reads row data):
  - off unless ``rag_sample_values_enabled`` is set;
  - runs through the connection's scoped client, so the access policy applies
    (restricted columns / disallowed tables are rejected, never sampled);
  - skips columns whose names look personal or secret, and non-text types;
  - keeps a column only if it has at most ``MAX_DISTINCT`` values, each
    truncated to ``MAX_VALUE_CHARS``;
  - bounded work: at most ``MAX_TABLES`` tables x ``MAX_COLUMNS_PER_TABLE``
    columns, each query under the client's statement timeout. Any failure
    skips that column and never fails the sync.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from nl_to_sql.core.models.schema import SchemaMetadata
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient

logger = structlog.get_logger(__name__)

MAX_DISTINCT = 12
MAX_VALUE_CHARS = 40
MAX_TABLES = 60
MAX_COLUMNS_PER_TABLE = 8

_TEXT_TYPES = re.compile(r"char|text|enum|user-defined|citext|bool", re.IGNORECASE)
# Fragments that are sensitive wherever they appear in a column name.
_SENSITIVE_FRAGMENTS = (
    "email",
    "phone",
    "passw",
    "secret",
    "token",
    "ssn",
    "iban",
    "passport",
    "salary",
)
# Whole name tokens (split on "_" and camelCase) that mark personal or free-text data.
_SENSITIVE_TOKENS = frozenset(
    {
        "name",
        "firstname",
        "lastname",
        "fullname",
        "address",
        "street",
        "zip",
        "postal",
        "postcode",
        "dob",
        "birth",
        "birthday",
        "wage",
        "income",
        "card",
        "account",
        "tax",
        "license",
        "ip",
        "lat",
        "latitude",
        "lon",
        "lng",
        "longitude",
        "geo",
        "note",
        "notes",
        "comment",
        "comments",
        "description",
        "body",
        "content",
        "url",
        "uuid",
        "id",
        "key",
        "hash",
        "salt",
        "mobile",
        "social",
    }
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _looks_sensitive(column_name: str) -> bool:
    lowered = column_name.lower()
    if any(f in lowered for f in _SENSITIVE_FRAGMENTS):
        return True
    tokens = {t for t in re.split(r"[_\W]+", _CAMEL.sub("_", column_name).lower()) if t}
    return bool(tokens & _SENSITIVE_TOKENS)


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _quote(identifier: str, dialect: str) -> str | None:
    """Quote an identifier for a profiling query; refuse anything unusual."""
    if not _IDENT.match(identifier):
        return None
    return f"`{identifier}`" if dialect == "mysql" else f'"{identifier}"'


def is_profileable(column_name: str, data_type: str) -> bool:
    return bool(_TEXT_TYPES.search(data_type or "")) and not _looks_sensitive(column_name)


async def profile_sample_values(
    schema: SchemaMetadata,
    db_client: AsyncDatabaseClient,
    dialect: str = "postgresql",
) -> dict[tuple[str, str], list[str]]:
    """Return {(table, column): [values]} for qualifying low-cardinality columns."""
    found: dict[tuple[str, str], list[str]] = {}
    for table in schema.tables[:MAX_TABLES]:
        qt = _quote(table.name, dialect)
        if qt is None:
            continue
        if table.schema_name and table.schema_name not in ("public", "") and dialect != "mysql":
            qs = _quote(table.schema_name, dialect)
            if qs is None:
                continue
            qt = f"{qs}.{qt}"
        candidates = [c for c in table.columns if is_profileable(c.name, c.data_type)]
        for column in candidates[:MAX_COLUMNS_PER_TABLE]:
            qc = _quote(column.name, dialect)
            if qc is None:
                continue
            # One more than the cap tells us the column is too high-cardinality.
            sql = (
                f"SELECT DISTINCT {qc} AS v FROM {qt} "  # noqa: S608 - identifiers validated + quoted
                f"WHERE {qc} IS NOT NULL LIMIT {MAX_DISTINCT + 1}"
            )
            try:
                rows: list[dict[str, Any]] = await db_client.execute_sql(sql)
            except Exception as exc:  # policy-blocked, timeout, permission — just skip
                logger.debug(
                    "Sample-value probe skipped",
                    table=table.name,
                    column=column.name,
                    error=str(exc),
                )
                continue
            if not rows or len(rows) > MAX_DISTINCT:
                continue
            values = sorted(
                {str(r.get("v"))[:MAX_VALUE_CHARS] for r in rows if r.get("v") is not None}
            )
            if values:
                found[(table.name, column.name)] = values
    logger.info("Sample values profiled", columns=len(found))
    return found


def apply_sample_values(
    schema: SchemaMetadata, values: dict[tuple[str, str], list[str]]
) -> SchemaMetadata:
    """Attach profiled values to the schema's ColumnInfo objects (copy)."""
    if not values:
        return schema
    tables = []
    for table in schema.tables:
        columns = [
            c.model_copy(update={"sample_values": values.get((table.name, c.name))})
            if (table.name, c.name) in values
            else c
            for c in table.columns
        ]
        tables.append(table.model_copy(update={"columns": columns}))
    return schema.model_copy(update={"tables": tables})
