"""SchemaCatalogService — owns the per-user schema catalog (source of truth).

The catalog (``user_schemas`` + ``user_schema_tables`` in Supabase) is the
authoritative record the Schema page reads from; Qdrant is a disposable index
rebuilt from the catalog. This service:

  - reads the catalog with pin/``is_new`` status joined (``get_catalog``),
  - reflects a user's live DB into the catalog (``sync_from_live_db``),
  - overlays/replaces an uploaded JSON schema (``apply_upload``),
  - edits sticky ``user_description`` (``set_table_description``),
  - clears the "new table" badge (``mark_seen``),

and re-embeds the affected user's chunks after any write.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nl_to_sql.core.models.schema import ColumnInfo, SchemaMetadata, TableInfo
from nl_to_sql.infrastructure.database.models import (
    FavoritedTable,
    UserSchema,
    UserSchemaTable,
)
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.rag.ingestion.schema_loader import SchemaLoader
from nl_to_sql.services.schema_ingestion import SchemaIngestionService

logger = structlog.get_logger(__name__)


@dataclass
class SyncResult:
    changed: bool
    new_tables: list[str]
    total_tables: int


@dataclass
class UploadResult:
    tables: int
    replaced: bool
    new_tables: list[str]


class SchemaCatalogService:
    """Owns the per-user schema catalog tables and keeps Qdrant in sync."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        ingestion: SchemaIngestionService,
        per_user_isolation: bool = False,
    ) -> None:
        self._session_factory = session_factory
        self._ingestion = ingestion
        self._isolation = per_user_isolation

    # ── Read model ────────────────────────────────────────────────────────────

    async def get_catalog(
        self, user_id: str, schema_name: str | None = None
    ) -> dict[str, Any]:
        """Return the catalog for a user with ``pinned`` + ``is_new`` joined."""
        async with self._session_factory() as db:
            header = (
                await db.execute(select(UserSchema).where(UserSchema.user_id == user_id))
            ).scalar_one_or_none()

            q = select(UserSchemaTable).where(UserSchemaTable.user_id == user_id)
            if schema_name:
                q = q.where(UserSchemaTable.schema_name == schema_name)
            q = q.order_by(UserSchemaTable.table_name)
            rows = (await db.execute(q)).scalars().all()

            favs = (
                await db.execute(
                    select(FavoritedTable).where(FavoritedTable.user_id == user_id)
                )
            ).scalars().all()

        pinned_names = {f.table_name for f in favs}
        tables = [
            {
                "id": r.id,
                "schema_name": r.schema_name,
                "table_name": r.table_name,
                "source": r.source,
                "columns": r.columns or [],
                "description": r.user_description or r.reflected_description,
                "pinned": r.table_name in pinned_names,
                "is_new": r.is_new,
                "last_seen_at": r.last_seen_at,
            }
            for r in rows
        ]
        return {
            "database_name": header.database_name if header else None,
            "dialect": header.dialect if header else "postgresql",
            "source": header.source if header else "reflected",
            "last_synced_at": header.last_synced_at if header else None,
            "tables": tables,
        }

    # ── Sync from live DB (auto-generate path) ────────────────────────────────

    async def sync_from_live_db(
        self,
        user_id: str,
        db_client: AsyncDatabaseClient,
        schema_name: str = "public",
    ) -> SyncResult:
        """Reflect the user's live DB into the catalog, then re-embed if changed."""
        loader = SchemaLoader(db_client)
        schema = await loader.load(schema_name)
        new_hash = SchemaIngestionService.compute_schema_hash(schema)
        now = datetime.utcnow()
        live_names = {t.name for t in schema.tables}
        new_tables: list[str] = []

        async with self._session_factory() as db:
            header = (
                await db.execute(select(UserSchema).where(UserSchema.user_id == user_id))
            ).scalar_one_or_none()
            old_hash = header.schema_hash if header else None

            rows = (
                await db.execute(
                    select(UserSchemaTable).where(
                        UserSchemaTable.user_id == user_id,
                        UserSchemaTable.schema_name == schema_name,
                    )
                )
            ).scalars().all()
            existing = {r.table_name: r for r in rows}

            for t in schema.tables:
                cols = [c.model_dump() for c in t.columns]
                row = existing.get(t.name)
                if row is None:
                    db.add(
                        UserSchemaTable(
                            user_id=user_id,
                            schema_name=schema_name,
                            table_name=t.name,
                            source="reflected",
                            columns=cols,
                            reflected_description=t.description,
                            first_seen_at=now,
                            last_seen_at=now,
                            is_new=True,
                        )
                    )
                    new_tables.append(t.name)
                else:
                    # Overlay: refresh reflected structure, keep sticky overrides.
                    row.columns = cols
                    row.reflected_description = t.description
                    row.last_seen_at = now

            # Drop reflected rows for tables that no longer exist in the live DB.
            for r in rows:
                if r.table_name not in live_names and r.source == "reflected":
                    await db.delete(r)

            if header is None:
                header = UserSchema(user_id=user_id)
                db.add(header)
            header.database_name = schema.database_name
            header.dialect = schema.dialect
            header.last_synced_at = now
            header.schema_hash = new_hash

            await db.flush()
            header.source = await self._derive_source(db, user_id)
            await db.commit()

        changed = old_hash != new_hash or bool(new_tables)
        if changed:
            await self._reembed(user_id)

        logger.info(
            "Schema catalog synced",
            user_id=user_id,
            changed=changed,
            new_tables=new_tables,
            total=len(live_names),
        )
        return SyncResult(changed=changed, new_tables=new_tables, total_tables=len(live_names))

    # ── Upload (BYOS path) ────────────────────────────────────────────────────

    async def apply_upload(
        self, user_id: str, raw_json: dict[str, Any], replace: bool
    ) -> UploadResult:
        """Overlay (or replace) the catalog from an uploaded JSON, then re-embed."""
        schema = self._parse_upload(raw_json)
        now = datetime.utcnow()
        new_tables: list[str] = []

        async with self._session_factory() as db:
            header = (
                await db.execute(select(UserSchema).where(UserSchema.user_id == user_id))
            ).scalar_one_or_none()
            if header is None:
                header = UserSchema(user_id=user_id)
                db.add(header)

            if replace:
                await db.execute(
                    delete(UserSchemaTable).where(UserSchemaTable.user_id == user_id)
                )
                existing: dict[tuple[str, str], UserSchemaTable] = {}
            else:
                rows = (
                    await db.execute(
                        select(UserSchemaTable).where(UserSchemaTable.user_id == user_id)
                    )
                ).scalars().all()
                existing = {(r.schema_name, r.table_name): r for r in rows}

            for t in schema.tables:
                cols = [c.model_dump() for c in t.columns]
                key = (t.schema_name, t.name)
                row = existing.get(key)
                if row is None:
                    db.add(
                        UserSchemaTable(
                            user_id=user_id,
                            schema_name=t.schema_name,
                            table_name=t.name,
                            source="uploaded",
                            columns=cols,
                            reflected_description=t.description,
                            first_seen_at=now,
                            last_seen_at=now,
                            is_new=not replace,
                        )
                    )
                    new_tables.append(t.name)
                else:
                    row.source = "uploaded"
                    row.columns = cols
                    if t.description:
                        row.reflected_description = t.description
                    row.last_seen_at = now

            header.database_name = schema.database_name
            header.dialect = schema.dialect
            header.raw_upload_json = raw_json
            header.schema_hash = SchemaIngestionService.compute_schema_hash(schema)
            header.last_synced_at = now

            await db.flush()
            header.source = await self._derive_source(db, user_id)
            await db.commit()

        await self._reembed(user_id)
        logger.info(
            "Schema upload applied",
            user_id=user_id,
            tables=len(schema.tables),
            replaced=replace,
        )
        return UploadResult(tables=len(schema.tables), replaced=replace, new_tables=new_tables)

    # ── Descriptions & seen flags ─────────────────────────────────────────────

    async def set_table_description(
        self, user_id: str, table_id: int, text: str | None
    ) -> dict[str, Any] | None:
        """Set the sticky ``user_description`` on a table and re-embed."""
        async with self._session_factory() as db:
            row = (
                await db.execute(
                    select(UserSchemaTable).where(
                        UserSchemaTable.id == table_id,
                        UserSchemaTable.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            row.user_description = text
            row.is_new = False
            await db.commit()
            result = {
                "id": row.id,
                "schema_name": row.schema_name,
                "table_name": row.table_name,
                "source": row.source,
                "columns": row.columns or [],
                "description": row.user_description or row.reflected_description,
                "is_new": row.is_new,
                "last_seen_at": row.last_seen_at,
            }
        await self._reembed(user_id)
        return result

    async def mark_seen(self, user_id: str, table_ids: list[int]) -> int:
        """Clear the ``is_new`` flag for the given tables. Returns rows updated."""
        if not table_ids:
            return 0
        async with self._session_factory() as db:
            rows = (
                await db.execute(
                    select(UserSchemaTable).where(
                        UserSchemaTable.user_id == user_id,
                        UserSchemaTable.id.in_(table_ids),
                    )
                )
            ).scalars().all()
            for r in rows:
                r.is_new = False
            await db.commit()
            return len(rows)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    async def _derive_source(db: AsyncSession, user_id: str) -> str:
        """Header source from the set of per-table sources: merged/uploaded/reflected."""
        sources = set(
            (
                await db.execute(
                    select(UserSchemaTable.source).where(
                        UserSchemaTable.user_id == user_id
                    )
                )
            ).scalars().all()
        )
        if "uploaded" in sources and "reflected" in sources:
            return "merged"
        if sources == {"uploaded"}:
            return "uploaded"
        return "reflected"

    def _catalog_to_schema(
        self, header: UserSchema | None, rows: list[UserSchemaTable]
    ) -> SchemaMetadata:
        """Build a SchemaMetadata for embedding from catalog rows (effective desc)."""
        tables: list[TableInfo] = []
        for r in rows:
            columns = [ColumnInfo(**c) for c in (r.columns or [])]
            tables.append(
                TableInfo(
                    name=r.table_name,
                    schema_name=r.schema_name,
                    columns=columns,
                    description=r.user_description or r.reflected_description,
                )
            )
        return SchemaMetadata(
            database_name=(header.database_name if header else None) or "user_schema",
            dialect=(header.dialect if header else "postgresql"),
            tables=tables,
        )

    async def _reembed(self, user_id: str) -> None:
        """Rebuild the user's Qdrant chunks from the catalog. Non-fatal on error."""
        try:
            async with self._session_factory() as db:
                header = (
                    await db.execute(
                        select(UserSchema).where(UserSchema.user_id == user_id)
                    )
                ).scalar_one_or_none()
                rows = (
                    await db.execute(
                        select(UserSchemaTable).where(
                            UserSchemaTable.user_id == user_id
                        )
                    )
                ).scalars().all()
            if not rows:
                return
            schema = self._catalog_to_schema(header, list(rows))
            uid = user_id if self._isolation else None
            await self._ingestion.ingest(schema, reset=True, user_id=uid)
        except Exception as exc:
            logger.warning("Catalog re-embed failed", user_id=user_id, error=str(exc))

    @staticmethod
    def _parse_upload(raw: dict[str, Any]) -> SchemaMetadata:
        """Tolerantly parse an uploaded schema JSON into SchemaMetadata.

        Accepts both ``data_type`` and ``type`` for columns and tolerates a
        missing ``database_name`` / ``dialect`` so the documented minimal
        upload format works.
        """
        tables: list[TableInfo] = []
        for t in raw.get("tables", []):
            columns: list[ColumnInfo] = []
            for c in t.get("columns", []):
                columns.append(
                    ColumnInfo(
                        name=c["name"],
                        data_type=str(c.get("data_type") or c.get("type") or "text"),
                        nullable=bool(c.get("nullable", not c.get("primary_key", False))),
                        primary_key=bool(c.get("primary_key", False)),
                        foreign_key=c.get("foreign_key"),
                        description=c.get("description"),
                    )
                )
            tables.append(
                TableInfo(
                    name=t["name"],
                    schema_name=str(t.get("schema_name", "public")),
                    columns=columns,
                    description=t.get("description"),
                )
            )
        return SchemaMetadata(
            database_name=str(raw.get("database_name") or "uploaded_schema"),
            dialect=str(raw.get("dialect") or "postgresql"),
            tables=tables,
        )
