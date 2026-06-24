"""P2 - Favorite/Pinned Tables: user-pinned tables that get retrieval priority."""
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import FavoritedTable
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/favorited-tables", tags=["Favorited Tables"])


class FavoritedTableOut(BaseModel):
    id: int
    table_name: str
    schema_name: str | None
    note: str | None
    created_at: datetime


class FavoritedTableCreate(BaseModel):
    table_name: str = Field(..., min_length=1, max_length=200)
    schema_name: str | None = Field(default=None, max_length=100)
    note: str | None = None


class FavoritedTablePatch(BaseModel):
    note: str | None = None


def _to_out(f: FavoritedTable) -> FavoritedTableOut:
    return FavoritedTableOut(
        id=f.id,
        table_name=f.table_name,
        schema_name=f.schema_name,
        note=f.note,
        created_at=f.created_at,
    )


@router.get("", response_model=list[FavoritedTableOut], summary="List favorited tables")
async def list_favorited_tables(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> list[FavoritedTableOut]:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(FavoritedTable)
            .where(FavoritedTable.user_id == current_user.id)
            .order_by(FavoritedTable.created_at.desc())
        )
        items = result.scalars().all()
    return [_to_out(f) for f in items]


@router.post("", response_model=FavoritedTableOut, status_code=status.HTTP_201_CREATED, summary="Pin a table")
async def pin_table(
    body: FavoritedTableCreate,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> FavoritedTableOut:
    async with session_service._session_factory() as db:
        f = FavoritedTable(
            user_id=current_user.id,
            table_name=body.table_name,
            schema_name=body.schema_name,
            note=body.note,
        )
        db.add(f)
        try:
            await db.commit()
            await db.refresh(f)
        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Table '{body.table_name}' is already pinned",
            ) from None
    logger.info("table pinned", user_id=current_user.id, table=body.table_name)
    return _to_out(f)


@router.patch("/{table_id}", response_model=FavoritedTableOut, summary="Update note on a pinned table")
async def patch_favorited_table(
    table_id: int,
    body: FavoritedTablePatch,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> FavoritedTableOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(FavoritedTable).where(
                FavoritedTable.id == table_id, FavoritedTable.user_id == current_user.id
            )
        )
        f = result.scalar_one_or_none()
        if f is None:
            raise HTTPException(status_code=404, detail="Pinned table not found")
        f.note = body.note
        await db.commit()
        await db.refresh(f)
    return _to_out(f)


@router.delete("/{table_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Unpin a table")
async def unpin_table(
    table_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> None:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(FavoritedTable).where(
                FavoritedTable.id == table_id, FavoritedTable.user_id == current_user.id
            )
        )
        f = result.scalar_one_or_none()
        if f is None:
            raise HTTPException(status_code=404, detail="Pinned table not found")
        await db.delete(f)
        await db.commit()
