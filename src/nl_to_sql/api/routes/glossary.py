"""P2 - Glossary / Business Dictionary: term definitions injected into query prompts."""
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import GlossaryEntry
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/glossary", tags=["Glossary"])

# Max glossary context injected per query to guard token budget (~500 chars ≈ ~125 tokens)
GLOSSARY_INJECT_CHAR_LIMIT = 500


class GlossaryEntryOut(BaseModel):
    id: int
    term: str
    definition: str
    created_at: datetime
    updated_at: datetime


class GlossaryEntryCreate(BaseModel):
    term: str = Field(..., min_length=1, max_length=200)
    definition: str = Field(..., min_length=1)


class GlossaryEntryPatch(BaseModel):
    term: str | None = Field(default=None, min_length=1, max_length=200)
    definition: str | None = Field(default=None, min_length=1)


class GlossaryListResponse(BaseModel):
    items: list[GlossaryEntryOut]
    total: int


def _to_out(e: GlossaryEntry) -> GlossaryEntryOut:
    return GlossaryEntryOut(
        id=e.id,
        term=e.term,
        definition=e.definition,
        created_at=e.created_at,
        updated_at=e.updated_at,
    )


@router.get("", response_model=GlossaryListResponse, summary="List glossary entries")
async def list_glossary(
    search: str | None = Query(default=None, description="Search in term and definition"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> GlossaryListResponse:
    async with session_service._session_factory() as db:
        q = select(GlossaryEntry).where(GlossaryEntry.user_id == current_user.id)
        if search:
            pattern = f"%{search}%"
            q = q.where(
                GlossaryEntry.term.ilike(pattern) | GlossaryEntry.definition.ilike(pattern)
            )
        count_result = await db.execute(select(func.count()).select_from(q.subquery()))
        total = count_result.scalar_one()
        q = q.order_by(GlossaryEntry.term.asc()).limit(limit).offset(offset)
        result = await db.execute(q)
        items = result.scalars().all()
    return GlossaryListResponse(items=[_to_out(e) for e in items], total=total)


@router.post("", response_model=GlossaryEntryOut, status_code=status.HTTP_201_CREATED, summary="Add a glossary entry")
async def create_glossary_entry(
    body: GlossaryEntryCreate,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> GlossaryEntryOut:
    async with session_service._session_factory() as db:
        entry = GlossaryEntry(
            user_id=current_user.id,
            term=body.term,
            definition=body.definition,
        )
        db.add(entry)
        try:
            await db.commit()
            await db.refresh(entry)
        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A glossary entry for '{body.term}' already exists",
            ) from None
    logger.info("glossary entry created", user_id=current_user.id, term=body.term)
    return _to_out(entry)


@router.get("/{entry_id}", response_model=GlossaryEntryOut, summary="Get a glossary entry")
async def get_glossary_entry(
    entry_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> GlossaryEntryOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(GlossaryEntry).where(
                GlossaryEntry.id == entry_id, GlossaryEntry.user_id == current_user.id
            )
        )
        entry = result.scalar_one_or_none()
    if entry is None:
        raise HTTPException(status_code=404, detail="Glossary entry not found")
    return _to_out(entry)


@router.patch("/{entry_id}", response_model=GlossaryEntryOut, summary="Update a glossary entry")
async def patch_glossary_entry(
    entry_id: int,
    body: GlossaryEntryPatch,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> GlossaryEntryOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(GlossaryEntry).where(
                GlossaryEntry.id == entry_id, GlossaryEntry.user_id == current_user.id
            )
        )
        entry = result.scalar_one_or_none()
        if entry is None:
            raise HTTPException(status_code=404, detail="Glossary entry not found")
        if body.term is not None:
            entry.term = body.term
        if body.definition is not None:
            entry.definition = body.definition
        entry.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(entry)
    return _to_out(entry)


@router.delete("/{entry_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a glossary entry")
async def delete_glossary_entry(
    entry_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> None:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(GlossaryEntry).where(
                GlossaryEntry.id == entry_id, GlossaryEntry.user_id == current_user.id
            )
        )
        entry = result.scalar_one_or_none()
        if entry is None:
            raise HTTPException(status_code=404, detail="Glossary entry not found")
        await db.delete(entry)
        await db.commit()
