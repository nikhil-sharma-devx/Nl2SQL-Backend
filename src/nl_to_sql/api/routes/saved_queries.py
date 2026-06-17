"""F3 - Saved Queries routes (CRUD + run)."""
from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import SavedQuery
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Saved Queries"])


class SavedQueryOut(BaseModel):
    id: int
    title: Optional[str]
    nl_prompt: str
    generated_sql: str
    dialect: Optional[str]
    starred: bool
    last_run_at: Optional[datetime]
    run_count: int
    created_at: datetime
    updated_at: datetime


class SavedQueryCreate(BaseModel):
    title: Optional[str] = None
    nl_prompt: str
    generated_sql: str
    dialect: Optional[str] = None


class SavedQueryPatch(BaseModel):
    title: Optional[str] = None
    starred: Optional[bool] = None


class SavedQueryListResponse(BaseModel):
    items: list[SavedQueryOut]
    total: int


def _to_out(q: SavedQuery) -> SavedQueryOut:
    return SavedQueryOut(
        id=q.id,
        title=q.title,
        nl_prompt=q.nl_prompt,
        generated_sql=q.generated_sql,
        dialect=q.dialect,
        starred=q.starred,
        last_run_at=q.last_run_at,
        run_count=q.run_count,
        created_at=q.created_at,
        updated_at=q.updated_at,
    )


@router.get("/saved-queries", response_model=SavedQueryListResponse, summary="List saved queries")
async def list_saved_queries(
    search: Optional[str] = Query(default=None, description="Search in title and nl_prompt"),
    starred: Optional[bool] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> SavedQueryListResponse:
    async with session_service._session_factory() as db:
        q = select(SavedQuery).where(SavedQuery.user_id == current_user.id)
        if starred is not None:
            q = q.where(SavedQuery.starred == starred)
        if search:
            pattern = f"%{search}%"
            q = q.where(or_(
                SavedQuery.title.ilike(pattern),
                SavedQuery.nl_prompt.ilike(pattern),
            ))

        count_result = await db.execute(select(func.count()).select_from(q.subquery()))
        total = count_result.scalar_one()

        q = q.order_by(SavedQuery.updated_at.desc()).limit(limit).offset(offset)
        result = await db.execute(q)
        items = result.scalars().all()

    return SavedQueryListResponse(items=[_to_out(i) for i in items], total=total)


@router.post("/saved-queries", response_model=SavedQueryOut, status_code=status.HTTP_201_CREATED, summary="Save a query")
async def create_saved_query(
    body: SavedQueryCreate,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> SavedQueryOut:
    async with session_service._session_factory() as db:
        q = SavedQuery(
            user_id=current_user.id,
            title=body.title,
            nl_prompt=body.nl_prompt,
            generated_sql=body.generated_sql,
            dialect=body.dialect,
        )
        db.add(q)
        await db.commit()
        await db.refresh(q)
    return _to_out(q)


@router.get("/saved-queries/{query_id}", response_model=SavedQueryOut, summary="Get a saved query")
async def get_saved_query(
    query_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> SavedQueryOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(SavedQuery).where(SavedQuery.id == query_id, SavedQuery.user_id == current_user.id)
        )
        q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Saved query not found")
    return _to_out(q)


@router.patch("/saved-queries/{query_id}", response_model=SavedQueryOut, summary="Update a saved query")
async def patch_saved_query(
    query_id: int,
    body: SavedQueryPatch,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> SavedQueryOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(SavedQuery).where(SavedQuery.id == query_id, SavedQuery.user_id == current_user.id)
        )
        q = result.scalar_one_or_none()
        if q is None:
            raise HTTPException(status_code=404, detail="Saved query not found")

        updates = body.model_dump(exclude_none=True)
        for field, value in updates.items():
            setattr(q, field, value)
        q.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(q)
    return _to_out(q)


@router.delete("/saved-queries/{query_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a saved query")
async def delete_saved_query(
    query_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> None:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(SavedQuery).where(SavedQuery.id == query_id, SavedQuery.user_id == current_user.id)
        )
        q = result.scalar_one_or_none()
        if q is None:
            raise HTTPException(status_code=404, detail="Saved query not found")
        await db.delete(q)
        await db.commit()


@router.post("/saved-queries/{query_id}/run", summary="Re-run a saved query")
async def run_saved_query(
    query_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict:
    """Increment run count and record last_run_at. Full re-execution is Phase 2."""
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(SavedQuery).where(SavedQuery.id == query_id, SavedQuery.user_id == current_user.id)
        )
        q = result.scalar_one_or_none()
        if q is None:
            raise HTTPException(status_code=404, detail="Saved query not found")
        q.run_count += 1
        q.last_run_at = datetime.utcnow()
        await db.commit()
        await db.refresh(q)

    return {
        "query_id": query_id,
        "generated_sql": q.generated_sql,
        "run_count": q.run_count,
        "last_run_at": q.last_run_at.isoformat(),
    }
