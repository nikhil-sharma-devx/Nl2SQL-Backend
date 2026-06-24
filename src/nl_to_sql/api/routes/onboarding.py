"""P2 - Onboarding State routes (GET/PATCH — fetch-once per new user)."""
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import OnboardingState
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Onboarding"])

AVAILABLE_ITEMS = [
    "connect_database",
    "run_first_query",
    "save_a_query",
    "pin_a_table",
    "add_custom_instructions",
    "add_glossary_term",
    "explore_templates",
]


class OnboardingStateOut(BaseModel):
    completed_items: list[str]
    available_items: list[str]
    progress_pct: int


class OnboardingPatch(BaseModel):
    completed_items: list[str]


async def _get_or_create(db: Any, user_id: str) -> OnboardingState:
    result = await db.execute(
        select(OnboardingState).where(OnboardingState.user_id == user_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = OnboardingState(user_id=user_id, completed_items=[])
        db.add(row)
        await db.flush()
    return row


def _to_out(row: OnboardingState) -> OnboardingStateOut:
    completed = row.completed_items or []
    total = len(AVAILABLE_ITEMS)
    done = len([c for c in completed if c in AVAILABLE_ITEMS])
    pct = round(done / total * 100) if total else 0
    return OnboardingStateOut(
        completed_items=completed,
        available_items=AVAILABLE_ITEMS,
        progress_pct=pct,
    )


@router.get("/onboarding", response_model=OnboardingStateOut, summary="Get onboarding state")
async def get_onboarding(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> OnboardingStateOut:
    async with session_service._session_factory() as db:
        row = await _get_or_create(db, current_user.id)
        await db.commit()
    return _to_out(row)


@router.patch("/onboarding", response_model=OnboardingStateOut, summary="Update onboarding state")
async def patch_onboarding(
    body: OnboardingPatch,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> OnboardingStateOut:
    async with session_service._session_factory() as db:
        row = await _get_or_create(db, current_user.id)
        row.completed_items = body.completed_items
        row.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(row)
    return _to_out(row)
