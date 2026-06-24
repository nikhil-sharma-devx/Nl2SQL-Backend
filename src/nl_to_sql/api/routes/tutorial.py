"""P2 - Tutorial progress and onboarding checklist routes."""
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import OnboardingState, TutorialProgress
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Tutorial & Onboarding"])

# Canonical step IDs for the walkthrough tutorial
TUTORIAL_STEPS = [
    "connect_database",
    "run_first_query",
    "save_query",
    "explore_history",
    "customize_settings",
]

# Canonical item IDs for the onboarding checklist
ONBOARDING_ITEMS = [
    "connect_database",
    "run_first_query",
    "save_a_query",
    "pin_a_table",
    "add_custom_instructions",
    "add_glossary_term",
    "explore_templates",
]


# ── Tutorial / Walkthrough ────────────────────────────────────────────────────


class TutorialProgressOut(BaseModel):
    completed_steps: list[str]
    available_steps: list[str]
    dismissed_at: datetime | None
    is_complete: bool


class TutorialProgressPatch(BaseModel):
    completed_steps: list[str] | None = None
    dismissed: bool | None = None


@router.get("/tutorial", response_model=TutorialProgressOut, summary="Get tutorial progress")
@router.get("/tutorial/progress", response_model=TutorialProgressOut, include_in_schema=False)
async def get_tutorial_progress(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> TutorialProgressOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(TutorialProgress).where(TutorialProgress.user_id == current_user.id)
        )
        row = result.scalar_one_or_none()

    completed = row.completed_steps if row else []
    dismissed_at = row.dismissed_at if row else None
    return TutorialProgressOut(
        completed_steps=completed,
        available_steps=TUTORIAL_STEPS,
        dismissed_at=dismissed_at,
        is_complete=set(TUTORIAL_STEPS).issubset(set(completed)),
    )


@router.patch("/tutorial", response_model=TutorialProgressOut, summary="Update tutorial progress")
@router.patch("/tutorial/progress", response_model=TutorialProgressOut, include_in_schema=False)
async def patch_tutorial_progress(
    body: TutorialProgressPatch,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> TutorialProgressOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(TutorialProgress).where(TutorialProgress.user_id == current_user.id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = TutorialProgress(user_id=current_user.id, completed_steps=[])
            db.add(row)

        if body.completed_steps is not None:
            # Only accept known step IDs; merge with existing
            valid = [s for s in body.completed_steps if s in TUTORIAL_STEPS]
            existing = set(row.completed_steps or [])
            row.completed_steps = list(existing | set(valid))

        if body.dismissed is True:
            row.dismissed_at = datetime.utcnow()
        elif body.dismissed is False:
            row.dismissed_at = None

        row.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(row)

    return TutorialProgressOut(
        completed_steps=row.completed_steps,
        available_steps=TUTORIAL_STEPS,
        dismissed_at=row.dismissed_at,
        is_complete=set(TUTORIAL_STEPS).issubset(set(row.completed_steps)),
    )


@router.delete("/tutorial", status_code=204, summary="Reset tutorial progress")
@router.delete("/tutorial/progress", status_code=204, include_in_schema=False)
async def reset_tutorial_progress(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> None:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(TutorialProgress).where(TutorialProgress.user_id == current_user.id)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            row.completed_steps = []
            row.dismissed_at = None
            row.updated_at = datetime.utcnow()
            await db.commit()


# ── Onboarding Checklist ──────────────────────────────────────────────────────


class OnboardingOut(BaseModel):
    completed_items: list[str]
    available_items: list[str]
    progress_pct: int


class OnboardingPatch(BaseModel):
    completed_items: list[str]


@router.get("/onboarding", response_model=OnboardingOut, summary="Get onboarding checklist state")
async def get_onboarding(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> OnboardingOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(OnboardingState).where(OnboardingState.user_id == current_user.id)
        )
        row = result.scalar_one_or_none()

    completed = row.completed_items if row else []
    pct = int(len(set(completed) & set(ONBOARDING_ITEMS)) / len(ONBOARDING_ITEMS) * 100)
    return OnboardingOut(completed_items=completed, available_items=ONBOARDING_ITEMS, progress_pct=pct)


@router.patch("/onboarding", response_model=OnboardingOut, summary="Mark onboarding items complete")
async def patch_onboarding(
    body: OnboardingPatch,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> OnboardingOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(OnboardingState).where(OnboardingState.user_id == current_user.id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = OnboardingState(user_id=current_user.id, completed_items=[])
            db.add(row)

        valid = [item for item in body.completed_items if item in ONBOARDING_ITEMS]
        existing = set(row.completed_items or [])
        row.completed_items = list(existing | set(valid))
        row.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(row)

    pct = int(len(set(row.completed_items) & set(ONBOARDING_ITEMS)) / len(ONBOARDING_ITEMS) * 100)
    return OnboardingOut(completed_items=row.completed_items, available_items=ONBOARDING_ITEMS, progress_pct=pct)
