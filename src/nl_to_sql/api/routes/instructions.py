"""F1 - Custom Instructions routes (GET/PUT /api/v1/instructions)."""
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import UserInstructions, UserInstructionsHistory
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Instructions"])

INSTRUCTIONS_CHAR_CAP = 2000


class InstructionsResponse(BaseModel):
    content: str
    enabled: bool
    char_count: int
    updated_at: datetime


class InstructionsUpdateRequest(BaseModel):
    content: str = Field(default="", max_length=INSTRUCTIONS_CHAR_CAP)
    enabled: bool = True

    @field_validator("content")
    @classmethod
    def validate_length(cls, v: str) -> str:
        if len(v) > INSTRUCTIONS_CHAR_CAP:
            raise ValueError(f"content must not exceed {INSTRUCTIONS_CHAR_CAP} characters")
        return v


@router.get(
    "/instructions",
    response_model=InstructionsResponse,
    summary="Get custom instructions",
)
async def get_instructions(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> InstructionsResponse:
    """Return the user's current custom instructions, or empty defaults."""
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserInstructions).where(UserInstructions.user_id == current_user.id)
        )
        instr = result.scalar_one_or_none()

    if instr is None:
        return InstructionsResponse(
            content="",
            enabled=True,
            char_count=0,
            updated_at=datetime.utcnow(),
        )

    return InstructionsResponse(
        content=instr.content,
        enabled=instr.enabled,
        char_count=instr.char_count,
        updated_at=instr.updated_at,
    )


@router.put(
    "/instructions",
    response_model=InstructionsResponse,
    summary="Update custom instructions",
)
async def update_instructions(
    body: InstructionsUpdateRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> InstructionsResponse:
    """Upsert custom instructions. Saves previous content to history for recoverability."""
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserInstructions).where(UserInstructions.user_id == current_user.id)
        )
        instr = result.scalar_one_or_none()

        if instr is not None and instr.content:
            # Archive previous version
            history = UserInstructionsHistory(
                user_id=current_user.id,
                content=instr.content,
                replaced_at=datetime.utcnow(),
            )
            db.add(history)

        if instr is None:
            instr = UserInstructions(user_id=current_user.id)
            db.add(instr)

        instr.content = body.content
        instr.enabled = body.enabled
        instr.char_count = len(body.content)
        instr.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(instr)

    logger.info("instructions updated", user_id=current_user.id, char_count=instr.char_count)

    return InstructionsResponse(
        content=instr.content,
        enabled=instr.enabled,
        char_count=instr.char_count,
        updated_at=instr.updated_at,
    )
