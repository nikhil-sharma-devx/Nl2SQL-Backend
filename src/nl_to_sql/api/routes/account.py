"""F7 - Account management: data retention + account deletion."""
from datetime import datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import AccountDeletion, UserSettings
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/account", tags=["Account"])

_RETENTION_OPTIONS = {"forever", "30d", "7d", "none"}
DELETION_GRACE_DAYS = 7


class RetentionResponse(BaseModel):
    data_retention: str


class RetentionUpdate(BaseModel):
    data_retention: str


class DeleteAccountRequest(BaseModel):
    confirm: str


class DeleteAccountResponse(BaseModel):
    status: str
    purge_after: str


@router.get("/retention", response_model=RetentionResponse, summary="Get data retention setting")
async def get_retention(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> RetentionResponse:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserSettings).where(UserSettings.user_id == current_user.id)
        )
        s = result.scalar_one_or_none()
    return RetentionResponse(data_retention=s.data_retention if s else "forever")


@router.put("/retention", response_model=RetentionResponse, summary="Update data retention setting")
async def update_retention(
    body: RetentionUpdate,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> RetentionResponse:
    if body.data_retention not in _RETENTION_OPTIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"data_retention must be one of {_RETENTION_OPTIONS}",
        )
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserSettings).where(UserSettings.user_id == current_user.id)
        )
        s = result.scalar_one_or_none()
        if s is None:
            s = UserSettings(user_id=current_user.id)
            db.add(s)
        s.data_retention = body.data_retention
        s.updated_at = datetime.utcnow()
        await db.commit()
    return RetentionResponse(data_retention=body.data_retention)


@router.post("/delete", status_code=202, summary="Schedule account deletion (7-day grace period)")
async def request_account_deletion(
    body: DeleteAccountRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> DeleteAccountResponse:
    """Require user to type their email to confirm. Creates a grace-period row."""
    if body.confirm.lower().strip() != current_user.email.lower().strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Confirmation does not match your email address",
        )

    purge_after = datetime.utcnow() + timedelta(days=DELETION_GRACE_DAYS)

    async with session_service._session_factory() as db:
        # Check for existing scheduled deletion
        result = await db.execute(
            select(AccountDeletion).where(AccountDeletion.user_id == current_user.id)
        )
        existing = result.scalar_one_or_none()
        if existing and existing.status == "scheduled":
            return DeleteAccountResponse(
                status="scheduled",
                purge_after=existing.purge_after.isoformat(),
            )

        deletion = AccountDeletion(
            user_id=current_user.id,
            purge_after=purge_after,
            status="scheduled",
        )
        db.add(deletion)
        await db.commit()

    logger.info("account deletion scheduled", user_id=current_user.id, purge_after=purge_after.isoformat())
    return DeleteAccountResponse(status="scheduled", purge_after=purge_after.isoformat())


@router.post("/delete/cancel", summary="Cancel a pending account deletion")
async def cancel_account_deletion(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict[str, Any]:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(AccountDeletion).where(
                AccountDeletion.user_id == current_user.id,
                AccountDeletion.status == "scheduled",
            )
        )
        deletion = result.scalar_one_or_none()
        if deletion is None:
            raise HTTPException(status_code=404, detail="No pending account deletion found")
        deletion.status = "cancelled"
        await db.commit()

    logger.info("account deletion cancelled", user_id=current_user.id)
    return {"status": "cancelled"}
