"""F2 + F4 - User Settings routes (GET/PATCH /api/v1/settings)."""
from datetime import datetime
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import UserSettings
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Settings"])

_KEYWORD_CASE = {"upper", "lower"}
_CTE_PREF = {"cte", "subquery"}
_ALIAS_STYLE = {"as", "implicit"}
_RETENTION = {"forever", "30d", "7d", "none"}


class SettingsResponse(BaseModel):
    sql_keyword_case: str
    sql_cte_pref: str
    sql_alias_style: str
    sql_indent: int
    default_dialect: str | None
    max_result_rows: int
    auto_execute: bool
    default_model: str | None
    data_retention: str


class SettingsPatchRequest(BaseModel):
    sql_keyword_case: str | None = None
    sql_cte_pref: str | None = None
    sql_alias_style: str | None = None
    sql_indent: int | None = Field(default=None, ge=1, le=8)
    default_dialect: str | None = None
    max_result_rows: int | None = Field(default=None, ge=1, le=100000)
    auto_execute: bool | None = None
    default_model: str | None = None
    data_retention: str | None = None


def _default_settings() -> SettingsResponse:
    return SettingsResponse(
        sql_keyword_case="upper",
        sql_cte_pref="cte",
        sql_alias_style="as",
        sql_indent=2,
        default_dialect=None,
        max_result_rows=1000,
        auto_execute=False,
        default_model=None,
        data_retention="forever",
    )


def _to_response(s: UserSettings) -> SettingsResponse:
    return SettingsResponse(
        sql_keyword_case=s.sql_keyword_case,
        sql_cte_pref=s.sql_cte_pref,
        sql_alias_style=s.sql_alias_style,
        sql_indent=s.sql_indent,
        default_dialect=s.default_dialect,
        max_result_rows=s.max_result_rows,
        auto_execute=s.auto_execute,
        default_model=s.default_model,
        data_retention=s.data_retention,
    )


@router.get("/settings", response_model=SettingsResponse, summary="Get user settings")
async def get_settings(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> SettingsResponse:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserSettings).where(UserSettings.user_id == current_user.id)
        )
        s = result.scalar_one_or_none()

    return _to_response(s) if s else _default_settings()


@router.patch("/settings", response_model=SettingsResponse, summary="Update user settings")
async def patch_settings(
    body: SettingsPatchRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> SettingsResponse:
    """Partial update of user settings. Validates enum fields server-side."""
    errors = []
    if body.sql_keyword_case is not None and body.sql_keyword_case not in _KEYWORD_CASE:
        errors.append(f"sql_keyword_case must be one of {_KEYWORD_CASE}")
    if body.sql_cte_pref is not None and body.sql_cte_pref not in _CTE_PREF:
        errors.append(f"sql_cte_pref must be one of {_CTE_PREF}")
    if body.sql_alias_style is not None and body.sql_alias_style not in _ALIAS_STYLE:
        errors.append(f"sql_alias_style must be one of {_ALIAS_STYLE}")
    if body.data_retention is not None and body.data_retention not in _RETENTION:
        errors.append(f"data_retention must be one of {_RETENTION}")
    if errors:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="; ".join(errors))

    async with session_service._session_factory() as db:
        result = await db.execute(
            select(UserSettings).where(UserSettings.user_id == current_user.id)
        )
        s = result.scalar_one_or_none()

        if s is None:
            s = UserSettings(user_id=current_user.id)
            db.add(s)

        updates = body.model_dump(exclude_none=True)
        for field, value in updates.items():
            setattr(s, field, value)
        s.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(s)

    logger.info("settings updated", user_id=current_user.id, fields=list(updates.keys()))
    return _to_response(s)
