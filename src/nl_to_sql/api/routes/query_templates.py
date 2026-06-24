"""P2 - Query Templates: parameterized SQL patterns with {{placeholder}} variables."""
import re
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import QueryTemplate
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/query-templates", tags=["Query Templates"])

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class TemplateParameter(BaseModel):
    name: str
    type: str = "string"
    description: str | None = None
    default: str | None = None


class QueryTemplateOut(BaseModel):
    id: int
    name: str
    description: str | None
    template_nl: str
    template_sql: str
    parameters: list[TemplateParameter]
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class QueryTemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    template_nl: str = Field(..., min_length=1)
    template_sql: str = Field(..., min_length=1)
    parameters: list[TemplateParameter] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class QueryTemplatePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    template_nl: str | None = None
    template_sql: str | None = None
    parameters: list[TemplateParameter] | None = None
    tags: list[str] | None = None


class QueryTemplateListResponse(BaseModel):
    items: list[QueryTemplateOut]
    total: int


class TemplateRenderRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class TemplateRenderResponse(BaseModel):
    nl: str
    sql: str
    missing_params: list[str]


def _to_out(t: QueryTemplate) -> QueryTemplateOut:
    return QueryTemplateOut(
        id=t.id,
        name=t.name,
        description=t.description,
        template_nl=t.template_nl,
        template_sql=t.template_sql,
        parameters=[TemplateParameter(**p) for p in (t.parameters or [])],
        tags=t.tags or [],
        created_at=t.created_at,
        updated_at=t.updated_at,
    )


@router.get("", response_model=QueryTemplateListResponse, summary="List query templates")
async def list_templates(
    search: str | None = Query(default=None),
    tag: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> QueryTemplateListResponse:
    async with session_service._session_factory() as db:
        q = select(QueryTemplate).where(QueryTemplate.user_id == current_user.id)
        if search:
            pattern = f"%{search}%"
            q = q.where(QueryTemplate.name.ilike(pattern))

        count_result = await db.execute(select(func.count()).select_from(q.subquery()))
        total = count_result.scalar_one()

        q = q.order_by(QueryTemplate.updated_at.desc()).limit(limit).offset(offset)
        result = await db.execute(q)
        items = result.scalars().all()

    # Filter by tag in Python (JSON array); acceptable at low counts
    if tag:
        items = [t for t in items if tag in (t.tags or [])]
        total = len(items)

    return QueryTemplateListResponse(items=[_to_out(t) for t in items], total=total)


@router.post("", response_model=QueryTemplateOut, status_code=status.HTTP_201_CREATED, summary="Create a query template")
async def create_template(
    body: QueryTemplateCreate,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> QueryTemplateOut:
    async with session_service._session_factory() as db:
        t = QueryTemplate(
            user_id=current_user.id,
            name=body.name,
            description=body.description,
            template_nl=body.template_nl,
            template_sql=body.template_sql,
            parameters=[p.model_dump() for p in body.parameters],
            tags=body.tags,
        )
        db.add(t)
        await db.commit()
        await db.refresh(t)
    logger.info("query template created", user_id=current_user.id, name=body.name)
    return _to_out(t)


@router.get("/{template_id}", response_model=QueryTemplateOut, summary="Get a query template")
async def get_template(
    template_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> QueryTemplateOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(QueryTemplate).where(
                QueryTemplate.id == template_id, QueryTemplate.user_id == current_user.id
            )
        )
        t = result.scalar_one_or_none()
    if t is None:
        raise HTTPException(status_code=404, detail="Query template not found")
    return _to_out(t)


@router.patch("/{template_id}", response_model=QueryTemplateOut, summary="Update a query template")
async def patch_template(
    template_id: int,
    body: QueryTemplatePatch,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> QueryTemplateOut:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(QueryTemplate).where(
                QueryTemplate.id == template_id, QueryTemplate.user_id == current_user.id
            )
        )
        t = result.scalar_one_or_none()
        if t is None:
            raise HTTPException(status_code=404, detail="Query template not found")

        if body.name is not None:
            t.name = body.name
        if body.description is not None:
            t.description = body.description
        if body.template_nl is not None:
            t.template_nl = body.template_nl
        if body.template_sql is not None:
            t.template_sql = body.template_sql
        if body.parameters is not None:
            t.parameters = [p.model_dump() for p in body.parameters]
        if body.tags is not None:
            t.tags = body.tags
        t.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(t)
    return _to_out(t)


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a query template")
async def delete_template(
    template_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> None:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(QueryTemplate).where(
                QueryTemplate.id == template_id, QueryTemplate.user_id == current_user.id
            )
        )
        t = result.scalar_one_or_none()
        if t is None:
            raise HTTPException(status_code=404, detail="Query template not found")
        await db.delete(t)
        await db.commit()


@router.post("/{template_id}/render", response_model=TemplateRenderResponse, summary="Render a template with values")
async def render_template(
    template_id: int,
    body: TemplateRenderRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> TemplateRenderResponse:
    """Substitute {{param}} placeholders with provided values. Returns rendered NL and SQL."""
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(QueryTemplate).where(
                QueryTemplate.id == template_id, QueryTemplate.user_id == current_user.id
            )
        )
        t = result.scalar_one_or_none()
    if t is None:
        raise HTTPException(status_code=404, detail="Query template not found")

    # Collect all placeholders from both templates
    all_params = set(_PLACEHOLDER_RE.findall(t.template_nl)) | set(_PLACEHOLDER_RE.findall(t.template_sql))

    # Build defaults map from parameter definitions
    defaults = {p["name"]: p.get("default", "") for p in (t.parameters or [])}

    def _substitute(text: str) -> str:
        def _replace(m: re.Match[str]) -> str:
            key = m.group(1)
            return str(body.values.get(key, defaults.get(key, m.group(0))))
        return _PLACEHOLDER_RE.sub(_replace, text)

    rendered_nl = _substitute(t.template_nl)
    rendered_sql = _substitute(t.template_sql)

    # Find still-unresolved placeholders (params with no value and no default)
    missing = [
        p for p in all_params
        if p not in body.values and not defaults.get(p)
    ]

    return TemplateRenderResponse(nl=rendered_nl, sql=rendered_sql, missing_params=missing)
