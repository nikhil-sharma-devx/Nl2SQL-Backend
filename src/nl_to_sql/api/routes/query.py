"""Query route — POST /api/v1/query."""
import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import (
    get_current_user,
    get_llm_provider,
    get_request_orchestrator,
    get_session_service,
)
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.core.models.query import QueryRequest, QueryResponse
from nl_to_sql.services.chat_session_service import ChatSessionService
from nl_to_sql.services.query_orchestrator import QueryOrchestrator
from nl_to_sql.services.question_suggestion_service import (
    QuestionSuggestionService,
    SuggestionRequest,
    SuggestionResponse,
)
from nl_to_sql.services.sql_explanation_service import SQLExplanationService

router = APIRouter(prefix="/api/v1", tags=["Query"])
_settings = get_settings()
_rate = f"{_settings.rate_limit_requests}/minute"


async def _load_user_settings(user_id: str, session_factory: Any) -> Any:
    """Fetch the UserSettings row for this user (returns None if not set yet)."""
    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import UserSettings
    async with session_factory() as db:
        result = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
        return result.scalar_one_or_none()


async def _load_user_instructions(user_id: str, session_factory: Any) -> str | None:
    """Return the user's custom instructions text if enabled and non-empty."""
    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import UserInstructions
    async with session_factory() as db:
        result = await db.execute(select(UserInstructions).where(UserInstructions.user_id == user_id))
        instr = result.scalar_one_or_none()
    if instr and instr.enabled and instr.content and instr.content.strip():
        return str(instr.content.strip())
    return None


async def _load_glossary_context(user_id: str, session_factory: Any, char_limit: int = 500) -> str | None:
    """Return a compact glossary block to inject into the prompt, capped at char_limit."""
    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import GlossaryEntry
    async with session_factory() as db:
        result = await db.execute(
            select(GlossaryEntry)
            .where(GlossaryEntry.user_id == user_id)
            .order_by(GlossaryEntry.term.asc())
            .limit(50)
        )
        entries = result.scalars().all()
    if not entries:
        return None
    lines = [f"- {e.term}: {e.definition}" for e in entries]
    block = "Business glossary:\n" + "\n".join(lines)
    if len(block) > char_limit:
        block = block[:char_limit].rsplit("\n", 1)[0]  # don't cut mid-line
    return block


async def _load_favorited_tables_hint(user_id: str, session_factory: Any) -> str | None:
    """Return a retrieval hint listing the user's pinned tables."""
    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import FavoritedTable
    async with session_factory() as db:
        result = await db.execute(
            select(FavoritedTable)
            .where(FavoritedTable.user_id == user_id)
            .order_by(FavoritedTable.created_at.desc())
            .limit(20)
        )
        tables = result.scalars().all()
    if not tables:
        return None
    names = ", ".join(t.table_name for t in tables)
    return f"User's prioritized tables (prefer these when relevant): {names}"


def _build_style_hints(user_settings: Any) -> dict[str, Any] | None:
    """Convert a UserSettings ORM row into a style_hints dict for the generator."""
    if user_settings is None:
        return None
    return {
        "cte_pref": user_settings.sql_cte_pref,
        "keyword_case": user_settings.sql_keyword_case,
        "alias_style": user_settings.sql_alias_style,
        "indent": user_settings.sql_indent,
        "max_result_rows": user_settings.max_result_rows,
    }


def _apply_user_settings(response: QueryResponse, user_settings: Any) -> QueryResponse:
    """Post-process a QueryResponse using the user's saved style + limit preferences.

    - Formats SQL with keyword case, indent, alias style
    - Uses user's default_dialect if none was set on the request
    - Truncates execution_result to max_result_rows
    """
    if user_settings is None:
        return response

    from nl_to_sql.services.sql_format import format_sql

    # Format the generated SQL according to user style preferences
    if response.sql and response.is_valid:
        try:
            response.sql = format_sql(
                response.sql,
                dialect=response.dialect or None,
                keyword_case=user_settings.sql_keyword_case,
                indent=user_settings.sql_indent,
                alias_style=user_settings.sql_alias_style,
            )
        except Exception:
            pass  # formatting failure must never block a response

    # Enforce max_result_rows
    if response.execution_result is not None:
        max_rows = user_settings.max_result_rows or 1000
        response.execution_result = response.execution_result[:max_rows]

    return response


class ExplainRequest(BaseModel):
    """Request body for SQL explanation."""

    sql: str = Field(..., max_length=50_000, description="The SQL query to explain")


class ExplainResponse(BaseModel):
    """Response body for SQL explanation."""

    sql: str = Field(..., description="The original SQL query")
    explanation: str = Field(..., description="Plain-English explanation of the SQL query")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Convert a natural language question to SQL",
    description=(
        "Accepts a plain-English question, retrieves relevant schema context "
        "from the vector store, and uses an LLM to generate a valid SQL query. "
        "Optionally executes the query against the configured database."
    ),
)
@limiter.limit(_rate)
async def nl_to_sql_query(
    request: Request,  # required by SlowAPI for IP extraction
    body: QueryRequest,
    background_tasks: BackgroundTasks,
    orchestrator: QueryOrchestrator = Depends(get_request_orchestrator),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> QueryResponse:
    """Main endpoint: NL question → SQL response."""
    user_settings = await _load_user_settings(current_user.id, session_service._session_factory)

    # Fill in the user's default dialect when the request doesn't specify one
    if not body.dialect and user_settings and user_settings.default_dialect:
        body.dialect = user_settings.default_dialect

    style_hints = _build_style_hints(user_settings)
    model_override = user_settings.default_model if user_settings else None

    custom_instructions, glossary_ctx, tables_hint = await asyncio.gather(
        _load_user_instructions(current_user.id, session_service._session_factory),
        _load_glossary_context(current_user.id, session_service._session_factory),
        _load_favorited_tables_hint(current_user.id, session_service._session_factory),
    )

    # Merge glossary and pinned-table hints into custom_instructions
    extra_parts = [p for p in (glossary_ctx, tables_hint) if p]
    if extra_parts:
        base = custom_instructions or ""
        custom_instructions = "\n\n".join(filter(None, [base, *extra_parts]))

    response = await orchestrator.run(body, style_hints=style_hints, model_override=model_override, custom_instructions=custom_instructions)
    response = _apply_user_settings(response, user_settings)
    background_tasks.add_task(_record_metrics, current_user.id, response.tokens_used, session_service._session_factory)
    return response


async def _record_metrics(user_id: str, tokens_used: int, session_factory: Any) -> None:
    """Background task: write a QueryMetrics row for billing/usage tracking."""
    import structlog
    log = structlog.get_logger(__name__)
    try:
        from nl_to_sql.infrastructure.database.models import QueryMetrics
        async with session_factory() as db:
            db.add(QueryMetrics(user_id=user_id, tokens_in=tokens_used, tokens_out=0))
            await db.commit()
        log.debug("metrics recorded", user_id=user_id, tokens=tokens_used)
    except Exception as exc:
        log.warning("failed to record query metrics", user_id=user_id, error=str(exc))


async def _stream_sql_generation(
    orchestrator: QueryOrchestrator,
    body: QueryRequest,
    user_settings: Any = None,
    style_hints: dict[str, Any] | None = None,
    model_override: str | None = None,
    custom_instructions: str | None = None,
    user_id: str | None = None,
    session_factory: Any = None,
) -> AsyncGenerator[str, None]:
    """Async generator for streaming SQL generation."""
    import json

    from fastapi.encoders import jsonable_encoder

    try:
        async for chunk in orchestrator.run_stream(body, style_hints=style_hints, model_override=model_override, custom_instructions=custom_instructions):
            # Apply user settings and record metrics on the final complete chunk
            if (
                isinstance(chunk, dict)
                and chunk.get("status") == "complete"
                and isinstance(chunk.get("data"), dict)
            ):
                if user_settings is not None:
                    from nl_to_sql.core.models.query import QueryResponse
                    try:
                        resp_obj = QueryResponse(**chunk["data"])
                        resp_obj = _apply_user_settings(resp_obj, user_settings)
                        chunk["data"] = resp_obj.model_dump()
                    except Exception:
                        pass  # never block stream on formatting failure

                # Record usage metrics inline — safe to await here, negligible latency
                if user_id and session_factory:
                    tokens = chunk["data"].get("tokens_used", 0) or 0
                    await _record_metrics(user_id, tokens, session_factory)

            safe_chunk = jsonable_encoder(chunk)
            yield f"data: {json.dumps(safe_chunk)}\n\n"
    except Exception as exc:
        error_data = json.dumps({
            "status": "error",
            "error": str(exc),
            "type": type(exc).__name__
        })
        yield f"data: {error_data}\n\n"
    yield "data: [DONE]\n\n"


@router.post(
    "/query/stream",
    summary="Convert a natural language question to SQL (streaming)",
    description=(
        "Same as /query but streams the SQL generation process in real-time "
        "for faster perceived latency. Returns Server-Sent Events (SSE)."
    ),
)
@limiter.limit(_rate)
async def nl_to_sql_query_stream(
    request: Request,
    body: QueryRequest,
    orchestrator: QueryOrchestrator = Depends(get_request_orchestrator),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> StreamingResponse:
    """Streaming endpoint: NL question → SQL response (streamed)."""
    user_settings = await _load_user_settings(current_user.id, session_service._session_factory)

    # Fill in the user's default dialect when the request doesn't specify one
    if not body.dialect and user_settings and user_settings.default_dialect:
        body.dialect = user_settings.default_dialect

    style_hints = _build_style_hints(user_settings)
    model_override = user_settings.default_model if user_settings else None

    custom_instructions, glossary_ctx, tables_hint = await asyncio.gather(
        _load_user_instructions(current_user.id, session_service._session_factory),
        _load_glossary_context(current_user.id, session_service._session_factory),
        _load_favorited_tables_hint(current_user.id, session_service._session_factory),
    )
    extra_parts = [p for p in (glossary_ctx, tables_hint) if p]
    if extra_parts:
        base = custom_instructions or ""
        custom_instructions = "\n\n".join(filter(None, [base, *extra_parts]))

    return StreamingResponse(
        _stream_sql_generation(
            orchestrator, body,
            user_settings=user_settings,
            style_hints=style_hints,
            model_override=model_override,
            custom_instructions=custom_instructions,
            user_id=current_user.id,
            session_factory=session_service._session_factory,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/query/explain",
    response_model=ExplainResponse,
    summary="Explain a SQL query in plain English",
    description=(
        "Accepts a SQL query and returns a natural-language explanation "
        "of what the query does, suitable for non-technical users."
    ),
)
@limiter.limit(_rate)
async def explain_sql(
    request: Request,
    body: ExplainRequest,
    llm_provider: Any = Depends(get_llm_provider),
    current_user: UserPublic = Depends(get_current_user),
) -> ExplainResponse:
    """Endpoint: SQL → plain-English explanation."""
    explanation_service = SQLExplanationService(llm_provider=llm_provider)
    explanation = await explanation_service.explain(body.sql)
    return ExplainResponse(sql=body.sql, explanation=explanation)


@router.post(
    "/query/suggestions",
    response_model=SuggestionResponse,
    summary="Generate follow-up question suggestions",
    description=(
        "Accepts the original question and generated SQL, then returns "
        "2-3 contextually relevant follow-up questions to help users explore deeper."
    ),
)
@limiter.limit(_rate)
async def get_query_suggestions(
    request: Request,
    body: SuggestionRequest,
    llm_provider: Any = Depends(get_llm_provider),
    current_user: UserPublic = Depends(get_current_user),
) -> SuggestionResponse:
    """Endpoint: Generate suggested follow-up questions."""
    suggestion_service = QuestionSuggestionService(llm_provider=llm_provider)
    return await suggestion_service.generate_suggestions(body)


class ExecuteRequest(BaseModel):
    """Request body for executing custom SQL."""

    sql: str = Field(..., max_length=50_000, description="The SQL query to execute")
    dialect: str | None = Field(None, description="SQL dialect (default: from settings)")


class SaveVersionRequest(BaseModel):
    """Request body for saving a new SQL version."""

    message_id: int = Field(..., description="The message ID this version belongs to")
    sql: str = Field(..., max_length=50_000, description="The edited SQL query")
    results: list[dict[str, Any]] | None = Field(None, description="Execution results")
    success: bool = Field(..., description="Whether execution succeeded")


class SaveVersionResponse(BaseModel):
    """Response body for saving a SQL version."""

    success: bool = Field(..., description="Whether the version was saved")
    version_number: int = Field(..., description="The version number assigned")
    total_versions: int = Field(..., description="Total versions for this message")


class ExecuteResponse(BaseModel):
    """Response body for SQL execution."""

    sql: str = Field(..., description="The executed SQL query")
    success: bool = Field(..., description="Whether execution succeeded")
    results: list[dict[str, Any]] | None = Field(None, description="Query results (if successful)")
    error: str | None = Field(None, description="Error message (if failed)")
    row_count: int = Field(0, description="Number of rows returned")
    truncated: bool = Field(False, description="True when results were capped at the server row limit")


@router.post(
    "/query/execute",
    response_model=ExecuteResponse,
    summary="Execute a SQL query against the database",
    description=(
        "Accepts a SQL query (user-edited or generated) and executes it "
        "against the configured database. Returns results or error message."
    ),
)
@limiter.limit(_rate)
async def execute_sql(
    request: Request,
    body: ExecuteRequest,
    orchestrator: QueryOrchestrator = Depends(get_request_orchestrator),
    current_user: UserPublic = Depends(get_current_user),
) -> ExecuteResponse:
    """Endpoint: Execute custom SQL query."""
    # Access the database client from orchestrator
    db_client = orchestrator._db_client

    if not db_client:
        return ExecuteResponse(
            sql=body.sql,
            success=False,
            error="Database client not configured",
        )

    # Validate: only allow SELECT statements, block dangerous functions
    from nl_to_sql.services.sql_validator import SQLValidatorService
    _validator = SQLValidatorService(dialect=body.dialect or _settings.sql_dialect)
    validation = _validator.validate(body.sql)
    if not validation.is_valid:
        return ExecuteResponse(
            sql=body.sql,
            success=False,
            error=f"SQL rejected: {'; '.join(validation.errors)}",
        )

    try:
        from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
        results = await db_client.execute_sql(validation.normalised_sql or body.sql)
        truncated = len(results) >= AsyncDatabaseClient.MAX_ROWS if results else False
        return ExecuteResponse(
            sql=body.sql,
            success=True,
            results=results,
            row_count=len(results) if results else 0,
            truncated=truncated,
        )
    except Exception as exc:
        return ExecuteResponse(
            sql=body.sql,
            success=False,
            error=str(exc),
        )


@router.post(
    "/query/execute/stream",
    summary="Execute SQL and stream large result sets",
    description=(
        "Executes a SQL query and streams results as Server-Sent Events (SSE), "
        "one batch at a time. Avoids buffering large datasets in memory."
    ),
)
@limiter.limit(_rate)
async def execute_sql_stream(
    request: Request,
    body: ExecuteRequest,
    orchestrator: QueryOrchestrator = Depends(get_request_orchestrator),
    current_user: UserPublic = Depends(get_current_user),
) -> StreamingResponse:
    """Stream large SQL result sets as SSE batches."""
    import json

    db_client = orchestrator._db_client

    async def _generate() -> AsyncGenerator[str, None]:
        if not db_client:
            yield f"data: {json.dumps({'error': 'Database client not configured'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        # Validate before streaming
        from nl_to_sql.services.sql_validator import SQLValidatorService
        _val = SQLValidatorService(dialect=body.dialect or _settings.sql_dialect)
        _vr = _val.validate(body.sql)
        if not _vr.is_valid:
            yield f"data: {json.dumps({'error': 'SQL rejected: ' + '; '.join(_vr.errors)})}\n\n"
            yield "data: [DONE]\n\n"
            return
        try:
            row_count = 0
            async for batch in db_client.execute_sql_stream(_vr.normalised_sql or body.sql):
                row_count += len(batch)
                yield f"data: {json.dumps({'rows': batch, 'batch_row_count': len(batch)})}\n\n"
            yield f"data: {json.dumps({'status': 'complete', 'total_rows': row_count})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/query/save-version",
    response_model=SaveVersionResponse,
    summary="Save a new version of edited SQL",
    description=(
        "Saves a new version when user edits and re-runs SQL. "
        "Returns the version number and total versions for tracking."
    ),
)
@limiter.limit(_rate)
async def save_sql_version(
    request: Request,
    body: SaveVersionRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> SaveVersionResponse:
    """Endpoint: Save edited SQL as a new version (persisted to database)."""
    from datetime import datetime

    import structlog
    from sqlalchemy import func, select

    from nl_to_sql.infrastructure.database.models import SqlVersion

    logger = structlog.get_logger(__name__)

    async with session_service._session_factory() as db_sess:
        from fastapi import HTTPException

        from nl_to_sql.infrastructure.database.models import ChatMessage, ChatSession

        ownership = await db_sess.execute(
            select(ChatMessage.id)
            .join(ChatSession, ChatSession.id == ChatMessage.session_id)
            .where(
                ChatMessage.id == body.message_id,
                ChatSession.user_id == current_user.id,
            )
        )
        if ownership.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Message not found.")

        count_result = await db_sess.execute(
            select(func.count()).select_from(SqlVersion).where(
                SqlVersion.message_id == body.message_id
            )
        )
        existing_count = count_result.scalar() or 0
        version_number = existing_count + 1

        new_version = SqlVersion(
            message_id=body.message_id,
            version_number=version_number,
            sql=body.sql,
            results=body.results,
            success=body.success,
            timestamp=datetime.utcnow(),
            is_original=version_number == 1,
        )
        db_sess.add(new_version)
        await db_sess.commit()

    logger.info(
        "SQL version saved",
        message_id=body.message_id,
        version_number=version_number,
        total_versions=version_number,
    )

    return SaveVersionResponse(
        success=True,
        version_number=version_number,
        total_versions=version_number,
    )


@router.get(
    "/query/versions/{message_id}",
    summary="Get all versions of a SQL query",
    description="Returns all saved versions for a specific message.",
)
async def get_sql_versions(
    message_id: int,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> dict[str, Any]:
    """Endpoint: Get all SQL versions for a message from database."""
    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import ChatMessage, ChatSession, SqlVersion

    # ChatMessage.id is a 32-bit PostgreSQL INTEGER. The client sometimes asks
    # for versions of an optimistic/not-yet-persisted message whose id is a
    # client-generated timestamp (e.g. Date.now() → 1_783_533_067_258), which
    # overflows int4 and makes asyncpg raise. Such an id can't exist, so short-
    # circuit to an empty collection instead of 500-ing. (2_147_483_647 = int4 max)
    if not (1 <= message_id <= 2_147_483_647):
        return {"message_id": message_id, "versions": [], "total_versions": 0}

    async with session_service._session_factory() as db_sess:
        result = await db_sess.execute(
            select(SqlVersion)
            .join(ChatMessage, ChatMessage.id == SqlVersion.message_id)
            .join(ChatSession, ChatSession.id == ChatMessage.session_id)
            .where(
                SqlVersion.message_id == message_id,
                ChatSession.user_id == current_user.id,
            )
            .order_by(SqlVersion.version_number)
        )
        versions = result.scalars().all()

    # A message with no saved edit-versions is a valid empty state, not an
    # error: return an empty collection (200) so the client renders the plain
    # SQL instead of surfacing a spurious 404 toast on every un-edited message.
    return {
        "message_id": message_id,
        "versions": [
            {
                "version": v.version_number,
                "sql": v.sql,
                "results": v.results,
                "success": v.success,
                "timestamp": v.timestamp.isoformat(),
                "is_original": v.is_original,
            }
            for v in versions
        ],
        "total_versions": len(versions),
    }
