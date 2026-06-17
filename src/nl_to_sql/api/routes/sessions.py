"""Session routes — manage chat sessions (scoped to authenticated user)."""
import asyncio
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.core.models.query import QueryResponse
from nl_to_sql.services.chat_session_service import ChatSessionService, make_json_serializable

logger = structlog.get_logger(__name__)


class AddMessageRequest(BaseModel):
    """Request body to add a custom/direct SQL message to a session."""

    question: str
    sql: str
    dialect: str
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    retrieved_tables: list[str] = Field(default_factory=list)
    used_tables: list[str] = Field(default_factory=list)
    execution_result: list[dict] | None = None
    execution_error: str | None = None
    tokens_used: int = 0
    cached: bool = False
    message: str | None = None
    intent_type: str | None = None
    suggested_chart: dict | None = None
    follow_up_questions: list[str] = Field(default_factory=list)


router = APIRouter(prefix="/api/v1", tags=["Sessions"])


@router.post(
    "/sessions",
    summary="Create a new chat session",
    description="Creates a new chat session for the authenticated user and returns its ID.",
)
async def create_session(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
):
    """Create a new chat session scoped to the current user."""
    session = await session_service.create_session(user_id=current_user.id)
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@router.get(
    "/sessions",
    summary="List all chat sessions",
    description="Returns a list of the authenticated user's chat sessions ordered by most recently updated.",
)
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
    response: Response = None,
):
    """List the current user's chat sessions."""
    if response:
        response.headers["Cache-Control"] = "private, max-age=5"
    sessions, total = await asyncio.gather(
        session_service.list_sessions(limit=limit, offset=offset, user_id=current_user.id),
        session_service.count_sessions(user_id=current_user.id),
    )
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "message_count": s.message_count,
            }
            for s in sessions
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/sessions/{session_id}",
    summary="Get a chat session with all messages",
    description="Returns a chat session and all its messages (must belong to the current user).",
)
async def get_session(
    session_id: str,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
):
    """Get a specific chat session with messages."""
    try:
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Security: ensure session belongs to current user (or is legacy/unowned)
        if session.user_id and session.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": [
                {
                    "id": msg.id,
                    "question": msg.question,
                    "timestamp": msg.timestamp.isoformat(),
                    "response": {
                        "question": msg.question,
                        "sql": msg.sql,
                        "dialect": msg.dialect,
                        "is_valid": msg.is_valid,
                        "validation_errors": msg.validation_errors or [],
                        "retrieved_tables": msg.retrieved_tables or [],
                        "used_tables": msg.used_tables or [],
                        "execution_result": make_json_serializable(msg.execution_result),
                        "execution_error": msg.execution_error,
                        "tokens_used": msg.tokens_used,
                        "cached": msg.cached,
                        "message": msg.message,
                        "intent_type": msg.intent_type,
                        "suggested_chart": msg.suggested_chart,
                        "follow_up_questions": msg.follow_up_questions or [],
                    },
                }
                for msg in session.messages
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get session",
            session_id=session_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session: {str(exc)}",
        )


@router.post(
    "/sessions/{session_id}/messages",
    summary="Add a message/direct query to a session",
)
async def add_session_message(
    session_id: str,
    body: AddMessageRequest,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
):
    """Save a custom/direct SQL query message into the session history."""
    try:
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.user_id and session.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Map request body to a QueryResponse
        response_data = QueryResponse(
            question=body.question,
            sql=body.sql,
            dialect=body.dialect,
            is_valid=body.is_valid,
            validation_errors=body.validation_errors,
            retrieved_tables=body.retrieved_tables,
            used_tables=body.used_tables,
            execution_result=body.execution_result,
            execution_error=body.execution_error,
            tokens_used=body.tokens_used,
            cached=body.cached,
            message=body.message,
            intent_type=body.intent_type,
            suggested_chart=body.suggested_chart,
            follow_up_questions=body.follow_up_questions,
        )

        message = await session_service.add_message(
            session_id=session_id,
            question=body.question,
            response=response_data,
        )

        return {
            "id": message.id,
            "question": message.question,
            "timestamp": message.timestamp.isoformat(),
            "response": {
                "question": message.question,
                "sql": message.sql,
                "dialect": message.dialect,
                "is_valid": message.is_valid,
                "validation_errors": message.validation_errors or [],
                "retrieved_tables": message.retrieved_tables or [],
                "used_tables": message.used_tables or [],
                "execution_result": make_json_serializable(message.execution_result),
                "execution_error": message.execution_error,
                "tokens_used": message.tokens_used,
                "cached": message.cached,
                "message": message.message,
                "intent_type": message.intent_type,
                "suggested_chart": message.suggested_chart,
                "follow_up_questions": message.follow_up_questions or [],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to add message to session",
            session_id=session_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add message to session: {str(exc)}",
        )


@router.delete(
    "/sessions/{session_id}",
    summary="Delete a chat session",
    description="Deletes a chat session and all its messages (must belong to the current user).",
)
async def delete_session(
    session_id: str,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
):
    """Delete a chat session."""
    # Verify ownership before deletion
    session = await session_service.get_session(session_id)
    if session and session.user_id and session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    await session_service.delete_session(session_id)
    return {"message": "Session deleted successfully"}


@router.delete(
    "/sessions",
    summary="Delete all chat sessions",
    description="Deletes all of the authenticated user's chat sessions and their messages.",
)
async def delete_all_sessions(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
):
    """Delete all of the current user's chat sessions."""
    await session_service.delete_all_sessions(user_id=current_user.id)
    return {"message": "All sessions deleted successfully"}
