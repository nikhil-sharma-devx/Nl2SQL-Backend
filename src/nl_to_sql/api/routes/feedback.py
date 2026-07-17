"""Feedback route — POST /api/v1/feedback."""
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.chat_session_service import ChatSessionService

router = APIRouter(prefix="/api/v1", tags=["Feedback"])
_settings = get_settings()
_rate = f"{_settings.rate_limit_requests}/minute"


class FeedbackRequest(BaseModel):
    """Feedback submission request."""

    question: str = Field(..., description="Original user question")
    generated_sql: str = Field(..., description="SQL that was generated")
    feedback_type: str = Field(..., description="'positive' or 'negative'")
    error_type: str | None = Field(None, description="Type of error (for negative feedback)")
    user_correction: str | None = Field(None, description="User's corrected SQL")
    user_notes: str | None = Field(None, description="Additional user notes")


class FeedbackResponse(BaseModel):
    """Feedback submission response."""

    success: bool = Field(..., description="Whether feedback was recorded")
    message: str = Field(..., description="Status message")


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback on query results",
    description=(
        "Accepts user feedback (thumbs up/down) with optional error details. "
        "This feedback is used to improve future SQL generation by learning "
        "from common mistakes."
    ),
)
@limiter.limit(_rate)
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> FeedbackResponse:
    """Submit feedback on query results."""
    import structlog

    from nl_to_sql.api.dependencies import get_container

    logger = structlog.get_logger(__name__)

    # Get container to access feedback_learner
    container = get_container()

    try:
        # Record feedback in learner
        feedback_learner = container.feedback_learner()
        await feedback_learner.record_feedback(
            question=body.question,
            generated_sql=body.generated_sql,
            feedback_type=body.feedback_type,
            error_type=body.error_type,
            user_correction=body.user_correction,
            user_notes=body.user_notes,
        )

        logger.info(
            "Feedback recorded successfully",
            feedback_type=body.feedback_type,
            question=body.question[:50],
            error_type=body.error_type,
        )

        # P2 — index thumbs-up pairs for semantic few-shot retrieval.
        if body.feedback_type == "positive" and get_settings().rag_few_shot_retrieval_enabled:
            try:
                await container.example_store().index_example(
                    question=body.question,
                    sql=body.generated_sql,
                    user_id=current_user.id,
                )
            except Exception as ex_exc:
                logger.warning("Failed to index positive feedback example", error=str(ex_exc))

        if body.feedback_type == "positive":
            return FeedbackResponse(
                success=True,
                message="Thank you for the positive feedback!",
            )
        else:
            return FeedbackResponse(
                success=True,
                message="Thank you for reporting the issue. We'll learn from this!",
            )
    except Exception as exc:
        logger.error("Failed to record feedback", error=str(exc))
        return FeedbackResponse(
            success=False,
            message=f"Failed to record feedback: {exc!s}",
        )


@router.get(
    "/feedback",
    summary="List feedback records",
    description="Retrieve a list of user feedback submissions.",
)
async def list_feedback(
    limit: int = Query(default=10, ge=1, le=100),
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> list[dict[str, Any]]:
    """Retrieve a list of user feedback submissions."""
    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import FeedbackRecord

    async with session_service._session_factory() as session:
        result = await session.execute(
            select(FeedbackRecord).order_by(FeedbackRecord.timestamp.desc()).limit(limit)
        )
        records = result.scalars().all()
        return [
            {
                "id": r.id,
                "query_id": r.query_id,
                "session_id": r.session_id,
                "feedback_type": r.feedback_type,
                "feedback_data": r.feedback_data,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in records
        ]
