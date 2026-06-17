"""Feedback route — POST /api/v1/feedback."""
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.config.settings import get_settings

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
) -> FeedbackResponse:
    """Submit feedback on query results."""
    from nl_to_sql.config.container import ApplicationContainer
    from nl_to_sql.api.dependencies import get_container
    import structlog

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
            message=f"Failed to record feedback: {str(exc)}",
        )
