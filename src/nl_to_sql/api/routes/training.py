"""Training data routes — GET/POST /api/v1/training/*."""
import io
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from nl_to_sql.api.dependencies import get_container
from nl_to_sql.config.container import ApplicationContainer
from nl_to_sql.services.training_data_service import TrainingDataService

router = APIRouter(prefix="/api/v1/training", tags=["Training Data"])


async def get_training_data_service(
    container: ApplicationContainer = Depends(get_container),
) -> TrainingDataService:
    """Get training data service instance."""
    return container.training_data_service()


@router.get("/stats")
async def get_training_stats(
    training_service: TrainingDataService = Depends(get_training_data_service),
) -> dict[str, Any]:
    """Get training data statistics."""
    return await training_service.get_training_stats()


@router.get("/export")
async def export_training_data(
    format: str = Query(default="json", pattern="^(json|jsonl)$"),
    limit: int = Query(default=1000, ge=1, le=10000),
    include_used: bool = False,
    training_service: TrainingDataService = Depends(get_training_data_service),
) -> str:
    """Export training data for fine-tuning.

    Args:
        format: Export format (json or jsonl for OpenAI fine-tuning).
        limit: Maximum number of records to export.
        include_used: Whether to include already-used records.

    Returns:
        JSON or JSONL string of training data.
    """
    return await training_service.export_training_data(
        format=format,
        limit=limit,
        include_used=include_used,
    )


@router.get("/download")
async def download_training_data(
    format: str = Query(default="jsonl", pattern="^(json|jsonl)$"),
    limit: int = Query(default=1000, ge=1, le=10000),
    include_used: bool = False,
    training_service: TrainingDataService = Depends(get_training_data_service),
) -> StreamingResponse:
    """Download training data as a file attachment.

    Returns a JSONL or JSON file that can be used for fine-tuning on any
    platform (Hugging Face, OpenAI, Together AI, Google Colab, etc.).
    """
    data = await training_service.export_training_data(
        format=format,
        limit=limit,
        include_used=include_used,
    )
    filename = f"training_data.{format}"
    content_type = "application/x-ndjson" if format == "jsonl" else "application/json"
    return StreamingResponse(
        io.BytesIO(data.encode("utf-8")),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/mark-used")
async def mark_training_data_used(
    ids: list[int],
    training_service: TrainingDataService = Depends(get_training_data_service),
) -> dict[str, int]:
    """Mark training records as used for fine-tuning.

    Args:
        ids: List of training data record IDs to mark.

    Returns:
        Number of records marked.
    """
    count = await training_service.mark_as_used(ids)
    return {"marked_count": count}
