"""Fine-tuning routes — POST/GET /api/v1/fine-tuning/*."""
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from nl_to_sql.api.dependencies import get_container, get_current_user
from nl_to_sql.config.container import ApplicationContainer
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.fine_tuning_service import FineTuningService

router = APIRouter(prefix="/api/v1/fine-tuning", tags=["Fine-Tuning"])


class PrepareFileResponse(BaseModel):
    file_path: str


class StartJobResponse(BaseModel):
    job_id: str
    status: str


class DeployResponse(BaseModel):
    deployed: bool
    active_model: str


async def get_fine_tuning_service(
    container: ApplicationContainer = Depends(get_container),
    current_user: UserPublic = Depends(get_current_user),
) -> FineTuningService:
    """Get fine-tuning service, preferring the user's stored Together AI key."""
    settings = container.config()

    # Resolve the effective Together AI key: user key takes priority over server key
    together_key = settings.together_api_key
    try:
        api_key_svc = container.api_key_service()
        user_key = await api_key_svc.get_key(current_user.id, "together")
        if user_key:
            together_key = user_key
    except Exception:
        pass

    provider = settings.fine_tuning_provider

    # Resolve the effective key for the configured provider
    if provider == "together":
        effective_key = together_key
        missing_msg = (
            "Together AI API key required for fine-tuning. "
            "Add your key in Profile → API Keys, or set TOGETHER_API_KEY in your environment."
        )
    else:
        effective_key = settings.openai_api_key
        missing_msg = "OpenAI API key required for fine-tuning. Set OPENAI_API_KEY in your environment."

    if not effective_key:
        raise HTTPException(status_code=400, detail=missing_msg)

    # If the user's Together AI key differs from the server env key, build a
    # per-request service instance so the user's key is used, not the singleton's.
    base_service = container.fine_tuning_service()
    if provider == "together" and together_key != settings.together_api_key:
        return FineTuningService(
            provider=provider,
            openai_api_key=settings.openai_api_key,
            together_api_key=together_key,
            training_data_service=container.training_data_service(),
        )

    return base_service


@router.post("/prepare", response_model=PrepareFileResponse)
async def prepare_training_file(
    format: str = Query(default="jsonl", pattern="^(json|jsonl)$"),
    limit: int = Query(default=1000, ge=1, le=10000),
    fine_tuning_service: FineTuningService = Depends(get_fine_tuning_service),
) -> dict[str, str]:
    """Prepare training data file for fine-tuning.

    Args:
        format: Export format (jsonl recommended).
        limit: Maximum number of records.

    Returns:
        Path to the prepared training file.
    """
    file_path = await fine_tuning_service.prepare_training_file(
        format=format,
        limit=limit,
    )
    return {"file_path": file_path}


@router.post("/start", response_model=StartJobResponse)
async def start_fine_tuning(
    model: str,
    training_file_path: str,
    hyperparameters: dict[str, Any] | None = None,
    fine_tuning_service: FineTuningService = Depends(get_fine_tuning_service),
) -> dict[str, str]:
    """Start a fine-tuning job.

    Args:
        model: Base model to fine-tune.
        training_file_path: Path to training data file.
        hyperparameters: Optional training hyperparameters.

    Returns:
        Fine-tuning job ID.
    """
    try:
        job_id = await fine_tuning_service.start_fine_tuning(
            model=model,
            training_file_path=training_file_path,
            hyperparameters=hyperparameters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "started"}


@router.get("/status/{job_id}")
async def check_job_status(
    job_id: str,
    fine_tuning_service: FineTuningService = Depends(get_fine_tuning_service),
) -> dict[str, Any]:
    """Check fine-tuning job status.

    Args:
        job_id: The fine-tuning job ID.

    Returns:
        Job status information.
    """
    return cast(dict[str, Any], await fine_tuning_service.check_job_status(job_id))


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(default=10, ge=1, le=100),
    container: ApplicationContainer = Depends(get_container),
    current_user: UserPublic = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List recent fine-tuning jobs.

    Args:
        limit: Maximum number of jobs to return.

    Returns:
        List of job status information.
    """
    try:
        fine_tuning_service = await get_fine_tuning_service(container, current_user)
    except HTTPException as exc:
        if exc.status_code == 400:
            return []
        raise
    return cast(list[dict[str, Any]], await fine_tuning_service.list_jobs(limit=limit))


@router.post("/deploy", response_model=DeployResponse)
async def deploy_fine_tuned_model(
    model_id: str,
    fine_tuning_service: FineTuningService = Depends(get_fine_tuning_service),
) -> dict[str, Any]:
    """Deploy a fine-tuned model and hot-swap the running LLM provider.

    Validates the model exists, then switches the application to use it
    immediately without a server restart.

    Args:
        model_id: The fine-tuned model ID returned by Together AI (e.g. "ft-abc123-llama-3-1").

    Returns:
        Deployment status and active model.
    """
    success = await fine_tuning_service.deploy_fine_tuned_model(model_id)
    return {"deployed": success, "active_model": model_id}
