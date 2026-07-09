"""Fine-tuning Service — Manages LLM fine-tuning with collected training data."""
import os
import tempfile
from collections.abc import Callable
from typing import Any, cast

import structlog

from nl_to_sql.services.training_data_service import TrainingDataService

logger = structlog.get_logger(__name__)

# Callback that activates a newly-deployed provider/model at runtime.
# Injected by the composition root so this service never imports the container
# (breaks the api.dependencies ↔ container ↔ fine_tuning_service import cycle).
SwitchProviderCallback = Callable[[str, str], Any]

TOGETHER_BASE_URL = "https://api.together.xyz/v1"


class FineTuningService:
    """Manages LLM fine-tuning with collected training data.

    Supported providers:
      - together: Together AI (Llama, Mistral — recommended, OpenAI-compatible API)
      - openai:   OpenAI (deprecated for many accounts as of 2025)

    SOLID:
      S — Only handles fine-tuning orchestration
      D — Depends on TrainingDataService and LLM provider APIs
    """

    def __init__(
        self,
        provider: str = "together",
        openai_api_key: str = "",
        together_api_key: str = "",
        training_data_service: TrainingDataService | None = None,
        switch_provider: SwitchProviderCallback | None = None,
    ) -> None:
        self._provider = provider
        self._openai_api_key = openai_api_key
        self._together_api_key = together_api_key
        # Resolve the effective key for the configured provider
        self._api_key = together_api_key if provider == "together" else openai_api_key
        self._training_data_service = training_data_service
        self._switch_provider = switch_provider
        self._logger = logger.bind(component="FineTuningService", provider=provider)

    def set_switch_provider(self, switch_provider: SwitchProviderCallback) -> None:
        """Inject the runtime provider-switch callback (called by the container)."""
        self._switch_provider = switch_provider

    def _openai_client(self, *, base_url: str | None = None) -> Any:
        """Return an AsyncOpenAI client, optionally pointed at an alternate base URL."""
        from openai import AsyncOpenAI
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return AsyncOpenAI(**kwargs)

    async def prepare_training_file(
        self,
        format: str = "jsonl",
        limit: int = 1000,
    ) -> str:
        """Export training data in fine-tuning format and save to temp file.

        Returns:
            Path to the temporary training file.
        """
        if self._training_data_service is None:
            raise ValueError("TrainingDataService not configured")

        self._logger.info("Preparing training file", format=format, limit=limit)

        training_data = await self._training_data_service.export_training_data(
            format="jsonl",
            limit=limit,
            include_used=False,
        )

        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.jsonl',
            delete=False,
            encoding='utf-8',
        )
        try:
            temp_file.write(training_data)
            temp_file.close()
            self._logger.info(
                "Training file prepared",
                file_path=temp_file.name,
                size_bytes=os.path.getsize(temp_file.name),
            )
            return temp_file.name
        except Exception as exc:
            self._logger.error("Failed to write training file", error=str(exc))
            raise

    async def start_fine_tuning(
        self,
        model: str,
        training_file_path: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> str:
        """Start fine-tuning job and return job ID."""
        self._logger.info("Starting fine-tuning job", model=model, training_file=training_file_path)

        if self._provider == "together":
            return await self._fine_tune_together(model, training_file_path, hyperparameters)
        elif self._provider == "openai":
            return await self._fine_tune_openai(model, training_file_path, hyperparameters)
        else:
            raise ValueError(f"Unsupported fine-tuning provider: {self._provider}")

    async def _fine_tune_together(
        self,
        model: str,
        file_path: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> str:
        """Together AI fine-tuning (OpenAI-compatible endpoint)."""
        try:
            client = self._openai_client(base_url=TOGETHER_BASE_URL)

            self._logger.info("Uploading training file to Together AI")
            with open(file_path, "rb") as f:
                file_obj = await client.files.create(file=f, purpose="fine-tune")

            self._logger.info("File uploaded", file_id=file_obj.id)

            job_params: dict[str, Any] = {
                "training_file": file_obj.id,
                "model": model,
            }
            if hyperparameters:
                job_params["hyperparameters"] = hyperparameters

            job = await client.fine_tuning.jobs.create(**job_params)

            self._logger.info("Fine-tuning job created", job_id=job.id, status=job.status)
            return cast(str, job.id)

        except Exception as exc:
            self._logger.error("Together AI fine-tuning failed", error=str(exc))
            raise

    async def _fine_tune_openai(
        self,
        model: str,
        file_path: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> str:
        """OpenAI fine-tuning API integration."""
        try:
            client = self._openai_client()

            self._logger.info("Uploading training file to OpenAI")
            with open(file_path, "rb") as f:
                file_obj = await client.files.create(file=f, purpose="fine-tune")

            self._logger.info("File uploaded", file_id=file_obj.id)

            job_params: dict[str, Any] = {
                "training_file": file_obj.id,
                "model": model,
            }
            if hyperparameters:
                job_params["hyperparameters"] = hyperparameters

            job = await client.fine_tuning.jobs.create(**job_params)

            self._logger.info("Fine-tuning job created", job_id=job.id, status=job.status)
            return cast(str, job.id)

        except Exception as exc:
            # Catch OpenAI's deprecation error and surface a helpful message
            err_str = str(exc)
            if "training_not_available" in err_str or "winding down" in err_str.lower():
                raise ValueError(
                    "OpenAI has discontinued self-serve fine-tuning for your organization. "
                    "Set FINE_TUNING_PROVIDER=together in your .env and add a TOGETHER_API_KEY "
                    "to use Together AI fine-tuning (supports Llama 3.1, Mistral, and more)."
                ) from exc
            self._logger.error("OpenAI fine-tuning failed", error=err_str)
            raise

    async def check_job_status(self, job_id: str) -> dict[str, Any]:
        """Check fine-tuning job status."""
        try:
            if self._provider == "together":
                client = self._openai_client(base_url=TOGETHER_BASE_URL)
            elif self._provider == "openai":
                client = self._openai_client()
            else:
                raise ValueError(f"Unsupported provider: {self._provider}")

            job = await client.fine_tuning.jobs.retrieve(job_id)
            return {
                "job_id": job.id,
                "status": job.status,
                "model": job.fine_tuned_model,
                "trained_tokens": job.trained_tokens,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "error": job.error.message if job.error else None,
            }

        except Exception as exc:
            self._logger.error("Failed to check job status", error=str(exc))
            raise

    async def list_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent fine-tuning jobs."""
        try:
            if self._provider == "together":
                client = self._openai_client(base_url=TOGETHER_BASE_URL)
            elif self._provider == "openai":
                client = self._openai_client()
            else:
                raise ValueError(f"Unsupported provider: {self._provider}")

            jobs = await client.fine_tuning.jobs.list(limit=limit)
            return [
                {
                    "job_id": job.id,
                    "status": job.status,
                    "model": job.fine_tuned_model,
                    "created_at": job.created_at,
                }
                for job in jobs.data
            ]

        except Exception as exc:
            self._logger.error("Failed to list jobs", error=str(exc))
            return []

    async def deploy_fine_tuned_model(self, model_id: str) -> bool:
        """Switch the running application to use a fine-tuned model.

        Validates the model exists, then hot-swaps the LLM provider singleton
        so all subsequent requests use the fine-tuned model without a restart.
        """
        try:
            self._logger.info("Deploying fine-tuned model", model_id=model_id)

            if self._provider in ("together", "openai"):
                base_url = TOGETHER_BASE_URL if self._provider == "together" else None
                client = self._openai_client(base_url=base_url)
                jobs = await client.fine_tuning.jobs.list(limit=100)
                valid_models = {
                    j.fine_tuned_model
                    for j in jobs.data
                    if j.fine_tuned_model and j.status == "succeeded"
                }
                if model_id not in valid_models:
                    raise ValueError(
                        f"Model '{model_id}' not found among succeeded fine-tuning jobs. "
                        f"Available: {sorted(valid_models)}"
                    )
            else:
                raise NotImplementedError(
                    f"Fine-tuning deployment is not supported for provider '{self._provider}'."
                )

            if self._switch_provider is None:
                raise RuntimeError(
                    "switch_provider callback not configured — cannot activate the "
                    "deployed model. The composition root must inject it."
                )
            self._switch_provider(self._provider, model_id)

            self._logger.info("Fine-tuned model deployed and activated", model_id=model_id, provider=self._provider)
            return True

        except Exception as exc:
            self._logger.error("Failed to deploy fine-tuned model", error=str(exc))
            raise
