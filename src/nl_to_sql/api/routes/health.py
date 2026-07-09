"""Health & readiness endpoints."""
import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from nl_to_sql.api.dependencies import get_container, get_db_client, get_vector_store
from nl_to_sql.config.container import ApplicationContainer
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient

router = APIRouter(prefix="/api/v1", tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]


class DeepCheckResult(BaseModel):
    ok: bool
    latency_ms: int
    error: str | None = None


class DeepHealthResponse(BaseModel):
    status: str
    checks: dict[str, DeepCheckResult]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 if the application process is running.",
)
async def liveness() -> HealthResponse:
    """Kubernetes liveness probe — always returns 200 if the process is up."""
    from nl_to_sql.config.settings import get_settings
    settings = get_settings()
    return HealthResponse(status="ok", environment=settings.app_env)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Returns 200 only when all dependent services (DB, vector store) are reachable.",
)
async def readiness(
    db: AsyncDatabaseClient = Depends(get_db_client),
    vector_store: IVectorStore = Depends(get_vector_store),
) -> ReadinessResponse:
    """Kubernetes readiness probe."""
    db_ok = await db.health_check()
    vs_ok = await vector_store.health_check()
    all_ok = db_ok and vs_ok

    return ReadinessResponse(
        status="ready" if all_ok else "degraded",
        checks={"database": db_ok, "vector_store": vs_ok},
    )


async def _timed_check(coro: Any) -> DeepCheckResult:
    """Run a health-check coroutine, capturing latency and any error."""
    start = time.perf_counter()
    try:
        ok = bool(await coro)
        return DeepCheckResult(ok=ok, latency_ms=int((time.perf_counter() - start) * 1000))
    except Exception as exc:
        return DeepCheckResult(
            ok=False,
            latency_ms=int((time.perf_counter() - start) * 1000),
            error=str(exc)[:200],
        )


@router.get(
    "/health/deep",
    response_model=DeepHealthResponse,
    summary="Deep dependency health check",
    description=(
        "Actively probes the database, vector store, and LLM provider, "
        "reporting per-dependency reachability and latency. Heavier than "
        "/ready — intended for dashboards and on-demand diagnostics, not "
        "high-frequency liveness polling."
    ),
)
async def deep_health(
    container: ApplicationContainer = Depends(get_container),
) -> DeepHealthResponse:
    """Probe every external dependency concurrently and report latency."""
    db = container.db_client()
    vector_store = container.vector_store()
    llm = container.llm_provider()

    db_res, vs_res, llm_res = await asyncio.gather(
        _timed_check(db.health_check()),
        _timed_check(vector_store.health_check()),
        _timed_check(llm.health_check()),
    )

    checks = {"database": db_res, "vector_store": vs_res, "llm_provider": llm_res}
    status = "ok" if all(c.ok for c in checks.values()) else "degraded"
    return DeepHealthResponse(status=status, checks=checks)
