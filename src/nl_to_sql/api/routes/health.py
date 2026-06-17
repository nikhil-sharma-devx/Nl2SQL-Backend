"""Health & readiness endpoints."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from nl_to_sql.api.dependencies import get_db_client, get_vector_store
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.core.interfaces.i_vector_store import IVectorStore

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]


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
