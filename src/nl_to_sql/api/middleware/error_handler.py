"""Global exception handler middleware — maps domain errors to HTTP responses."""
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from nl_to_sql.core.exceptions import (
    DatabaseExecutionError,
    EmptySchemaError,
    LLMProviderError,
    NLToSQLBaseError,
    RateLimitError,
    SchemaRetrievalError,
    SQLGenerationError,
    SQLValidationError,
)

_ERROR_STATUS_MAP: dict[type, int] = {
    EmptySchemaError: 503,
    SchemaRetrievalError: 503,
    RateLimitError: 429,
    LLMProviderError: 502,
    SQLGenerationError: 422,
    SQLValidationError: 422,
    DatabaseExecutionError: 422,
    NLToSQLBaseError: 500,
}

_ERROR_CODE_MAP: dict[type, str] = {
    EmptySchemaError: "SCHEMA_NOT_LOADED",
    SchemaRetrievalError: "SCHEMA_RETRIEVAL_FAILED",
    RateLimitError: "RATE_LIMIT_EXCEEDED",
    LLMProviderError: "LLM_PROVIDER_ERROR",
    SQLGenerationError: "SQL_GENERATION_FAILED",
    SQLValidationError: "SQL_VALIDATION_FAILED",
    DatabaseExecutionError: "DATABASE_EXECUTION_FAILED",
    NLToSQLBaseError: "INTERNAL_ERROR",
}


async def domain_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert domain exceptions to structured JSON error responses.

    All domain errors include:
      - code: machine-readable error code
      - message: human-readable description
      - details: optional extra info
      - retry_after: seconds to wait (for rate limit errors)
    """
    status_code = 500
    error_code = "INTERNAL_ERROR"
    for exc_type, code in _ERROR_STATUS_MAP.items():
        if isinstance(exc, exc_type):
            status_code = code
            error_code = _ERROR_CODE_MAP.get(exc_type, "INTERNAL_ERROR")
            break

    payload: dict[str, Any] = {
        "code": error_code,
        "message": str(exc),
    }

    if isinstance(exc, NLToSQLBaseError) and exc.detail:
        payload["details"] = {"detail": exc.detail}

    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        payload["retry_after"] = exc.retry_after

    return JSONResponse(status_code=status_code, content=payload)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected server errors."""
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again.",
        },
    )
