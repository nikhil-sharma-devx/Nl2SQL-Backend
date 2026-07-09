"""Global exception handler middleware — maps domain errors to HTTP responses.

Every error response — domain, HTTP, request-validation, or unhandled — is
returned in one canonical envelope so the frontend can map errors uniformly
(item 13):

    {
      "error":      <machine-readable code, mirror of "code">,
      "code":       <machine-readable code>,
      "message":    <human-readable description>,
      "request_id": <correlation id, from X-Request-ID>,
      "details":    <optional extra info>,
      "retry_after": <optional seconds, rate-limit only>
    }
"""
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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


def _request_id(request: Request) -> str | None:
    """Correlation ID stamped by RequestLoggingMiddleware (None if absent)."""
    return getattr(request.state, "request_id", None)


async def domain_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert domain exceptions to structured JSON error responses.

    All domain errors include:
      - code: machine-readable error code
      - message: human-readable description
      - request_id: correlation ID for log lookup
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

    payload = _envelope(request, error_code, str(exc))

    if isinstance(exc, NLToSQLBaseError) and exc.detail:
        payload["details"] = {"detail": exc.detail}

    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        payload["retry_after"] = exc.retry_after

    return JSONResponse(status_code=status_code, content=payload)


def _envelope(
    request: Request,
    code: str,
    message: str,
    details: Any = None,
) -> dict[str, Any]:
    """Build the canonical error envelope shared by every handler.

    `error` mirrors `code` so clients can key off either name (item 13).
    """
    payload: dict[str, Any] = {
        "error": code,
        "code": code,
        "message": message,
        "request_id": _request_id(request),
    }
    if details is not None:
        payload["details"] = details
    return payload


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected server errors."""
    return JSONResponse(
        status_code=500,
        content=_envelope(
            request,
            "INTERNAL_SERVER_ERROR",
            "An unexpected error occurred. Please try again.",
        ),
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Route FastAPI/Starlette HTTPExceptions through the canonical envelope.

    Without this, `raise HTTPException(404, "...")` returns Starlette's default
    `{"detail": ...}` shape with no `code`/`request_id`, breaking the FE's
    uniform error mapping.
    """
    code = f"HTTP_{exc.status_code}"
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    payload = _envelope(request, code, message)
    headers = getattr(exc, "headers", None)
    return JSONResponse(status_code=exc.status_code, content=payload, headers=headers)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Route 422 request-validation errors through the canonical envelope."""
    # exc.errors() items contain non-JSON-serializable objects (e.g. ValueError
    # in "ctx"); jsonable_encoder normalizes them.
    from fastapi.encoders import jsonable_encoder

    payload = _envelope(
        request,
        "REQUEST_VALIDATION_FAILED",
        "Request validation failed.",
        details={"errors": jsonable_encoder(exc.errors())},
    )
    return JSONResponse(status_code=422, content=payload)
