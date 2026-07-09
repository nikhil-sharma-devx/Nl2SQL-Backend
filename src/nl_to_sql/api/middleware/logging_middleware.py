"""Structured request logging middleware."""
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every incoming request with timing and correlation ID.

    Adds an `X-Request-ID` header to every response for tracing.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        # Expose the ID to exception handlers and bind it to the structlog
        # context so every log line emitted while serving this request carries
        # the same request_id (merge_contextvars is in the processor chain).
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)

        log = logger.bind(
            method=request.method,
            path=request.url.path,
        )
        log.info("Request started")

        try:
            response: Response = await call_next(request)  # type: ignore[operator]

            duration_ms = (time.perf_counter() - start) * 1000
            log.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            structlog.contextvars.clear_contextvars()
