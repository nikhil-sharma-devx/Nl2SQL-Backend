"""Observability — OpenTelemetry Distributed Tracing setup & GenAI semantic conventions."""
import contextlib
import functools
import logging
from collections.abc import Callable, Generator
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

# Try importing opentelemetry, provide fallback/mock if imports fail.
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


def setup_tracing(
    service_name: str = "nl-to-sql-rag",
    otlp_endpoint: str | None = None,
    enable_console_export: bool = False,
) -> None:
    """Initialize standard OpenTelemetry trace provider and OTLP / console exporters."""
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("opentelemetry-api and sdk are not installed. Tracing is disabled.")
        return

    try:
        resource = Resource.create(attributes={"service.name": service_name})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # OTLP Exporter target (e.g. Jaeger, Phoenix collector, Otel Collector)
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info(f"OpenTelemetry OTLP Exporter registered on endpoint: {otlp_endpoint}")
            except Exception as e:
                logger.error(f"Failed to initialize OTLP gRPC Exporter: {e}")
                if enable_console_export:
                    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        elif enable_console_export:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("OpenTelemetry Console Span Exporter registered.")
        else:
            logger.info("Tracing initialized with no-op provider (no exporters configured).")
    except Exception as e:
        logger.error(f"Failed to configure OpenTelemetry Tracing: {e}")


def instrument_app(app: Any) -> None:
    """Instrument FastAPI application endpoints using opentelemetry instrumentation helper."""
    if not OPENTELEMETRY_AVAILABLE:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI application instrumented successfully with OpenTelemetry.")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-fastapi is not installed. FastAPI routes will not be auto-instrumented.")
    except Exception as e:
        logger.warning(f"Could not instrument FastAPI application with OpenTelemetry: {e}")


def get_tracer() -> Any:
    """Return the active OpenTelemetry tracer instance, or None if disabled."""
    if OPENTELEMETRY_AVAILABLE:
        return trace.get_tracer("nl-to-sql-rag")
    return None


_P = ParamSpec("_P")
_R = TypeVar("_R")


@contextlib.contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context manager to run a block of code inside an OpenTelemetry trace span.

    Attributes map to standard GenAI/LLM semantic conventions where possible.
    """
    tracer = get_tracer()
    if tracer:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        span.set_attribute(k, v)
            yield span
    else:
        yield None


def trace_function(name: str | None = None, attributes: dict[str, Any] | None = None) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to instrument any synchronous or asynchronous function with an Otel span."""
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(span_name, attributes) as span:
                try:
                    res = await func(*args, **kwargs)  # type: ignore[misc]
                    # Capture LLM evaluation tokens if returned in model properties
                    if span and res and hasattr(res, 'tokens_used') and res.tokens_used:
                        span.set_attribute("gen_ai.usage.total_tokens", res.tokens_used)
                    return res
                except Exception as exc:
                    if span:
                        span.record_exception(exc)
                        span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, str(exc)))
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(span_name, attributes) as span:
                try:
                    res = func(*args, **kwargs)
                    if span and res and hasattr(res, 'tokens_used') and res.tokens_used:
                        span.set_attribute("gen_ai.usage.total_tokens", res.tokens_used)
                    return res
                except Exception as exc:
                    if span:
                        span.record_exception(exc)
                        span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, str(exc)))
                    raise

        import asyncio
        result: Callable[_P, _R]
        if asyncio.iscoroutinefunction(func):
            result = async_wrapper  # type: ignore[assignment]
        else:
            result = sync_wrapper
        return result
    return decorator


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the currently active OpenTelemetry span, if tracing is active."""
    if not OPENTELEMETRY_AVAILABLE:
        return
    try:
        span = trace.get_current_span()
        if span.is_recording() and value is not None:
            span.set_attribute(key, value)
    except Exception:
        pass

