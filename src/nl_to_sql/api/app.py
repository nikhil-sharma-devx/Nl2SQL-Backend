"""FastAPI application factory."""
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import logging.handlers
from pathlib import Path
import sys

import structlog
import structlog.stdlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from nl_to_sql.api.middleware.error_handler import (
    domain_exception_handler,
    unhandled_exception_handler,
)
from nl_to_sql.api.middleware.logging_middleware import RequestLoggingMiddleware
from nl_to_sql.api.middleware.rate_limiter import limiter
from nl_to_sql.api.routes import (
    account,
    analytics,
    auth,
    auth_sessions,
    config,
    data,
    feedback,
    fine_tuning,
    health,
    history,
    instructions,
    profile,
    query,
    saved_queries,
    schema,
    sessions,
    training,
    usage,
    user_settings,
)
from nl_to_sql.config.container import ApplicationContainer
from nl_to_sql.config.settings import get_settings
from nl_to_sql.core.exceptions import NLToSQLBaseError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler — runs startup and shutdown logic."""
    container: ApplicationContainer = app.container  # type: ignore[attr-defined]

    # Fetch all service singletons once (avoids lazy-init races inside gather)
    query_history = container.query_history()
    session_service = container.session_service()
    analytics_service = container.analytics_service()
    feedback_service = container.feedback_service()
    training_service = container.training_data_service()
    api_key_service = container.api_key_service()
    user_db_service = container.user_db_service()

    # Required services — errors propagate and abort startup
    await asyncio.gather(
        query_history.initialize(),
        session_service.initialize(),
    )

    # Optional services — run in parallel, log failures but don't abort
    opt_results = await asyncio.gather(
        analytics_service.initialize(),
        feedback_service.initialize(),
        training_service.initialize(),
        api_key_service.initialize(),
        user_db_service.initialize(),
        return_exceptions=True,
    )
    _log = structlog.get_logger()
    for name, result in zip(("analytics", "feedback", "training", "api_key", "user_db"), opt_results):
        if isinstance(result, Exception):
            _log.warning(f"Failed to initialize {name} service", error=str(result))

    # Auto-ingest schema from live database on startup
    try:
        settings = get_settings()
        if settings.auto_ingest_schema_on_startup:
            async def _auto_ingest_workflow():
                try:
                    structlog.get_logger().info("Starting automatic schema ingestion from live database")
                    db_client = container.db_client()
                    ingestion_service = container.schema_ingestion()
                    vector_store = container.vector_store()

                    # Reflect current schema from database
                    schema_dict = await db_client.reflect_schema(schema_name="public")
                    current_table_count = len(schema_dict['tables'])
                    current_tables = [t['name'] for t in schema_dict['tables']]

                    # Compute hash of current schema using unified service logic
                    from nl_to_sql.services.schema_ingestion import SchemaIngestionService
                    schema_metadata = SchemaIngestionService.build_schema_from_dict(schema_dict)
                    current_schema_hash = SchemaIngestionService.compute_schema_hash(schema_metadata)

                    # Get stored hash from vector store
                    stored_hash = vector_store.get_schema_hash()

                    # Check if schema has changed
                    should_ingest = False
                    if stored_hash is None:
                        structlog.get_logger().info(
                            "No stored schema hash found, performing initial ingestion",
                            table_count=current_table_count,
                            tables=current_tables
                        )
                        should_ingest = True
                    elif stored_hash != current_schema_hash:
                        structlog.get_logger().info(
                            "Schema change detected on startup",
                            stored_hash=stored_hash[:16],
                            current_hash=current_schema_hash[:16],
                            table_count=current_table_count,
                            tables=current_tables
                        )
                        should_ingest = True
                    else:
                        chunk_count = await vector_store.count()
                        actual_chunks = max(0, chunk_count - 1)
                        if actual_chunks == 0:
                            structlog.get_logger().info(
                                "Schema hash matches but vector store is empty, re-ingesting",
                                table_count=current_table_count
                            )
                            should_ingest = True
                        else:
                            structlog.get_logger().info(
                                "Schema is up-to-date, skipping ingestion",
                                stored_chunks=actual_chunks,
                                table_count=current_table_count,
                                tables=current_tables
                            )

                    if should_ingest:
                        structlog.get_logger().info("Starting background schema ingestion...")
                        chunk_count = await ingestion_service.ingest(schema_metadata, reset=True)
                        structlog.get_logger().info(
                            "Background schema ingestion complete",
                            tables_ingested=len(schema_metadata.tables),
                            chunks_ingested=chunk_count
                        )
                except Exception as e:
                    structlog.get_logger().error("Background schema ingestion failed", error=str(e))
            
            # Run the entire workflow in the background to avoid blocking API startup
            asyncio.create_task(_auto_ingest_workflow())
        else:
            structlog.get_logger().info("Auto-ingest schema on startup is disabled")
    except Exception as exc:
        structlog.get_logger().warning(
            "Failed to start auto-ingest background task",
            error=str(exc)
        )

    # Start schema monitor in background to avoid blocking API startup
    if settings.schema_monitor_enabled:
        async def _start_monitor():
            try:
                schema_monitor = container.schema_monitor()
                await schema_monitor.start()
            except Exception as exc:
                structlog.get_logger().warning("Failed to start schema monitor", error=str(exc))
        asyncio.create_task(_start_monitor())

    yield

    # Shutdown: stop schema monitor then dispose all DB services in parallel
    if settings.schema_monitor_enabled:
        try:
            await container.schema_monitor().stop()
        except Exception:
            pass

    await asyncio.gather(
        query_history.dispose(),
        session_service.dispose(),
        analytics_service.dispose(),
        feedback_service.dispose(),
        training_service.dispose(),
        api_key_service.dispose(),
        return_exceptions=True,
    )


class WeeklyRotatingFileHandler(logging.FileHandler):
    """File handler that rotates weekly and flushes after every record.

    Filenames include the week range, e.g.:
      application_2026-06-08_to_2026-06-14.log
    Flush-after-emit ensures log entries appear in the file immediately
    (no OS-level buffering delay).
    """

    def __init__(self, log_dir: Path, encoding: str = "utf-8") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_week_start = None
        super().__init__(self._get_filepath(), encoding=encoding, delay=False)

    def _week_range(self, date_val):
        from datetime import timedelta
        start = date_val - timedelta(days=date_val.weekday())
        end = start + timedelta(days=6)
        return start, end

    def _get_filepath(self) -> Path:
        from datetime import datetime
        today = datetime.now().date()
        start, end = self._week_range(today)
        self.current_week_start = start
        filename = f"application_{start.strftime('%Y-%m-%d')}_to_{end.strftime('%Y-%m-%d')}.log"
        return self.log_dir / filename

    def emit(self, record: logging.LogRecord) -> None:
        from datetime import datetime
        today = datetime.now().date()
        start, _ = self._week_range(today)
        if self.current_week_start != start:
            self.acquire()
            try:
                if self.stream:
                    self.stream.close()
                self.baseFilename = str(self._get_filepath().resolve())
                self.stream = self._open()
            finally:
                self.release()
        super().emit(record)
        self.flush()  # write to disk immediately — no buffering delay


def configure_logging(log_level: str = "INFO", log_file: str | None = None, is_production: bool = False) -> None:
    """Configure stdlib logging and structlog.

    Terminal output:
      - Uvicorn handles its own console output unchanged (colored access logs,
        startup messages). We do NOT remove or replace uvicorn's handlers so
        the terminal shows the standard  INFO:     127.0.0.1 - "GET /" 200  lines.
      - App-level WARNING/ERROR also reaches the terminal via root logger.

    File output (when APP_LOG_FILE is set):
      - Everything: structlog JSON events + uvicorn access/error lines.
      - Flushed immediately after every record so the file is always current.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # ── Root logger ─────────────────────────────────────────────────────────
    # Only WARNING+ from app code reaches the terminal (structlog INFO/DEBUG
    # are JSON blobs that would clutter the access-log stream).
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers basicConfig may have already added
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s  %(name)s: %(message)s"))
    root.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────────────────────
    file_handler: logging.FileHandler | None = None
    if log_file:
        log_path = Path(log_file)
        file_handler = WeeklyRotatingFileHandler(log_path.parent, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        # Plain %(message)s works for both structlog JSON and uvicorn plain text
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(file_handler)

    # ── Silence noisy third-party loggers ────────────────────────────────────
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Uvicorn loggers ──────────────────────────────────────────────────────
    # Leave uvicorn's own handlers INTACT so the terminal keeps its standard
    # colored  "INFO:     ..."  format.  We only add our file handler
    # (if configured) so access + error lines also land in the log file.
    for lg_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(lg_name)
        # Do NOT clear handlers or change propagate — uvicorn owns its console output.
        if file_handler is not None and file_handler not in lg.handlers:
            lg.addHandler(file_handler)

    # ── Structlog ────────────────────────────────────────────────────────────
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance.

    Using a factory function (rather than a module-level app) allows:
      - Easy testing (create a fresh app per test).
      - Clean separation of config from app creation.
    """
    settings = get_settings()
    configure_logging(settings.app_log_level, settings.app_log_file, settings.is_production)

    # ── Observability: LangSmith, OpenTelemetry & Arize Phoenix ─────────────
    if settings.langsmith_tracing and settings.langsmith_api_key:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    trace_endpoint = settings.otel_exporter_otlp_endpoint
    if not trace_endpoint and settings.phoenix_active:
        trace_endpoint = settings.phoenix_endpoint

    if trace_endpoint or settings.otel_console_exporter:
        from nl_to_sql.infrastructure.observability.tracing import setup_tracing
        setup_tracing(
            service_name=settings.otel_service_name,
            otlp_endpoint=trace_endpoint,
            enable_console_export=settings.otel_console_exporter,
        )

    # Create container and wire dependencies
    container = ApplicationContainer()
    container.wire(modules=[
        "nl_to_sql.api.dependencies",
        "nl_to_sql.api.routes.auth",
        "nl_to_sql.api.routes.query",
        "nl_to_sql.api.routes.config",
        "nl_to_sql.api.routes.history",
        "nl_to_sql.api.routes.schema",
        "nl_to_sql.api.routes.sessions",
        "nl_to_sql.api.routes.analytics",
        "nl_to_sql.api.routes.feedback",
        "nl_to_sql.api.routes.training",
        "nl_to_sql.api.routes.fine_tuning",
        "nl_to_sql.api.routes.profile",
        "nl_to_sql.api.routes.instructions",
        "nl_to_sql.api.routes.user_settings",
        "nl_to_sql.api.routes.saved_queries",
        "nl_to_sql.api.routes.usage",
        "nl_to_sql.api.routes.data",
        "nl_to_sql.api.routes.account",
        "nl_to_sql.api.routes.auth_sessions",
    ])

    app = FastAPI(
        title="NL-to-SQL RAG API",
        description=(
            "Production-grade pipeline that converts natural language questions "
            "into SQL queries using Retrieval-Augmented Generation (RAG)."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    if trace_endpoint or settings.otel_console_exporter:
        from nl_to_sql.infrastructure.observability.tracing import instrument_app
        instrument_app(app)

    # Store container in app state for access in lifespan
    app.container = container  # type: ignore[attr-defined]

    # ── Rate limiting ──────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)

    # ── GZip compression ──────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── CORS ───────────────────────────────────────────────────────────────────
    # In Docker the nginx reverse-proxy makes all traffic same-origin, so CORS
    # is only relevant for direct API access (dev, tooling, etc.).
    # allow_credentials=False is intentional: the app uses Bearer tokens in
    # Authorization headers, not cookies, so the browser credential flag is not needed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging ────────────────────────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception handlers ─────────────────────────────────────────────────────
    app.add_exception_handler(NLToSQLBaseError, domain_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(auth.router)  # /auth/register, /auth/login, /auth/google, /auth/me
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(schema.router)
    app.include_router(config.router)
    app.include_router(history.router)
    app.include_router(sessions.router)
    app.include_router(analytics.router)
    app.include_router(feedback.router)
    app.include_router(training.router)
    app.include_router(fine_tuning.router)
    app.include_router(profile.router)
    # Phase 1 feature routers
    app.include_router(instructions.router)
    app.include_router(user_settings.router)
    app.include_router(saved_queries.router)
    app.include_router(usage.router)
    app.include_router(data.router)
    app.include_router(account.router)
    app.include_router(auth_sessions.router)

    structlog.get_logger(__name__).info(
        "Application created",
        env=settings.app_env,
        llm_provider=settings.llm_provider,
        vector_store=settings.vector_store_provider,
        dialect=settings.sql_dialect,
    )

    return app
