# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DEPRECATED — do not use for new builds                                    ║
# ║                                                                              ║
# ║  This was the fallback Dockerfile for when the build context had to be      ║
# ║  ./backend only.  The canonical build is now:                               ║
# ║                                                                              ║
# ║    docker build -f docker/Dockerfile .          (production)                ║
# ║    docker-compose -f docker/docker-compose.yml up   (local dev)             ║
# ║                                                                              ║
# ║  This file lists deps by hand and diverges from pyproject.toml — it will    ║
# ║  drift and is NOT kept up-to-date.  Use docker/Dockerfile instead.         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install all production dependencies directly (mirrors pyproject.toml [dependencies])
RUN pip install --no-cache-dir \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "groq>=0.9.0" \
    "openai>=1.0.0" \
    "chromadb>=0.5.0" \
    "faiss-cpu>=1.8.0" \
    "sentence-transformers>=2.2.0" \
    "rank-bm25>=0.2.2" \
    "sqlglot>=23.0.0" \
    "sqlalchemy[asyncio]>=2.0.30" \
    "asyncpg>=0.29.0" \
    "redis[asyncio]>=5.0.0" \
    "opentelemetry-api>=1.20.0" \
    "opentelemetry-sdk>=1.20.0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20.0" \
    "pydantic[email]>=2.7.0" \
    "pydantic-settings>=2.2.0" \
    "python-dotenv>=1.0.0" \
    "dependency-injector>=4.41.0" \
    "structlog>=24.1.0" \
    "httpx>=0.27.0" \
    "slowapi>=0.1.9" \
    "passlib[bcrypt]>=1.7.4" \
    "pyjwt>=2.8.0" \
    "google-auth>=2.29.0" \
    "python-jose[cryptography]>=3.3.0" \
    "aiosmtplib>=3.0.1"

# ── Stage 2: Production ───────────────────────────────────────────────────────
FROM python:3.13-slim AS production

RUN useradd --no-create-home --shell /bin/false appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source
COPY src/ ./src/

RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "nl_to_sql.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
