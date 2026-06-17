# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip hatchling

# Copy project definition and source
COPY pyproject.toml .
COPY src/ ./src/

# Install the package and all production dependencies
RUN pip install --no-cache-dir .

# ── Stage 2: Production ───────────────────────────────────────────────────────
FROM python:3.12-slim AS production

RUN useradd --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copy installed packages and entry-points from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source
COPY src/ ./src/

# Persistent storage for vector data and logs
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "nl_to_sql.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
