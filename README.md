# NL2SQL — Backend

A production-grade REST API that converts plain-English questions into validated, executable SQL using a Retrieval-Augmented Generation (RAG) pipeline. Supports PostgreSQL and MySQL, streams results over SSE, self-corrects failed SQL, and ships with auth, analytics, fine-tuning, observability, glossary, query templates, and onboarding built in.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Docker (Recommended)](#docker-recommended)
- [Environment Variables](#environment-variables)
- [API Overview](#api-overview)
- [Project Structure](#project-structure)
- [RAG Pipeline](#rag-pipeline)
- [Observability](#observability)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Natural language → SQL** — RAG pipeline retrieves schema context then prompts the LLM to generate dialect-aware SQL
- **Streaming** — SSE endpoint streams partial results token-by-token as the model generates them
- **Self-correction loop** — validates generated SQL with `sqlglot`; retries up to N times with error context if validation fails
- **Hybrid retrieval** — combines dense vector search (Qdrant/Chroma/FAISS) with BM25 keyword search, re-ranked by a cross-encoder
- **Query intelligence** — query rewriting, expansion, intent classification, and FK-aware table selection before retrieval
- **Semantic cache** — identical or near-identical questions return cached results instantly without hitting the LLM
- **Schema auto-ingestion** — reflects live database schema on startup, re-ingests on change (configurable poll interval)
- **Glossary / Business Dictionary** — per-user term definitions automatically injected into query prompts (token-budget-aware)
- **Query Templates** — parameterized NL + SQL templates with `{{placeholder}}` variables; render endpoint substitutes values at call time
- **Favorited Tables** — users pin tables that receive retrieval priority in hybrid search
- **Onboarding & Tutorial** — server-tracked onboarding checklist and step-by-step tutorial progress, persisted per user
- **Notification Preferences** — per-user email digest, in-app alert, and marketing opt-in settings
- **Authentication** — email/password with OTP verification, Google OAuth 2.0, JWT sessions, password reset
- **BYOK** — users supply their own LLM API keys; server keys are the fallback
- **Analytics** — token usage, success/failure rates, table popularity, intent distribution, prompt version tracking
- **Training data pipeline** — collects feedback, exports fine-tuning JSONL, starts and monitors OpenAI/Together fine-tune jobs
- **Observability** — OpenTelemetry (OTLP), LangFuse tracing, structured JSON logs (structlog)
- **Rate limiting** — per-IP rate limiting via SlowAPI, configurable requests-per-minute
- **Background workers** — data retention purge and TTL cleanup run as async tasks

---

## Tech Stack

| Layer | Library / Service | Notes |
|---|---|---|
| API framework | FastAPI + Uvicorn | Async, application factory pattern |
| Language | Python 3.13 | |
| LLM | Groq · OpenAI · Anthropic · Gemini | Pluggable via `LLM_PROVIDER` env var |
| Embeddings | Gemini `text-embedding-004` (API) · HuggingFace (local) | Pluggable via `EMBEDDING_PROVIDER`; no PyTorch required for Gemini |
| Vector store | Qdrant (default) · Chroma · FAISS | Swappable via `VECTOR_STORE_PROVIDER` |
| Keyword search | BM25 (rank-bm25) | Hybrid retrieval alongside vector search |
| Database | PostgreSQL via async SQLAlchemy | Supabase recommended for managed hosting |
| Migrations | Alembic | 3 migrations: initial schema, perf indexes, filter column indexes |
| Cache | Redis · in-memory fallback · semantic cache | Semantic cache uses cosine similarity threshold |
| Auth | JWT + Google OAuth + aiosmtplib OTP | passlib bcrypt, python-jose |
| DI | dependency-injector | Constructor-injection container |
| SQL parsing | sqlglot | Dialect-aware validation and formatting |
| Rate limiting | SlowAPI | Per-IP, configurable |
| Logging | structlog | JSON output, weekly-rotating file handler |
| Observability | OpenTelemetry · LangFuse | All optional; LangFuse singleton initialized at startup |
| Build | Hatchling | pyproject.toml, no setup.py |
| Linting | Ruff | Replaces flake8 + isort + pyupgrade |
| Type checking | Mypy (strict) | |
| Testing | pytest + pytest-asyncio | |
| Container | Docker multi-stage (python:3.13-slim) | Non-root `appuser`, healthcheck included |

---

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │           FastAPI Application         │
                          │                                       │
   HTTP / SSE ───────────►│  Routes → Services → Orchestrator    │
                          │                                       │
                          └────────────┬──────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │              Query Orchestrator                  │
              │                                                  │
              │  1. Classify intent (SELECT / aggregation / DDL) │
              │  2. Rewrite + expand query                       │
              │  3. Select candidate tables (FK-aware)           │
              │  4. Hybrid retrieval  ──► Reranker               │
              │  5. Build schema context + inject glossary       │
              │  6. Prompt LLM → generate SQL                    │
              │  7. Validate (sqlglot) → self-correct if needed  │
              │  8. Execute (optional) → cache → stream          │
              └──────┬──────────────┬──────────────┬────────────┘
                     │              │              │
            ┌────────▼───┐  ┌───────▼──────┐  ┌───▼──────────┐
            │  Vector DB  │  │  PostgreSQL  │  │  Redis Cache │
            │  (Qdrant)   │  │  (Supabase)  │  │              │
            └────────────┘  └──────────────┘  └──────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- PostgreSQL (or a [Supabase](https://supabase.com) project)
- Qdrant — `docker run -p 6333:6333 qdrant/qdrant`
- Redis — `docker run -p 6379:6379 redis:7-alpine`
- A Groq or OpenAI API key

### Local Development

```bash
git clone https://github.com/nikhil-sharma-devx/Nl2SQL-Backend.git
cd Nl2SQL-Backend

# Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install package + dev dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env — set DATABASE_URL, HISTORY_DATABASE_URL, GROQ_API_KEY, SECRET_KEY, JWT_SECRET_KEY

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn nl_to_sql.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

### Docker (Recommended)

Starts the API, Qdrant, and Redis in one command:

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL, GROQ_API_KEY, SECRET_KEY, JWT_SECRET_KEY at minimum

docker compose up --build
```

The `api` container waits for Qdrant and Redis health checks before starting.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in every `REQUIRED` value. All other variables have sensible defaults.

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL asyncpg connection string for the query database |
| `HISTORY_DATABASE_URL` | Yes | PostgreSQL asyncpg connection string for sessions/users/history |
| `SECRET_KEY` | Yes | App secret — `openssl rand -hex 32` |
| `JWT_SECRET_KEY` | Yes | JWT signing secret — `openssl rand -hex 32` |
| `GROQ_API_KEY` | Yes* | Required when `LLM_PROVIDER=groq` |
| `OPENAI_API_KEY` | Yes* | Required when `LLM_PROVIDER=openai` |
| `LLM_PROVIDER` | No | `groq` (default) or `openai` |
| `LLM_MODEL` | No | Default: `llama-3.3-70b-versatile` |
| `VECTOR_STORE_PROVIDER` | No | `qdrant` (default), `chroma`, or `faiss` |
| `QDRANT_URL` | No | Default: `http://qdrant:6333` (Docker) or `http://localhost:6333` |
| `CACHE_PROVIDER` | No | `redis` (default) or `in_memory` |
| `REDIS_URL` | No | Default: `redis://redis:6379/0` |
| `GOOGLE_CLIENT_ID` | No | Enables Google OAuth login |
| `SMTP_HOST` / `SMTP_USERNAME` / `SMTP_PASSWORD` | No | Required for OTP email verification |
| `SQL_DIALECT` | No | `postgresql` (default) or `mysql` |
| `SQL_MAX_RETRIES` | No | Self-correction retries on SQL failure (default: 3) |
| `AUTO_INGEST_SCHEMA_ON_STARTUP` | No | Reflect and ingest schema from live DB on boot (default: `true`) |
| `SEMANTIC_CACHE_ENABLED` | No | Cache near-identical questions (default: `true`) |
| `LANGFUSE_SECRET_KEY` | No | LangFuse secret key — enables LLM trace logging |
| `LANGFUSE_PUBLIC_KEY` | No | LangFuse public key |
| `LANGFUSE_HOST` | No | LangFuse host (default: `https://cloud.langfuse.com`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | OpenTelemetry collector endpoint |

See [`.env.example`](.env.example) for the complete reference including fine-tuning and rate limiting variables.

---

## API Overview

All routes are prefixed `/api/v1` and documented at `/docs`.

| Group | Endpoints | Description |
|---|---|---|
| **Auth** | `POST /auth/register` `POST /auth/login` `POST /auth/google` `POST /auth/verify-otp` `POST /auth/forgot-password` `POST /auth/reset-password` | Registration, login, OAuth, OTP |
| **Query** | `POST /query` `POST /query/stream` (SSE) `POST /query/explain` `POST /query/execute` `POST /query/suggestions` | Core NL→SQL pipeline |
| **Schema** | `POST /schema/ingest` `POST /schema/refresh` `GET /schema/status` `GET /schema/visualize` | Schema management |
| **Sessions** | `GET/POST /sessions` `GET/DELETE /sessions/{id}` `POST /sessions/{id}/messages` | Chat session history |
| **History** | `GET /history` `DELETE /history` `GET /history/export` | Query history log |
| **Saved Queries** | `GET/POST /saved-queries` `PATCH/DELETE /saved-queries/{id}` `POST /saved-queries/{id}/run` | Bookmarked queries |
| **Query Templates** | `GET/POST /query-templates` `GET/PATCH/DELETE /query-templates/{id}` `POST /query-templates/{id}/render` | Parameterized NL+SQL templates with `{{placeholder}}` substitution |
| **Glossary** | `GET/POST /glossary` `GET/PATCH/DELETE /glossary/{id}` | Business dictionary — terms injected into query prompts |
| **Favorited Tables** | `GET/POST /favorited-tables` `DELETE /favorited-tables/{id}` | User-pinned tables with retrieval priority |
| **Onboarding** | `GET/PATCH /onboarding` | Onboarding checklist state (7 items, progress %) |
| **Tutorial** | `GET /tutorial/progress` `POST /tutorial/step/{step_id}/complete` `POST /tutorial/reset` | Step-by-step tutorial progress |
| **Notification Prefs** | `GET/PATCH /notification-preferences` | Email digest, in-app alerts, marketing opt-ins |
| **Analytics** | `GET /analytics/summary` `/popular-queries` `/failure-patterns` `/table-usage` `/intent-distribution` | Usage analytics |
| **Training** | `GET /training/stats` `GET /training/export` `GET /training/download` | Fine-tuning data |
| **Fine-tuning** | `POST /fine-tuning/prepare` `POST /fine-tuning/start` `GET /fine-tuning/status/{id}` `POST /fine-tuning/deploy` | LLM fine-tuning jobs |
| **Profile / BYOK** | `GET/PUT/DELETE /profile/api-keys/{provider}` | Per-user API keys |
| **Config** | `GET/PUT /config/database` `GET/PUT /config/llm` `GET /config/models` | Runtime config |
| **Settings** | `GET/PATCH /settings` `GET/PUT /instructions` | User preferences |
| **Account** | `GET/PUT /account/retention` `POST /account/delete` | Account management |
| **Health** | `GET /health` `GET /ready` | Liveness and readiness probes |

---

## Project Structure

```
backend/
├── src/nl_to_sql/
│   ├── api/
│   │   ├── app.py              # Application factory (lifespan, middleware, routers)
│   │   ├── dependencies.py     # FastAPI dependency providers
│   │   ├── middleware/         # Error handler, request logger, rate limiter
│   │   └── routes/             # One module per route group (25 routers)
│   │       ├── query.py            # Core NL→SQL + stream + explain + execute
│   │       ├── auth.py             # Registration, login, OAuth, OTP
│   │       ├── glossary.py         # Business dictionary CRUD
│   │       ├── query_templates.py  # Parameterized template CRUD + render
│   │       ├── favorited_tables.py # Pinned table management
│   │       ├── onboarding.py       # Onboarding checklist state
│   │       ├── tutorial.py         # Tutorial step progress
│   │       ├── notification_prefs.py  # Notification preferences
│   │       └── ...                 # 17 additional route modules
│   ├── config/
│   │   ├── settings.py         # Pydantic-settings, all env vars
│   │   └── container.py        # dependency-injector ApplicationContainer
│   ├── core/
│   │   ├── models/             # Pydantic domain models (query, auth, schema, …)
│   │   ├── interfaces/         # Abstract base classes (ILLMProvider, IVectorStore, …)
│   │   └── exceptions.py       # Domain exception hierarchy
│   ├── infrastructure/
│   │   ├── database/           # Async SQLAlchemy client, ORM models, schema sync
│   │   ├── llm/                # OpenAI + Groq provider adapters
│   │   ├── embeddings/         # Gemini API embedder (default) · HuggingFace (local fallback)
│   │   ├── vector_store/       # Qdrant, Chroma, FAISS adapters
│   │   ├── cache/              # Redis, in-memory, semantic cache
│   │   ├── bm25_store.py       # BM25 index (rank-bm25)
│   │   └── observability/      # OpenTelemetry setup + LangFuse client
│   ├── rag/
│   │   ├── ingestion/          # Schema → chunks → embeddings → vector store
│   │   └── retrieval/          # Query embed → BM25 + vector search → rerank → context
│   ├── services/               # Business logic layer (26+ service modules)
│   │   ├── query_orchestrator.py   # Top-level pipeline coordinator
│   │   ├── sql_generator.py        # LLM prompting
│   │   ├── sql_validator.py        # sqlglot validation + self-correction
│   │   ├── auth_service.py         # JWT, OAuth, OTP
│   │   ├── analytics_service.py
│   │   ├── fine_tuning_service.py
│   │   └── ...
│   └── workers/
│       ├── retention_worker.py # Async background data retention enforcement
│       └── purge_worker.py     # Expired record purge
├── alembic/
│   └── versions/
│       ├── 0001_initial_schema.py
│       ├── 0002_performance_indexes.py
│       └── 0003_filter_column_indexes.py   # Indexes on deleted_at, status columns
├── Dockerfile                  # Multi-stage build (python:3.13-slim)
├── docker-compose.yml          # API + Qdrant + Redis
├── pyproject.toml              # Hatchling build, Ruff, Mypy, pytest config
└── .env.example                # Full environment variable reference
```

---

## RAG Pipeline

```
                    ┌─────────────────────────┐
  Schema DDL ──────►│   Ingestion Pipeline     │
  (live DB or file) │                          │
                    │  Schema loader           │
                    │    → Chunker             │
                    │    → Doc builder         │
                    │    → Gemini embed        │
                    │    → BM25 indexer        │
                    │    → Vector writer       │
                    └─────────────────────────┘

                    ┌─────────────────────────┐
  User question ───►│   Retrieval Pipeline    │
                    │                         │
                    │  1. Embed query          │
                    │  2. Vector search        │──┐
                    │  3. BM25 search          │  ├─► Cross-encoder rerank
                    │  4. FK graph expansion   │──┘
                    │  5. Inject glossary      │
                    │  6. Context builder      │
                    └────────────┬────────────┘
                                 │ schema context + glossary
                    ┌────────────▼────────────┐
                    │  SQL Generator (LLM)    │
                    │  → sqlglot validate     │
                    │  → self-correct (×N)    │
                    └─────────────────────────┘
```

---

## Observability

The app ships with optional observability integrations — all configured via env vars, none required in local dev:

| Tool | Env vars | What it captures |
|---|---|---|
| **OpenTelemetry** | `OTEL_EXPORTER_OTLP_ENDPOINT` | Spans for every request, LLM call, DB query |
| **LangFuse** | `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_HOST` | LLM traces, generations, prompt diffs — singleton client flushed on shutdown |

Structured JSON logs are written by `structlog` to stdout and an optional weekly-rotating log file (`APP_LOG_FILE`).

---

## Deployment

### Cloud database (Supabase)

The app is designed to work with [Supabase](https://supabase.com) managed PostgreSQL out of the box. Use the **Direct connection** string from *Project Settings → Database* as `DATABASE_URL` and `HISTORY_DATABASE_URL`.

### Production Docker

```bash
# Build
docker build -t nl2sql-backend .

# Run with your .env
docker run --env-file .env -p 8000:8000 nl2sql-backend
```

### Scaling

- **Stateless workers** — all state lives in PostgreSQL + Qdrant + Redis, so multiple API replicas work without coordination.
- **Uvicorn workers** — set `WEB_CONCURRENCY` env var or `--workers` in the `CMD` to match your vCPU count.
- **Vector store** — Qdrant supports horizontal scaling and cloud-hosted deployments.

---

## Contributing

This project is maintained by [@nikhil-sharma-devx](https://github.com/nikhil-sharma-devx).

---

## License

MIT — see [LICENSE](LICENSE) for details.
