# Vectrix — Backend

> Where natural language meets your schema.

A REST API that converts plain-English questions into validated, executable SQL using a Retrieval-Augmented Generation (RAG) pipeline, then answers in a sentence with an evidence-based confidence level. Supports PostgreSQL and MySQL (SQLite for local development), streams results over SSE, self-corrects failed SQL, and ships with auth, teams, per-connection access control, analytics, fine-tuning, observability, glossary, query templates, and onboarding.

This file is the reference for backend configuration and the API. The repo-level overview and quickstart live in the [root README](../README.md).

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
- [Environment Variables](#environment-variables)
- [API Overview](#api-overview)
- [Project Structure](#project-structure)
- [RAG Pipeline](#rag-pipeline)
- [Trust: answers, assumptions, confidence](#trust-answers-assumptions-confidence)
- [Teams and access control](#teams-and-access-control)
- [Testing](#testing)
- [SQL accuracy benchmark](#sql-accuracy-benchmark)
- [Observability](#observability)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Natural language → SQL** — RAG pipeline retrieves schema context then prompts the LLM to generate dialect-aware SQL
- **Streaming** — SSE endpoint streams partial results token-by-token as the model generates them
- **Self-correction loop** — validates generated SQL with `sqlglot`; retries up to N times with error context if validation fails
- **Hybrid retrieval** — dense vector search; with Qdrant (the default) it is fused with keyword search. A BM25 + cross-encoder reranking chain (`rag/retrieval/RetrievalChain`) exists but is not on the live query path
- **Query intelligence** — query rewriting, expansion, intent classification, and FK-aware table selection before retrieval
- **Semantic cache** — identical or near-identical questions return cached results instantly without hitting the LLM
- **Schema auto-ingestion** — reflects live database schema on startup, re-ingests on change (configurable poll interval)
- **Plain-English answers** — after execution, a one-sentence answer is written from (at most 20 of) the result rows; users can turn this off in Settings → Data & Privacy
- **Assumptions** — the generator lists the interpretation choices it made (how "revenue" or "this year" was defined) alongside the SQL
- **Evidence-based confidence** — every answer is rated high / medium / low from what the pipeline observed (validation, self-correction, execution, empty results), with plain-English reasons
- **Sample values (opt-in)** — index the distinct values of small categorical columns so filters use exact literals (`RAG_SAMPLE_VALUES_ENABLED`)
- **Teams (workspaces)** — share a database connection, its certified metrics and a workspace glossary; three roles (admin / editor / member); people join by accepting an in-app invitation
- **Platform admin** — accounts with `users.is_admin` can see every user, read and delete any chat, delete connections and teams, and deactivate accounts
- **Per-connection access policy** — allow-list the tables the AI may use and restrict sensitive columns; enforced on every execution path
- **Glossary / Business Dictionary** — per-user term definitions automatically injected into query prompts (token-budget-aware)
- **Query Templates** — parameterized NL + SQL templates with `{{placeholder}}` variables; render endpoint substitutes values at call time
- **Favorited Tables** — users pin tables that receive retrieval priority in hybrid search
- **Onboarding & Tutorial** — server-tracked onboarding checklist and step-by-step tutorial progress, persisted per user
- **Notification Preferences** — per-user email digest, in-app alert, and marketing opt-in settings
- **Authentication** — email/password with OTP verification, Google OAuth 2.0, short-lived JWTs + rotating refresh tokens, password reset. One account per email: Google sign-in links to an existing email account (and counts as verified); registering an email that already exists is refused; Google users add a password through Forgot password (emailed code)
- **Multiple database connections** — connect and switch between several target databases per user (BYOD), each with its own derived schema catalog and vector index
- **Dashboards & auto-charting** — save query results as dashboard widgets; the backend recommends a chart type per result shape
- **Scheduled queries & alerts** — cron-style recurring NL queries with email notification on completion/failure
- **Metrics catalog** — certified, reusable business metrics (name + SQL definition) injected into the SQL-generation prompt
- **Export & share** — export results as CSV/JSON/SQL/PDF, or create a revocable public share link/email/Slack post
- **Multi-turn conversation** — follow-up questions resolve against prior turns in the same chat session
- **BYOK** — users supply their own LLM API keys; server keys are the fallback
- **Analytics** — token usage, success/failure rates, table popularity, intent distribution, prompt version tracking
- **Training data pipeline** — collects feedback, exports fine-tuning JSONL, starts and monitors OpenAI/Together fine-tune jobs
- **Observability** — OpenTelemetry (OTLP), LangFuse tracing, structured JSON logs (structlog)
- **Rate limiting** — per-IP/per-user rate limiting via SlowAPI, configurable requests-per-minute
- **Background workers** — scheduled-query execution, data retention purge, and TTL cleanup run as async tasks

---

## Tech Stack

| Layer | Library / Service | Notes |
|---|---|---|
| API framework | FastAPI + Uvicorn | Async, application factory pattern |
| Language | Python 3.12+ | CI runs 3.12 |
| LLM | Groq · OpenAI · Anthropic · Gemini | Pluggable via `LLM_PROVIDER` env var |
| Embeddings | Gemini `gemini-embedding-001` (default) · Cohere | API-hosted — no local model or PyTorch |
| Vector store | Qdrant (default) · Chroma · FAISS | Swappable via `VECTOR_STORE_PROVIDER` |
| Keyword search | BM25 (rank-bm25) | Hybrid retrieval alongside vector search |
| Database | PostgreSQL via async SQLAlchemy | Supabase recommended for managed hosting |
| Migrations | Alembic | 22 migrations (see `alembic/versions/`); `0021` adds workspaces, the access policy, answer settings and stored answer metadata; `0022` adds workspace invitations |
| Cache | In-memory (default) · Redis · semantic cache | Semantic cache uses a cosine similarity threshold; cache keys include the connection's access-policy fingerprint |
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

- Python 3.12+
- PostgreSQL (or a [Supabase](https://supabase.com) project)
- Qdrant (local install or hosted instance) — or set `VECTOR_STORE_PROVIDER=chroma`/`faiss`
- Redis — optional; only when `CACHE_PROVIDER=redis`
- An API key for your LLM provider (Groq by default) and your embedding provider (Gemini by default)

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

---

## Environment Variables

[`.env.example`](.env.example) is the complete, authoritative list (it also marks which values are secrets). The table below covers the ones you're most likely to set; defaults are the values in `src/nl_to_sql/config/settings.py`. Never commit a filled-in `.env`.

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL asyncpg connection string for the query database |
| `HISTORY_DATABASE_URL` | Yes | PostgreSQL asyncpg connection string for sessions/users/history |
| `SECRET_KEY` | Yes | App secret — `openssl rand -hex 32` |
| `JWT_SECRET_KEY` | Yes | JWT signing secret — `openssl rand -hex 32` |
| `GROQ_API_KEY` | Yes* | Required when `LLM_PROVIDER=groq` |
| `OPENAI_API_KEY` | Yes* | Required when `LLM_PROVIDER=openai` |
| `LLM_PROVIDER` | No | `groq` (default), `openai`, `anthropic` or `gemini` |
| `LLM_MODEL` | No | Default: `openai/gpt-oss-120b` |
| `VECTOR_STORE_PROVIDER` | No | `qdrant` (default), `chroma`, or `faiss` |
| `QDRANT_URL` | No | Default: `http://localhost:6333` |
| `CACHE_PROVIDER` | No | `in_memory` (default) or `redis` — `.env.example` sets `redis` for production-like setups |
| `REDIS_URL` | No | Default: `redis://localhost:6379/0` |
| `GOOGLE_CLIENT_ID` | No | Enables Google OAuth login |
| `SMTP_HOST` / `SMTP_USERNAME` / `SMTP_PASSWORD` | No | Required for OTP email verification |
| `SQL_DIALECT` | No | `postgresql` (default) or `mysql` |
| `SQL_MAX_RETRIES` | No | Self-correction retries on SQL failure (default: 3) |
| `AUTO_INGEST_SCHEMA_ON_STARTUP` | No | Reflect and ingest schema from live DB on boot (default: `true`) |
| `SEMANTIC_CACHE_ENABLED` | No | Cache near-identical questions (default: `true`) |
| `EMBEDDING_PROVIDER` | No | `gemini` (default) or `cohere` — set the matching API key |
| `ADMIN_SETUP_TOKEN` | No | One-time token for the **Set up admin** page (`/setup-admin`): while no owner exists, a signed-in user who enters it becomes the owner. Inert once an owner exists; empty disables setup. Admin status itself lives in `users.is_admin` / `users.is_owner` |
| `TARGET_DB_READONLY` / `TARGET_STATEMENT_TIMEOUT_MS` | No | Generated SQL runs read-only (default `true`) with a timeout (default `30000`) — keep these on |
| `RAG_SAMPLE_VALUES_ENABLED` | No | Index distinct values of small categorical columns on the next schema sync (default `false`). Reads row data, honours the access policy, skips columns whose names look personal |
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
| **Auth** | `POST /auth/register` `POST /auth/login` `POST /auth/google` `POST /auth/verify-otp` `POST /auth/forgot-password` `POST /auth/reset-password` `POST /auth/refresh` | Registration, login, OAuth, OTP, refresh-token rotation |
| **Auth Sessions** | `GET /auth-sessions` `DELETE /auth-sessions/{id}` `POST /auth-sessions/revoke-all` | Active login-session management |
| **Query** | `POST /query` `POST /query/stream` (SSE) `POST /query/explain` `POST /query/execute` `POST /query/suggestions` | Core NL→SQL pipeline, incl. multi-turn follow-ups |
| **Schema** | `POST /schema/sync` `POST /schema/ingest` `GET /schema/tables` `GET /schema/status` `GET /schema/visualize` | Schema catalog: live-DB sync + upload overlay |
| **Connections** | `GET/POST /connections` `PUT/DELETE /connections/{id}` `POST /connections/{id}/test` `POST /connections/{id}/select` `PUT /connections/{id}/policy` `PUT /connections/{id}/workspace` | Multiple database connections (BYOD) per user, including ones shared with you; access policy and sharing (owner only) |
| **Workspaces** | `GET/POST /workspaces` `GET/PATCH/DELETE /workspaces/{id}` `GET /workspaces/directory` `POST /workspaces/{id}/invitations` `DELETE /workspaces/{id}/invitations/{inv_id}` `GET /workspaces/invitations` `POST /workspaces/invitations/{inv_id}/accept` `/decline` `PATCH/DELETE /workspaces/{id}/members/{user_id}` `GET/POST /workspaces/{id}/glossary` `PATCH/DELETE /workspaces/{id}/glossary/{entry_id}` | Teams: invitations, directory, members and roles, workspace glossary |
| **Admin** | `GET /admin/users` `GET/PATCH /admin/users/{id}` `GET/DELETE /admin/users/{id}/chats/{session_id}` `DELETE /admin/users/{id}/connections/{connection_id}` `DELETE /admin/workspaces/{id}` | Platform admin — **admin only**; never returns connection secrets |
| **Sessions** | `GET/POST /sessions` `GET/DELETE /sessions/{id}` `POST /sessions/{id}/messages` | Chat session history |
| **History** | `GET /history` `DELETE /history` `GET /history/export` | Query history log |
| **Saved Queries** | `GET/POST /saved-queries` `PATCH/DELETE /saved-queries/{id}` `POST /saved-queries/{id}/run` | Bookmarked queries |
| **Query Templates** | `GET/POST /query-templates` `GET/PATCH/DELETE /query-templates/{id}` `POST /query-templates/{id}/render` | Parameterized NL+SQL templates with `{{placeholder}}` substitution |
| **Glossary** | `GET/POST /glossary` `GET/PATCH/DELETE /glossary/{id}` | Business dictionary — terms injected into query prompts |
| **Favorited Tables** | `GET/POST /favorited-tables` `DELETE /favorited-tables/{id}` | User-pinned tables with retrieval priority |
| **Dashboards** | `GET/POST /dashboards` `GET/PATCH/DELETE /dashboards/{id}` `POST /dashboards/{id}/duplicate` | Saved widgets with auto-recommended chart type |
| **Scheduled Queries** | `GET/POST /schedules` `GET/PUT/DELETE /schedules/{id}` `POST /schedules/{id}/pause` `/resume` `/run-now` | Recurring NL queries with email alerts |
| **Metrics** | `GET/POST /metrics` `GET/PUT/DELETE /metrics/{id}` `POST /metrics/{id}/certify` `/uncertify` | Certified business-metrics catalog |
| **Exports & Shares** | `POST /exports/{format}` `POST /shares` `DELETE /shares/{id}` `GET /shares/{token}` (public) | CSV/JSON/SQL/PDF export, revocable share links |
| **Onboarding** | `GET/PATCH /onboarding` | Onboarding checklist state (7 items, progress %) |
| **Tutorial** | `GET /tutorial/progress` `POST /tutorial/step/{step_id}/complete` `POST /tutorial/reset` | Step-by-step tutorial progress |
| **Notification Prefs** | `GET/PATCH /notification-preferences` | Email digest, in-app alerts, marketing opt-ins |
| **Analytics** | `GET /analytics/summary` `/popular-queries` `/failure-patterns` `/table-usage` `/intent-distribution` | Usage analytics |
| **Training** | `GET /training/stats` `GET /training/export` `GET /training/download` | Fine-tuning data — admin only |
| **Fine-tuning** | `POST /fine-tuning/prepare` `POST /fine-tuning/start` `GET /fine-tuning/status/{id}` `GET /fine-tuning/jobs` `POST /fine-tuning/deploy` | LLM fine-tuning jobs — **admin only** (the training export spans every user's queries) |
| **Profile / BYOK** | `GET/PUT/DELETE /profile/api-keys/{provider}` | Per-user API keys |
| **Config** | `GET/PUT /config/database` `GET/PUT /config/llm` `GET/PUT /config/rag` `GET /config/models` | Runtime config, incl. RAG feature flags |
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
│   │   └── routes/             # One module per route group (~29 routers)
│   │       ├── query.py            # Core NL→SQL + stream + explain + execute
│   │       ├── auth.py             # Registration, login, OAuth, OTP
│   │       ├── connections.py      # Multiple DB connections (BYOD) per user
│   │       ├── schema.py           # Schema catalog sync/upload
│   │       ├── dashboards.py       # Saved widgets + auto-charting
│   │       ├── schedules.py        # Scheduled queries & alerts
│   │       ├── metrics.py          # Certified metrics catalog
│   │       ├── exports.py          # Export (CSV/JSON/SQL/PDF) + share links
│   │       ├── glossary.py         # Business dictionary CRUD
│   │       ├── query_templates.py  # Parameterized template CRUD + render
│   │       ├── favorited_tables.py # Pinned table management
│   │       ├── onboarding.py       # Onboarding checklist state
│   │       ├── tutorial.py         # Tutorial step progress
│   │       ├── notification_prefs.py  # Notification preferences
│   │       └── ...                 # remaining route modules
│   ├── config/
│   │   ├── settings.py         # Pydantic-settings, all env vars
│   │   └── container.py        # dependency-injector ApplicationContainer
│   ├── core/
│   │   ├── access_policy.py    # Per-connection table allow-list / restricted columns (pure logic)
│   │   ├── models/             # Pydantic domain models (query, auth, schema, …)
│   │   ├── interfaces/         # Abstract base classes (ILLMProvider, IVectorStore, …)
│   │   └── exceptions.py       # Domain exception hierarchy
│   ├── infrastructure/
│   │   ├── database/           # Async SQLAlchemy client, ORM models, schema sync,
│   │   │                       # policy_client.py (applies the access policy to every query)
│   │   ├── llm/                # OpenAI + Groq provider adapters
│   │   ├── embeddings/         # Gemini API embedder
│   │   ├── vector_store/       # Qdrant, Chroma, FAISS adapters
│   │   ├── cache/              # Redis, in-memory, semantic cache
│   │   ├── bm25_store.py       # BM25 index (rank-bm25)
│   │   └── observability/      # OpenTelemetry setup + LangFuse client
│   ├── rag/
│   │   ├── ingestion/          # Schema → chunks → embeddings → vector store
│   │   └── retrieval/          # Query embed → BM25 + vector search → rerank → context
│   ├── services/               # Business logic layer (26+ service modules)
│   │   ├── query_orchestrator.py   # Top-level pipeline coordinator
│   │   ├── answer_summarizer.py    # One-sentence answer from executed rows
│   │   ├── answer_confidence.py    # Evidence-based high/medium/low confidence
│   │   ├── workspace_service.py    # Teams: invitations, directory, membership + the only place roles are checked
│   │   ├── admin_service.py        # Platform admin: every user, chat, connection, team
│   │   ├── sql_generator.py        # LLM prompting
│   │   ├── sql_validator.py        # sqlglot validation + self-correction
│   │   ├── auth_service.py         # JWT, OAuth, OTP
│   │   ├── analytics_service.py
│   │   ├── fine_tuning_service.py
│   │   └── ...
│   ├── evaluation/             # SQL accuracy benchmark (result comparison + runner)
│   └── workers/
│       ├── retention_worker.py # Async background data retention enforcement
│       └── purge_worker.py     # Expired record purge
├── alembic/
│   └── versions/                # 22 migrations (0021: workspaces, access policy, answers; 0022: invitations)
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
                    │  2. Vector search        │  (hybrid with keywords on Qdrant)
                    │  3. LLM table selection  │
                    │  4. FK graph expansion   │
                    │  5. Access-policy filter │  (drop disallowed tables / restricted columns)
                    │  6. Inject glossary      │  (personal + workspace terms)
                    │  7. Context builder      │
                    └────────────┬────────────┘
                                 │ schema context + glossary
                    ┌────────────▼────────────┐
                    │  SQL Generator (LLM)    │  → SQL + assumptions
                    │  → sqlglot validate     │
                    │  → self-correct (×N)    │
                    │  → execute (read-only,  │
                    │    policy-enforced)     │
                    │  → answer + confidence  │
                    └─────────────────────────┘
```

`rag/retrieval/RetrievalChain` (BM25 + RRF fusion + cross-encoder reranking) is built and tested but is **not** on the live query path, which uses `services/schema_retriever.py` + `services/query_orchestrator.py`.

---

## Trust: answers, assumptions, confidence

Every `QueryResponse` (and every saved chat message) carries four fields aimed at readers who can't check SQL themselves:

| Field | Source | Notes |
|---|---|---|
| `answer` | `services/answer_summarizer.py` | One or two sentences written from at most 20 result rows (≤ 4 KB). Only when the query was executed and returned rows. The only path that sends row data to the LLM — users can turn it off (`answer_summary_enabled` in `PATCH /settings`); scheduled queries never use it. |
| `assumptions` | the SQL generator's JSON output | 0–4 plain-English interpretation choices, cleaned and length-capped. |
| `confidence` | `services/answer_confidence.py` | `low` when the schema can't answer it, no tables matched, validation failed after all retries, or execution failed. `medium` when it needed self-correction, returned no rows, used an unretrieved table, or made 3+ assumptions. Otherwise `high`. Deterministic — the model never rates itself. |
| `confidence_reasons` | same | Plain-English reasons for `medium`/`low`. |

## Teams and access control

**Workspaces.** Any user can create a workspace and becomes its admin. Admins **invite** verified Vectrix accounts and pick a role; the invitee gets an in-app notification (badge + toast, from `GET /workspaces/invitations`) and accepts or declines on the Team page. Only acceptance creates a membership — invitations live in their own table, so pending people never get access. Who can be picked (`GET /workspaces/directory`): verified accounts on the **same email domain** as the inviter (everyone on `@gmail.com` counts as one company); anyone else only by typing their **exact** email; platform admins see everyone. Roles:

| Role | Can |
|---|---|
| `member` | Query connections shared into the workspace; read its glossary and certified metrics |
| `editor` | …and manage the workspace glossary |
| `admin` | …and manage members and roles, share their own connections into the workspace, rename or delete it |

Role checks live only in `services/workspace_service.py`. Non-members get `404` (existence is never confirmed); members without the needed role get `403 WORKSPACE_FORBIDDEN`. The last admin can't leave or be demoted. Only connection-level knowledge is shared — chats, saved queries, dashboards and schedules stay personal. A member selects a shared connection like their own; if their access is revoked they fall back to their own default on the next request.

**Access policy (per connection, owner-managed).** `allowed_tables` (null = all) and `restricted_columns` (`table.column`) are enforced in three independent layers:

1. **Schema context** — disallowed tables and restricted column lines (including sample values) are removed before the LLM sees them.
2. **SQL check** — every statement on every path (`execute_sql`, `explain`, streaming execution) is parsed with sqlglot; a reference to a disallowed table or restricted column in any clause, through any alias or CTE, is rejected with `403 ACCESS_POLICY_VIOLATION`. Unparseable SQL is rejected while a policy is active. Inside the query pipeline the rejection is fed back to the model, which rewrites the query.
3. **Row redaction** — restricted columns that surface through `SELECT *` are returned as `[restricted]`.

The policy is applied by wrapping the connection's client (`infrastructure/database/policy_client.py`) wherever a client is resolved, so interactive queries, previews, dashboards and scheduled queries all inherit it.

**Platform admins.** Accounts with `users.is_admin = true` get `/api/v1/admin/*`: list and search every account, read any user's chats (questions, SQL, answers, up to 50 result rows per message), delete chats / connections / teams, and deactivate or reactivate accounts (deactivation revokes the user's sessions and refresh tokens; every request re-checks `is_active`). Deletions and deactivations are written to the application log; reading a user's data is not logged, and there is no audit table. Connection strings and API keys are never returned.

**Who is an admin.** `users.is_admin` marks admins; `users.is_owner` marks the single owner (a partial unique index allows only one). The first owner is claimed once on the **Set up admin** page (`/setup-admin`, `POST /api/v1/admin/setup`) with `ADMIN_SETUP_TOKEN`; it only works while no owner exists. After that, any admin can make another active, verified account an admin from **Admin → Admins** (`POST /api/v1/admin/admins`), but only the owner can remove an admin (`DELETE /api/v1/admin/admins/{user_id}`) or deactivate an admin's account. The owner can't be removed.

**Deactivated accounts.** A deactivated account never gets a session: a valid password or Google sign-in answers `403 ACCOUNT_DEACTIVATED` (or `403 REACTIVATION_PENDING` when a request is already waiting) and no tokens are issued. The owner is emailed about the attempt (at most once per account per hour, per worker). The `ACCOUNT_DEACTIVATED` error carries a 30-minute reactivation token (`details.detail`) that the sign-in page uses to send `POST /api/v1/auth/reactivation-request` with an optional message; every active admin is emailed. Admins approve (reactivates the account) or reject on **Admin → Requests** (`/api/v1/admin/reactivation-requests`), and the user is emailed the outcome. Reactivating the account directly from the Users tab also closes its pending request. Requests live in `account_reactivation_requests` (one pending per user). Emails need the SMTP settings; without them the flow still works, but nothing is sent.

**Daily query limits.** Each AI query generation (`POST /query`, `/query/stream`) counts against the user's allowance for the current UTC day: their own `users.daily_query_limit` if an admin set one (Admin → user → Daily query limit, `PUT /api/v1/admin/users/{id}/limit`), else `DEFAULT_DAILY_QUERY_LIMIT` (0 = unlimited). Admins are never limited. Over the limit the API answers `429 DAILY_LIMIT_REACHED` with `Retry-After` set to midnight UTC. Counts live in `user_daily_usage`, separate from chat history, so deleting chats doesn't give the allowance back; the increment is a single conditional `UPDATE`, so concurrent requests can't overshoot. `GET /api/v1/usage` reports `queries_today` and `daily_limit`.

**Audit log.** Every admin change is recorded in `admin_audit_events` and shown on Admin → Audit log (`GET /api/v1/admin/audit`, filterable by action): owner claimed, admin granted/revoked, account deactivated/reactivated, reactivation approved/rejected, chat/connection/team deleted, daily limit changed. Viewing a user's data is not recorded. Recording is best-effort — a failed write is logged and never undoes the action.

**Databases.** Connections can be PostgreSQL or MySQL (`aiomysql`). SQLite connections are accepted only outside production: a SQLite URL is a file on the server, so in production it would let users read server-side files. Schema reading and read-only enforcement cover all three — PostgreSQL via `pg_catalog` and `SET TRANSACTION READ ONLY`, MySQL via SQLAlchemy's inspector with `SET TRANSACTION READ ONLY` and `max_execution_time`, SQLite with `PRAGMA query_only`. Tables outside PostgreSQL's `public` schema are written `schema.table` in prompts and the graph.

Create or reset an admin account (the password is read from the environment, never the command line):

```bash
ADMIN_PASSWORD='…at least 12 characters…' python scripts/create_admin.py admin@example.com
```

The account is marked `is_admin`; add `--owner` to also make it the owner (refused if another account already is).

---

## Testing

```bash
make test               # unit + integration (pytest)
make lint               # ruff
make type-check         # mypy
```

CI (`.github/workflows/ci.yml`) runs on every push and pull request: `ruff check src tests`, `mypy src/nl_to_sql`, a single-Alembic-head check, and the full pytest suite. It needs no secrets — the tests use temporary SQLite databases and stubbed LLM/SMTP clients, and settings fall back to safe defaults without a `.env`.

Security-relevant behaviour is tested against a real (temporary SQLite) database rather than mocks — see `tests/unit/test_workspace_sharing.py` (roles, sharing, revocation) and `tests/unit/test_access_policy.py` (alias/CTE/filter/`*` bypass attempts).

## SQL accuracy benchmark

Measures whether generated SQL returns the **right answer**, not just valid SQL: each question in `tests/fixtures/golden_sql.json` (60 cases across 9 categories) is run through the real pipeline, then the generated SQL and a hand-written gold query are both executed read-only and their result sets compared (column names/order ignored; row order only when the gold query orders).

```bash
make seed && make ingest                     # once: create + index the sample e-commerce schema
make eval-sql ARGS="--limit 15"              # quick run
make eval-sql ARGS="--out report.json --fail-under 0.8"
```

It needs an LLM key and a database containing the sample schema (`DATABASE_URL`); it exits with a clear message if either is missing. It bypasses the response cache and never writes training data. The result-comparison logic itself is unit-tested and runs in CI (`tests/unit/test_result_compare.py`, `tests/unit/test_benchmark.py`).

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

### Running in production

```bash
pip install -e "."
alembic upgrade head
uvicorn nl_to_sql.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4
```

Run behind a reverse proxy (e.g. Nginx) with TLS termination.

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
