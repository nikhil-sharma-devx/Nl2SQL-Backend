"""Application settings — loaded from environment variables or .env file."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import structlog
from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

def find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or .env up the tree."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".env").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()

# Load .env into os.environ so all external libraries (like huggingface_hub) can access their keys/settings.
load_dotenv(PROJECT_ROOT / ".env")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"


class Settings(BaseSettings):
    """All application configuration via Pydantic BaseSettings.

    Values are read from environment variables (case-insensitive) or a
    .env file in the working directory.

    SOLID:
      S — Only concerns itself with configuration values.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"  # noqa: S104 — required for Docker/container deployment
    app_port: int = 8000
    app_log_level: str = "INFO"
    # Default writes to {project_root}/data/logs/; override with APP_LOG_FILE= (empty) to disable
    app_log_file: str = str(PROJECT_ROOT / "data" / "logs" / "application.log")
    secret_key: str = "change-me-in-production"

    # ── Authentication ───────────────────────────────────────────────────────
    jwt_secret_key: str = "change-me-jwt-secret-32-chars-min"
    jwt_algorithm: str = "HS256"
    # Short-lived access token; clients silently refresh it with a refresh token.
    jwt_expire_minutes: int = 60  # 1 hour
    # Long-lived refresh token (rotated on every use). Bounds stolen-token blast
    # radius: a leaked access token is only valid for jwt_expire_minutes.
    refresh_token_expire_days: int = 30
    google_client_id: str = ""

    # ── Email / SMTP ─────────────────────────────────────────────────────────
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = "noreply@nl2sql.local"

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_provider: Literal["groq", "openai", "anthropic", "gemini"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0
    groq_api_key: str = Field(default="", description="Required when llm_provider=groq")
    openai_api_key: str = Field(default="", description="Required when llm_provider=openai")
    anthropic_api_key: str = Field(default="", description="Required when llm_provider=anthropic")
    gemini_api_key: str = Field(default="", description="Required when llm_provider=gemini")
    together_api_key: str = Field(default="", description="Required when fine_tuning_provider=together")

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: Literal["huggingface", "gemini"] = "huggingface"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    huggingface_model: str = "all-MiniLM-L6-v2"
    gemini_embedding_model: str = "models/text-embedding-004"
    # Separate key for embeddings — falls back to gemini_api_key if not set
    gemini_embedding_api_key: str = Field(default="", description="Gemini API key for embeddings; falls back to gemini_api_key")

    @property
    def resolved_gemini_embedding_api_key(self) -> str:
        return self.gemini_embedding_api_key or self.gemini_api_key

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_store_provider: Literal["chroma", "faiss", "qdrant"] = "qdrant"
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")
    chroma_collection_name: str = "schema_chunks"
    vector_store_top_k: int = 5
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "schema_chunks"

    # Per-user isolation for the (shared) vector-store collection. When enabled,
    # schema chunks are tagged with `user_id` on write and every read matches the
    # requesting user's own chunks OR shared (un-tagged) chunks, so BYOD users
    # never retrieve each other's tables while everyone still sees the default
    # database's globally-ingested schema. Shared re-ingests (startup / monitor /
    # legacy refresh) delete only the shared chunks, preserving per-user data.
    schema_per_user_isolation: bool = True

    # ── SQL ──────────────────────────────────────────────────────────────────
    sql_dialect: Literal["postgresql", "mysql"] = "postgresql"
    sql_validation_enabled: bool = True
    sql_max_retries: int = 3

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql://user:password@localhost:5432/dbname?sslmode=require"
    history_database_url: str = "postgresql://user:password@localhost:5432/dbname?sslmode=require"

    # A JSON string mapping connection names to connection strings.
    # Example: '{"Local DB": "postgresql+asyncpg://...", "Prod DB": "postgresql+asyncpg://..."}'
    available_databases: str = ""

    @property
    def parsed_available_databases(self) -> dict[str, str]:
        """Parse available_databases JSON string into a dict, fallback to default local db."""
        import json
        default_db = {"Default DB": self.database_url}
        if not self.available_databases:
            return default_db
        try:
            parsed = json.loads(self.available_databases)
            if not isinstance(parsed, dict):
                return default_db
            return parsed
        except Exception:
            return default_db

    # ── Cache ────────────────────────────────────────────────────────────────
    cache_provider: Literal["in_memory", "redis"] = "in_memory"
    cache_ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379/0"

    # Semantic Cache
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    semantic_cache_collection: str = "semantic_cache"

    # ── Connection Pooling ───────────────────────────────────────────────────
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # ── Target DB hardening ──────────────────────────────────────────────────
    # Generated SQL runs against the *target* database in a read-only
    # transaction with a statement timeout (PostgreSQL only).
    target_db_readonly: bool = True
    target_statement_timeout_ms: int = 30_000

    # ── Query Intelligence ───────────────────────────────────────────────────
    query_rewriting_enabled: bool = True
    query_expansion_enabled: bool = True
    parallel_retrieval_enabled: bool = True
    max_context_tokens: int = 8000
    intent_classification_enabled: bool = True

    hybrid_search_alpha: float = 0.5  # weight for vector vs BM25 in hybrid retrieval

    # ── Schema Monitor ───────────────────────────────────────────────────────
    schema_monitor_enabled: bool = True
    schema_monitor_interval_seconds: int = 300

    # ── Background maintenance jobs (account purge + data retention) ──────────
    # Runs the idempotent purge/retention workers on an interval loop inside the
    # app process (advisory-locked so only one worker fires per tick). Disable
    # if you schedule these via an external cron instead.
    background_jobs_enabled: bool = True
    background_jobs_interval_seconds: int = 86_400  # daily

    # ── Auto Ingest on Startup ───────────────────────────────────────────────
    auto_ingest_schema_on_startup: bool = True

    # ── Model Warm-up ────────────────────────────────────────────────────────
    # Load the embedding + reranker models at boot so the first user query
    # doesn't pay the one-time ~1-3 s SentenceTransformer/CrossEncoder cost.
    warm_models_on_startup: bool = True

    # ── Lazy Loading ─────────────────────────────────────────────────────────
    lazy_loading_enabled: bool = True
    chunk_cache_size: int = 100

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_batch_size: int = 32

    # ── Re-ranking ───────────────────────────────────────────────────────────
    reranker_enabled: bool = True
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"  # FlashRank model name (ONNX, no PyTorch)
    reranker_top_k: int = 10

    # ── BM25 Sparse Retrieval ────────────────────────────────────────────────
    bm25_enabled: bool = True  # Feature flag — enable for hybrid retrieval
    bm25_index_path: str = str(PROJECT_ROOT / "data" / "bm25_index.pkl")
    bm25_top_k: int = 5

    # ── Chunker ──────────────────────────────────────────────────────────────
    chunk_strategy: str = "table"  # 'table', 'fixed', 'sentence', or 'parent_child'
    chunk_size: int = 1000  # Max chars per chunk (for 'fixed' and 'sentence')
    chunk_overlap: int = 200  # Overlap between chunks (for 'fixed')

    # ── RAG Quality Upgrades (Phase 3) ────────────────────────────────────────
    # All flags are runtime-adjustable via PUT /api/v1/config/rag and control the
    # live retrieval + ingestion pipeline. Ingest-changing flags (descriptions,
    # parent-child) require a schema re-ingest to take effect on existing data.

    # P1 — LLM-generated natural-language description per table at ingest time,
    # embedded alongside the DDL to close the NL↔DDL vocabulary gap. Adds one LLM
    # call per table during ingestion; default off (opt-in + requires re-ingest).
    rag_schema_descriptions_enabled: bool = False

    # P3 — Multi-query retrieval: run vector search on the original question plus
    # a few synonym expansions and union-dedupe the hits. Cheap recall win; on by default.
    rag_multi_query_enabled: bool = True
    rag_multi_query_max: int = 3  # max number of *extra* query variants (besides the original)

    # P2 — Few-shot example retrieval: index successful NL→SQL pairs in a dedicated
    # vector collection and inject the top-k most similar as concrete examples.
    rag_few_shot_retrieval_enabled: bool = True
    rag_few_shot_top_k: int = 3
    rag_few_shot_collection: str = "query_examples"

    # P4 — Parent-child chunking: index fine-grained column-level child chunks for
    # retrieval precision while returning the full parent table chunk for generation.
    # Default off (opt-in + requires re-ingest).
    rag_parent_child_chunking_enabled: bool = False

    # P5 — HyDE: embed an LLM-generated hypothetical schema description of the
    # answer instead of the raw question. Adds one LLM call per query; default off.
    rag_hyde_enabled: bool = False

    # P7 — Adaptive TOP_K: scale the retrieval top_k by estimated query complexity
    # (joins/aggregations) instead of a fixed value. On by default.
    rag_adaptive_top_k_enabled: bool = True
    rag_adaptive_top_k_min: int = 2
    rag_adaptive_top_k_max: int = 15

    # ── Fine-tuning ──────────────────────────────────────────────────────────
    fine_tuning_enabled: bool = False
    fine_tuning_provider: Literal["openai", "groq", "together"] = "together"

    # ── Observability & Centralized Logs ─────────────────────────────────────
    otel_service_name: str = "nl-to-sql-rag"
    otel_exporter_otlp_endpoint: str = ""  # e.g., http://localhost:4317
    otel_console_exporter: bool = False

    # Langfuse Tracing
    langfuse_enabled: bool = False
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key (sk-lf-...)")
    langfuse_public_key: str = Field(default="", description="Langfuse public key (pk-lf-...)")
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # Arize Phoenix Tracing
    phoenix_active: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60  # per minute

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated allowlist of frontend origins. In production the wildcard
    # is dropped and only these origins are permitted (behind nginx everything
    # is same-origin, so an empty list is safe). In development we allow all.
    cors_allowed_origins: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # ── Admin ─────────────────────────────────────────────────────────────────
    admin_emails: str = Field(default="", description="Comma-separated email addresses granted admin access")

    @property
    def admin_email_list(self) -> list[str]:
        return [e.strip().lower() for e in self.admin_emails.split(",") if e.strip()]

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key(cls, v: str, info: object) -> str:
        """Validate groq key if provider is groq."""
        provider = getattr(info, "data", {}).get("llm_provider")
        if provider == "groq" and not v:
            logger.warning("groq_api_key is empty while llm_provider is groq")
        return v

    @field_validator("anthropic_api_key")
    @classmethod
    def validate_anthropic_key(cls, v: str, info: object) -> str:
        provider = getattr(info, "data", {}).get("llm_provider")
        if provider == "anthropic" and not v:
            logger.warning("anthropic_api_key is empty while llm_provider is anthropic")
        return v

    @field_validator("gemini_api_key")
    @classmethod
    def validate_gemini_key(cls, v: str, info: object) -> str:
        provider = getattr(info, "data", {}).get("llm_provider")
        if provider == "gemini" and not v:
            logger.warning("gemini_api_key is empty while llm_provider is gemini")
        return v

    _WEAK_SECRET_DEFAULTS: frozenset[str] = frozenset({
        "change-me-in-production",
        "change-me-jwt-secret-32-chars-min",
    })

    @model_validator(mode="after")
    def _align_settings(self) -> "Settings":
        """Auto-set embedding dimensions and dynamic production defaults."""
        # Reject placeholder secrets in non-development environments
        if self.app_env != "development":
            if self.secret_key in self._WEAK_SECRET_DEFAULTS:
                raise ValueError(
                    "secret_key must be set to a strong random value in non-development environments. "
                    "Set the SECRET_KEY environment variable."
                )
            if self.jwt_secret_key in self._WEAK_SECRET_DEFAULTS:
                raise ValueError(
                    "jwt_secret_key must be set to a strong random value in non-development environments. "
                    "Set the JWT_SECRET_KEY environment variable."
                )
            # Require at least 20 unique characters for entropy
            if len(set(self.jwt_secret_key)) < 20:
                raise ValueError(
                    "jwt_secret_key has insufficient entropy. Use a randomly generated value of at least 32 characters."
                )

        if self.embedding_provider == "huggingface":
            # P6 — extend this map when swapping in a higher-quality embedding
            # model. Switching the model requires a full re-ingestion so the
            # stored vectors match the new model's dimensionality.
            hf_model_dims = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "BAAI/bge-small-en-v1.5": 384,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-large-en-v1.5": 1024,
                "intfloat/e5-base-v2": 768,
                "intfloat/e5-large-v2": 1024,
                "jinaai/jina-embeddings-v3": 1024,
            }
            if self.huggingface_model in hf_model_dims:
                self.embedding_dimensions = hf_model_dims[self.huggingface_model]
        elif self.embedding_provider == "gemini":
            gemini_model_dims = {
                "models/text-embedding-004": 768,
            }
            if self.gemini_embedding_model in gemini_model_dims:
                self.embedding_dimensions = gemini_model_dims[self.gemini_embedding_model]

        # Switch CACHE_PROVIDER to redis in production by default if not explicitly provided
        if self.app_env == "production" and "cache_provider" not in self.model_fields_set:
            self.cache_provider = "redis"

        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def llm_api_key(self) -> str:
        """Return the active LLM provider's API key."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        return self.groq_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton.

    Using lru_cache means the .env file is read once at startup, not on
    every request.
    """
    return Settings()
