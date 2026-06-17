"""Alembic environment — async SQLAlchemy with application settings."""
from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Make src importable when running alembic from the backend/ root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nl_to_sql.infrastructure.database.models import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _raw_url() -> str:
    """Return the raw DB URL from env, settings, or alembic.ini."""
    url = os.environ.get("DATABASE_URL") or os.environ.get("HISTORY_DATABASE_URL")
    if url:
        return url
    try:
        from nl_to_sql.config.settings import get_settings
        return get_settings().history_database_url
    except Exception:
        return config.get_main_option("sqlalchemy.url")


def _get_url() -> str:
    """Sync URL for offline migrations (SQL script generation)."""
    import re
    url = _raw_url()
    url = url.replace("+asyncpg", "").replace("+aiosqlite", "")
    url = url.replace("postgresql://", "postgresql://", 1)  # keep as-is for psycopg2
    url = re.sub(r"[?&]channel_binding=[^&]*", "", url)
    url = re.sub(r"[?&]$", "", url)
    return url


def _get_async_url() -> str:
    """Async URL for online migrations (live connection via asyncpg)."""
    import re
    url = _raw_url()
    # Ensure asyncpg driver
    url = url.replace("postgresql+asyncpg://", "postgresql+asyncpg://", 1)
    if "postgresql+asyncpg://" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    # asyncpg uses ssl=require, not sslmode=require
    url = url.replace("sslmode=", "ssl=")
    # strip channel_binding — not supported by asyncpg
    url = re.sub(r"[?&]channel_binding=[^&]*", "", url)
    url = re.sub(r"[?&]$", "", url)
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations against a live async engine."""
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _get_async_url()
    connectable = async_engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
