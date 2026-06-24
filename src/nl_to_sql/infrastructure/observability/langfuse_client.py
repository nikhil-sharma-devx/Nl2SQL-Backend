"""Langfuse observability client — singleton initialized once at app startup.

The singleton is used by the low-level SDK (manual traces/generations).
The @observe decorator from langfuse.decorators reads LANGFUSE_* env vars
directly, so it works independently of this client, but both share the same
project/keys set during initialization.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_langfuse: Any | None = None


def initialize_langfuse(secret_key: str, public_key: str, host: str) -> bool:
    """Initialize the Langfuse client singleton. Returns True on success."""
    global _langfuse
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
        logger.info("Langfuse tracing initialized (host=%s)", host)
        return True
    except ImportError:
        logger.warning("langfuse package not installed — LLM observability disabled")
        return False
    except Exception as exc:
        logger.warning("Langfuse initialization failed: %s", exc)
        return False


def get_langfuse() -> Any | None:
    """Return the active Langfuse client, or None if not initialized."""
    return _langfuse


def flush_langfuse() -> None:
    """Flush all pending Langfuse events. Call on graceful shutdown."""
    if _langfuse is not None:
        try:
            _langfuse.flush()
            logger.debug("Langfuse events flushed")
        except Exception as exc:
            logger.debug("Langfuse flush failed (non-fatal): %s", exc)
