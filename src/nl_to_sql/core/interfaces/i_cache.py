"""ICache — Abstract interface for caching backends."""
from abc import ABC, abstractmethod
from typing import Any


class ICache(ABC):
    """Contract for async caching.

    SOLID: Dependency Inversion — high-level services depend on this
           abstraction, not on Redis or in-memory concretely.
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key. Returns None on miss.

        Args:
            key: Cache key string.

        Returns:
            Cached value or None if not found / expired.
        """
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key string.
            value: Serialisable value to store.
            ttl: Time-to-live in seconds; None uses the default TTL.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a key from the cache.

        Args:
            key: Cache key string.
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Flush all cache entries."""
        ...
