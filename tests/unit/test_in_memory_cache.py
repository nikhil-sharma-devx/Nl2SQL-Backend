"""Unit tests for InMemoryCache — TTL, get/set/delete/clear."""
import asyncio

import pytest

from nl_to_sql.infrastructure.cache.in_memory_cache import InMemoryCache


@pytest.fixture
def cache() -> InMemoryCache:
    return InMemoryCache(default_ttl=60)


class TestGetSet:

    async def test_set_and_get_string(self, cache: InMemoryCache) -> None:
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    async def test_set_and_get_dict(self, cache: InMemoryCache) -> None:
        data = {"sql": "SELECT 1", "is_valid": True}
        await cache.set("key2", data)
        result = await cache.get("key2")
        assert result == data

    async def test_set_and_get_list(self, cache: InMemoryCache) -> None:
        await cache.set("key3", [1, 2, 3])
        result = await cache.get("key3")
        assert result == [1, 2, 3]

    async def test_miss_returns_none(self, cache: InMemoryCache) -> None:
        result = await cache.get("nonexistent_key")
        assert result is None

    async def test_overwrite_returns_new_value(self, cache: InMemoryCache) -> None:
        await cache.set("k", "original")
        await cache.set("k", "updated")
        result = await cache.get("k")
        assert result == "updated"

    async def test_none_value_storable(self, cache: InMemoryCache) -> None:
        await cache.set("null_key", None)
        result = await cache.get("null_key")
        assert result is None  # None stored and None returned — same result as miss


class TestTTL:

    async def test_entry_expires_after_ttl(self) -> None:
        short_cache = InMemoryCache(default_ttl=1)
        await short_cache.set("expiring", "soon")
        assert await short_cache.get("expiring") == "soon"
        await asyncio.sleep(1.05)
        assert await short_cache.get("expiring") is None

    async def test_explicit_ttl_overrides_default(self) -> None:
        cache = InMemoryCache(default_ttl=60)
        await cache.set("short", "value", ttl=1)
        assert await cache.get("short") == "value"
        await asyncio.sleep(1.05)
        assert await cache.get("short") is None

    async def test_zero_ttl_never_expires(self, cache: InMemoryCache) -> None:
        # ttl=0 means no expiry
        await cache.set("permanent", "stays", ttl=0)
        await asyncio.sleep(0.1)
        result = await cache.get("permanent")
        assert result == "stays"

    async def test_long_ttl_entry_survives(self, cache: InMemoryCache) -> None:
        await cache.set("long_lived", "here", ttl=3600)
        result = await cache.get("long_lived")
        assert result == "here"


class TestDelete:

    async def test_delete_removes_key(self, cache: InMemoryCache) -> None:
        await cache.set("to_delete", "value")
        await cache.delete("to_delete")
        assert await cache.get("to_delete") is None

    async def test_delete_nonexistent_key_is_safe(self, cache: InMemoryCache) -> None:
        # Should not raise
        await cache.delete("key_that_does_not_exist")

    async def test_delete_does_not_affect_other_keys(self, cache: InMemoryCache) -> None:
        await cache.set("keep", "this")
        await cache.set("remove", "that")
        await cache.delete("remove")
        assert await cache.get("keep") == "this"
        assert await cache.get("remove") is None


class TestClear:

    async def test_clear_removes_all_entries(self, cache: InMemoryCache) -> None:
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)
        await cache.clear()
        assert await cache.get("a") is None
        assert await cache.get("b") is None
        assert await cache.get("c") is None

    async def test_clear_on_empty_cache_is_safe(self) -> None:
        empty = InMemoryCache(default_ttl=60)
        await empty.clear()  # must not raise


class TestConcurrency:

    async def test_concurrent_writes_do_not_corrupt(self, cache: InMemoryCache) -> None:
        async def write(i: int) -> None:
            await cache.set(f"concurrent_{i}", i)

        await asyncio.gather(*(write(i) for i in range(50)))

        for i in range(50):
            assert await cache.get(f"concurrent_{i}") == i
