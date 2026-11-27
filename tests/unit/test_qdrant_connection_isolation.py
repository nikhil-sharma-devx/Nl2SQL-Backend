"""Qdrant connection-isolation tests.

Verifies that schema reads are scoped by ``connection_id`` (own-OR-shared) and
that the own-else-shared dedup prefers the active connection's chunk. Skipped
when ``qdrant_client`` isn't installed (the module imports it at top level).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client.models import IsEmptyCondition, PayloadField

from nl_to_sql.infrastructure.vector_store.qdrant_store import (
    _dedupe_prefer_own,
    _no_hash_filter,
)
from nl_to_sql.infrastructure.vector_store.tenant_scope import (
    own_or_shared_should as _user_scope_should,
)


def _pt(connection_id=None, table="orders", schema="public", is_child=False):
    payload = {"table_name": table, "schema_name": schema}
    if connection_id is not None:
        payload["connection_id"] = connection_id
    if is_child:
        payload["is_child"] = True
    return SimpleNamespace(payload=payload)


def test_scope_should_matches_own_or_shared() -> None:
    """A connection scope must produce two OR-conditions: own + untagged/shared."""
    conds = _user_scope_should("conn-A")
    assert conds is not None
    assert len(conds) == 2  # connection_id == X  OR  connection_id IS EMPTY (shared)


def test_no_scope_returns_shared_only_condition() -> None:
    """C7 fail-closed: no connection_id must mean "shared-only", never "unrestricted".

    Previously this returned ``None``, which — passed as Filter(should=None) —
    applied no restriction at all, so a resolution failure (or any caller
    with connection_id=None) could read every tenant's chunks.
    """
    conds = _user_scope_should(None)
    assert conds is not None
    assert len(conds) == 1
    assert conds[0] == IsEmptyCondition(is_empty=PayloadField(key="connection_id"))


def test_no_hash_filter_scopes_by_connection() -> None:
    f = _no_hash_filter("conn-A", ("schema_hash",))
    # own-OR-shared conditions live on `should`; reserved types excluded via must_not.
    assert f.should is not None
    assert f.must_not is not None


def test_no_hash_filter_unscoped_is_shared_only() -> None:
    """C7: an unscoped caller's filter must still restrict to shared chunks."""
    f = _no_hash_filter(None, ("schema_hash",))
    assert f.should == [IsEmptyCondition(is_empty=PayloadField(key="connection_id"))]


def test_dedup_prefers_own_over_shared() -> None:
    """When both the connection's own and the shared copy match, keep own."""
    shared = _pt(connection_id=None)
    own = _pt(connection_id="conn-A")
    kept = _dedupe_prefer_own([shared, own], "conn-A")
    assert len(kept) == 1
    assert kept[0].payload.get("connection_id") == "conn-A"


def test_dedup_noop_when_unscoped() -> None:
    items = [_pt(connection_id=None), _pt(connection_id="conn-A")]
    assert _dedupe_prefer_own(items, None) == items


def test_dedup_keeps_child_chunks_distinct() -> None:
    """Column-level child chunks share a table_name but must not collapse."""
    parent = _pt(connection_id="conn-A")
    child = _pt(connection_id="conn-A", is_child=True)
    kept = _dedupe_prefer_own([parent, child], "conn-A")
    assert len(kept) == 2


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
