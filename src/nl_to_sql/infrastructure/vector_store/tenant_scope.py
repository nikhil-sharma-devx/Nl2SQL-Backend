"""Shared "own-or-shared" Qdrant visibility filter.

Both the schema store (``qdrant_store.py``) and the few-shot example store
(``example_store.py``) need the same tenant-isolation rule: a point is
visible if it is tagged with the caller's id, or untagged (shared). This is
centralized here so the fail-closed invariant — no id means shared-only,
*never* unrestricted — is enforced once instead of hand-copied per store.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, IsEmptyCondition, MatchValue, PayloadField


def own_or_shared_should(scope_id: str | None, field: str = "connection_id") -> list[Any]:
    """OR-conditions scoping reads to the caller's own points plus shared ones.

    A point matches if it is tagged with ``scope_id`` (``field == scope_id``)
    **or** is un-tagged/shared (the ``field`` payload key is missing).

    With no ``scope_id`` (fail-closed) the returned condition list has exactly
    one entry — "untagged/shared only" — so an unscoped caller can never see
    another tenant's points. This must never be empty: an empty ``should``
    list applies no restriction at all, which is what let a resolution
    failure silently degrade into an unrestricted cross-tenant read.
    """
    shared_only = IsEmptyCondition(is_empty=PayloadField(key=field))
    if scope_id is None:
        return [shared_only]
    return [
        FieldCondition(key=field, match=MatchValue(value=scope_id)),
        shared_only,
    ]
