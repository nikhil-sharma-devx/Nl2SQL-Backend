"""Unit tests for api.dependencies.resolve_active_connection fail-closed behavior (C7).

Regression test: any exception during connection resolution — not just a
missing/invalid token — used to be swallowed identically, returning
``(None, None, db_client)``. That's still the returned shape (every
downstream per-tenant filter now treats ``None`` as shared-only, not
unrestricted — see test_qdrant_connection_isolation.py /
test_semantic_cache_isolation.py / test_training_data_service.py), but the
two distinct failure modes (no/invalid credentials vs. an authenticated
user's connection lookup failing) must both be exercised and the latter must
not be confused with a successful resolution.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nl_to_sql.api.dependencies import resolve_active_connection


class _FakeCredentials:
    def __init__(self, token: str = "fake-token") -> None:
        self.credentials = token


@pytest.fixture
def fake_container():
    container = MagicMock()
    container.db_client.return_value = "default-db-client"
    return container


class TestResolveActiveConnection:
    async def test_no_credentials_returns_none_scope(self, fake_container) -> None:
        with patch("nl_to_sql.api.dependencies._get_container", return_value=fake_container):
            user_id, connection_id, db_client = await resolve_active_connection(None)
        assert (user_id, connection_id) == (None, None)
        assert db_client == "default-db-client"

    async def test_invalid_token_fails_closed(self, fake_container) -> None:
        with (
            patch("nl_to_sql.api.dependencies._get_container", return_value=fake_container),
            patch(
                "nl_to_sql.services.auth_service.decode_access_token",
                side_effect=Exception("bad token"),
            ),
        ):
            user_id, connection_id, db_client = await resolve_active_connection(
                _FakeCredentials()
            )
        assert (user_id, connection_id) == (None, None)
        assert db_client == "default-db-client"

    async def test_authenticated_but_connection_lookup_fails_closed(self, fake_container) -> None:
        """The exact C7 bug: a valid token but a transient connection-service
        error must fail closed, not silently return as if resolved."""
        token_data = MagicMock(user_id="user-123")
        conn_svc = MagicMock()
        conn_svc.get_active_connection_id = AsyncMock(side_effect=RuntimeError("db hiccup"))
        fake_container.connection_service.return_value = conn_svc

        with (
            patch("nl_to_sql.api.dependencies._get_container", return_value=fake_container),
            patch(
                "nl_to_sql.services.auth_service.decode_access_token",
                return_value=token_data,
            ),
        ):
            user_id, connection_id, db_client = await resolve_active_connection(
                _FakeCredentials()
            )
        assert (user_id, connection_id) == (None, None)
        assert db_client == "default-db-client"

    async def test_successful_resolution_returns_real_scope(self, fake_container) -> None:
        token_data = MagicMock(user_id="user-123")
        conn_svc = MagicMock()
        conn_svc.get_active_connection_id = AsyncMock(return_value="conn-abc")
        # Resolution goes through get_scoped_client (owned/shared + access policy).
        conn_svc.get_scoped_client = AsyncMock(return_value="user-db-client")
        fake_container.connection_service.return_value = conn_svc

        with (
            patch("nl_to_sql.api.dependencies._get_container", return_value=fake_container),
            patch(
                "nl_to_sql.services.auth_service.decode_access_token",
                return_value=token_data,
            ),
        ):
            user_id, connection_id, db_client = await resolve_active_connection(
                _FakeCredentials()
            )
        assert user_id == "user-123"
        assert connection_id == "conn-abc"
        assert db_client == "user-db-client"
