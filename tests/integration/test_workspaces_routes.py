"""Integration tests for workspace routes and the new connection policy/share routes.

Services are mocked (their logic is covered against a real DB in
tests/unit/test_workspace_sharing.py); these tests pin the HTTP contract:
auth is required, domain errors map to the canonical envelope with the
documented status codes, and request validation rejects bad roles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from nl_to_sql.core.exceptions import (
    AccessPolicyViolationError,
    WorkspaceConflictError,
    WorkspaceNotFoundError,
    WorkspacePermissionError,
    WorkspaceValidationError,
)
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.services.connection_service import ConnectionInfo
from nl_to_sql.services.workspace_service import DirectoryUser, InvitationInfo, WorkspaceSummary

USER = UserPublic(
    id="ws-user-1",
    email="admin@example.com",
    full_name="Admin",
    auth_provider="email",
    is_verified=True,
    created_at=datetime(2026, 1, 1),
)
NOW = datetime(2026, 11, 26)


def _app(*, authed: bool = True, ws: Any = None, conns: Any = None) -> Any:
    with patch("nl_to_sql.infrastructure.llm.groq_provider.AsyncGroq"):
        from nl_to_sql.api.app import create_app
        from nl_to_sql.api.dependencies import (
            get_connection_service,
            get_current_user,
            get_workspace_service,
        )

        app = create_app()
        if authed:
            async def _user() -> UserPublic:
                return USER

            app.dependency_overrides[get_current_user] = _user
        if ws is not None:
            app.dependency_overrides[get_workspace_service] = lambda: ws
        if conns is not None:
            app.dependency_overrides[get_connection_service] = lambda: conns
        return app


async def _call(app: Any, method: str, url: str, **kw: Any) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, url, **kw)


def _ws_svc() -> MagicMock:
    svc = MagicMock()
    svc.list_for_user = AsyncMock(return_value=[WorkspaceSummary("w1", "Analytics", "admin", 2, 1, NOW)])
    svc.create = AsyncMock(return_value=WorkspaceSummary("w2", "New", "admin", 1, 0, NOW))
    svc.invite = AsyncMock(
        return_value=InvitationInfo(
            "inv-1", "w1", "Analytics", "member", "u2", "bob@example.com", None, "admin@example.com", NOW
        )
    )
    svc.my_invitations = AsyncMock(return_value=[])
    svc.respond = AsyncMock(return_value="w1")
    svc.directory = AsyncMock(return_value=[DirectoryUser("u2", "bob@example.com", "Bob", "invited")])
    return svc


async def test_requires_auth() -> None:
    resp = await _call(_app(authed=False, ws=_ws_svc()), "GET", "/api/v1/workspaces")
    assert resp.status_code == 401


async def test_list_and_create() -> None:
    svc = _ws_svc()
    app = _app(ws=svc)
    listed = await _call(app, "GET", "/api/v1/workspaces")
    assert listed.status_code == 200
    assert listed.json()[0]["role"] == "admin"
    created = await _call(app, "POST", "/api/v1/workspaces", json={"name": "New"})
    assert created.status_code == 201
    svc.create.assert_awaited_once_with(USER.id, "New")


async def test_invite_rejects_unknown_role_before_service() -> None:
    svc = _ws_svc()
    resp = await _call(
        _app(ws=svc), "POST", "/api/v1/workspaces/w1/invitations", json={"email": "b@x.com", "role": "owner"}
    )
    assert resp.status_code == 422
    svc.invite.assert_not_awaited()


async def test_invite_creates_pending_invitation() -> None:
    svc = _ws_svc()
    resp = await _call(
        _app(ws=svc), "POST", "/api/v1/workspaces/w1/invitations", json={"email": "bob@example.com", "role": "member"}
    )
    assert resp.status_code == 201
    assert resp.json()["invitee_email"] == "bob@example.com"
    svc.invite.assert_awaited_once_with(USER.id, "w1", "bob@example.com", "member")


async def test_fixed_paths_are_not_read_as_workspace_ids() -> None:
    svc = _ws_svc()
    app = _app(ws=svc)
    assert (await _call(app, "GET", "/api/v1/workspaces/invitations")).status_code == 200
    resp = await _call(app, "GET", "/api/v1/workspaces/directory", params={"q": "bob", "workspace_id": "w1"})
    assert resp.status_code == 200
    assert resp.json()[0]["status"] == "invited"
    svc.directory.assert_awaited_once_with(USER.id, USER.email, "bob", is_admin=False, workspace_id="w1")


async def test_decline_calls_respond_with_false() -> None:
    svc = _ws_svc()
    resp = await _call(_app(ws=svc), "POST", "/api/v1/workspaces/invitations/inv-1/decline")
    assert resp.status_code == 204
    svc.respond.assert_awaited_once_with(USER.id, "inv-1", accept=False)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/admin/users"),
        ("GET", "/api/v1/admin/users/u2"),
        ("PATCH", "/api/v1/admin/users/u2"),
        ("GET", "/api/v1/admin/users/u2/chats/s1"),
        ("DELETE", "/api/v1/admin/users/u2/chats/s1"),
        ("DELETE", "/api/v1/admin/users/u2/connections/c1"),
        ("DELETE", "/api/v1/admin/workspaces/w1"),
    ],
)
async def test_admin_routes_refuse_non_admins(method: str, path: str) -> None:
    resp = await _call(_app(), method, path, json={"is_active": False} if method == "PATCH" else None)
    assert resp.status_code == 403


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (WorkspaceNotFoundError("Workspace not found."), 404, "WORKSPACE_NOT_FOUND"),
        (WorkspacePermissionError("needs admin"), 403, "WORKSPACE_FORBIDDEN"),
        (WorkspaceValidationError("No Vectrix account uses that email."), 422, "WORKSPACE_INVALID"),
        (WorkspaceConflictError("already in this workspace"), 409, "WORKSPACE_CONFLICT"),
    ],
)
async def test_domain_errors_use_canonical_envelope(exc: Exception, status: int, code: str) -> None:
    svc = _ws_svc()
    svc.invite = AsyncMock(side_effect=exc)
    resp = await _call(
        _app(ws=svc), "POST", "/api/v1/workspaces/w1/invitations", json={"email": "b@x.com", "role": "member"}
    )
    assert resp.status_code == status
    body = resp.json()
    assert body["code"] == code
    assert body["message"] == str(exc)


async def test_policy_route_calls_owner_only_service() -> None:
    conns = MagicMock()
    conns.update_policy = AsyncMock(
        return_value=ConnectionInfo(
            connection_id="c1",
            name="Warehouse",
            db_type="postgresql",
            is_default=False,
            has_dsn=True,
            url_preview=None,
            created_at=NOW,
            updated_at=NOW,
            allowed_tables=["orders"],
            restricted_columns=["customers.email"],
        )
    )
    resp = await _call(
        _app(conns=conns),
        "PUT",
        "/api/v1/connections/c1/policy",
        json={"allowed_tables": ["orders"], "restricted_columns": ["customers.email"]},
    )
    assert resp.status_code == 200
    assert resp.json()["restricted_columns"] == ["customers.email"]
    conns.update_policy.assert_awaited_once_with(USER.id, "c1", ["orders"], ["customers.email"])


async def test_policy_violation_maps_to_403() -> None:
    from nl_to_sql.api.middleware.error_handler import _ERROR_CODE_MAP, _ERROR_STATUS_MAP

    err = AccessPolicyViolationError("Column `customers.email` is restricted on this connection.")
    status = next(code for t, code in _ERROR_STATUS_MAP.items() if isinstance(err, t))
    code = next(c for t, c in _ERROR_CODE_MAP.items() if isinstance(err, t))
    assert (status, code) == (403, "ACCESS_POLICY_VIOLATION")
