"""Workspaces + connection sharing + access policy, against a real SQLite DB.

The removed Phase-6a team feature failed on exactly these behaviours (members
not linked to real user ids, roles that couldn't be assigned or didn't take
effect), so they're exercised end to end through the real services rather
than mocks: ConnectionService and WorkspaceService share one database.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nl_to_sql.core.exceptions import (
    ConnectionNotFoundError,
    WorkspaceConflictError,
    WorkspaceNotFoundError,
    WorkspacePermissionError,
    WorkspaceValidationError,
)
from nl_to_sql.infrastructure.database.models import User
from nl_to_sql.infrastructure.database.policy_client import PolicyEnforcingClient
from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient
from nl_to_sql.services.connection_service import ConnectionService
from nl_to_sql.services.workspace_service import WorkspaceService

_DSN = "sqlite+aiosqlite:///./_ws_target.db"


async def _noop_test(_url: str) -> None:
    return None


@dataclass
class World:
    conns: ConnectionService
    ws: WorkspaceService


@pytest.fixture
async def world() -> AsyncIterator[World]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite+aiosqlite:///{path}"
    conns = ConnectionService(database_url=url, secret_key="unit-test-secret")
    await conns.initialize()  # create_all → every table, incl. workspaces
    conns._test_client = _noop_test  # type: ignore[method-assign,assignment]
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        for uid, email, verified in (
            ("alice", "Alice@Example.com", True),
            ("bob", "bob@example.com", True),
            ("carol", "carol@example.com", True),
            ("mallory", "mallory@example.com", True),
            ("gina", "gina@gmail.com", True),
            ("gus", "gus@gmail.com", True),
            ("pending", "pending@example.com", False),
        ):
            db.add(User(id=uid, email=email, is_verified=verified))
        await db.commit()
    try:
        yield World(conns, WorkspaceService(factory))
    finally:
        await conns.dispose()
        await engine.dispose()
        try:
            os.remove(path)
        except OSError:
            pass


async def _join(w: World, admin: str, ws_id: str, email: str, user_id: str, role: str = "member") -> None:
    """Invite + accept: the only way into a workspace."""
    inv = await w.ws.invite(admin, ws_id, email, role)
    await w.ws.respond(user_id, inv.id, accept=True)


async def _team(w: World) -> str:
    """alice (admin) + bob (member); returns the workspace id."""
    team = await w.ws.create("alice", "Analytics")
    await _join(w, "alice", team.id, "bob@example.com", "bob")
    return team.id


# ── Membership: ids are linked, roles stick ─────────────────────────────────


async def test_creator_is_admin_and_members_link_to_real_user_ids(world: World) -> None:
    ws_id = await _team(world)
    detail = await world.ws.get("alice", ws_id)
    by_email = {m.email: (m.user_id, m.role) for m in detail.members}
    assert by_email == {"Alice@Example.com": ("alice", "admin"), "bob@example.com": ("bob", "member")}
    assert await world.ws.role_of("bob", ws_id) == "member"


async def test_invite_email_is_case_insensitive(world: World) -> None:
    ws_id = await _team(world)
    inv = await world.ws.invite("alice", ws_id, "  CAROL@example.COM ", "editor")
    assert (inv.invitee_user_id, inv.role, inv.workspace_name) == ("carol", "editor", "Analytics")


# ── Invitations: nothing happens until the invitee accepts ─────────────────


async def test_invitation_grants_nothing_until_accepted(world: World) -> None:
    ws_id = await _team(world)
    cid = await _shared_connection(world, ws_id)
    inv = await world.ws.invite("alice", ws_id, "carol@example.com", "editor")
    assert await world.ws.role_of("carol", ws_id) is None
    assert await world.conns.get_client("carol", cid) is None
    with pytest.raises(WorkspaceNotFoundError):
        await world.ws.get("carol", ws_id)
    # The invitee is notified (this list drives the badge + toast)...
    mine = await world.ws.my_invitations("carol")
    assert [(i.id, i.workspace_name, i.invited_by_email) for i in mine] == [
        (inv.id, "Analytics", "Alice@Example.com")
    ]
    # ...and admins see it as pending.
    assert [i.invitee_email for i in (await world.ws.get("alice", ws_id)).pending_invitations] == [
        "carol@example.com"
    ]


async def test_accepting_joins_with_the_invited_role(world: World) -> None:
    ws_id = await _team(world)
    inv = await world.ws.invite("alice", ws_id, "carol@example.com", "editor")
    assert await world.ws.respond("carol", inv.id, accept=True) == ws_id
    assert await world.ws.role_of("carol", ws_id) == "editor"
    assert await world.ws.my_invitations("carol") == []


async def test_declining_leaves_you_out_and_can_be_reinvited(world: World) -> None:
    ws_id = await _team(world)
    inv = await world.ws.invite("alice", ws_id, "carol@example.com")
    assert await world.ws.respond("carol", inv.id, accept=False) is None
    assert await world.ws.role_of("carol", ws_id) is None
    again = await world.ws.invite("alice", ws_id, "carol@example.com")
    assert again.id != inv.id


async def test_only_the_invitee_can_answer_and_only_once(world: World) -> None:
    ws_id = await _team(world)
    inv = await world.ws.invite("alice", ws_id, "carol@example.com")
    with pytest.raises(WorkspaceNotFoundError):
        await world.ws.respond("mallory", inv.id, accept=True)
    await world.ws.respond("carol", inv.id, accept=True)
    with pytest.raises(WorkspaceNotFoundError):
        await world.ws.respond("carol", inv.id, accept=False)


async def test_duplicate_pending_invitation_conflicts(world: World) -> None:
    ws_id = await _team(world)
    await world.ws.invite("alice", ws_id, "carol@example.com")
    with pytest.raises(WorkspaceConflictError, match="pending"):
        await world.ws.invite("alice", ws_id, "carol@example.com")


async def test_admin_can_cancel_a_pending_invitation(world: World) -> None:
    ws_id = await _team(world)
    inv = await world.ws.invite("alice", ws_id, "carol@example.com")
    with pytest.raises(WorkspacePermissionError):
        await world.ws.cancel_invitation("bob", ws_id, inv.id)
    await world.ws.cancel_invitation("alice", ws_id, inv.id)
    assert await world.ws.my_invitations("carol") == []
    with pytest.raises(WorkspaceNotFoundError):
        await world.ws.respond("carol", inv.id, accept=True)


async def test_unverified_accounts_cannot_be_invited(world: World) -> None:
    ws_id = await _team(world)
    with pytest.raises(WorkspaceValidationError, match="finish signing up"):
        await world.ws.invite("alice", ws_id, "pending@example.com")


# ── Directory: same domain, exact email across domains, admins see all ─────


async def test_directory_is_scoped_to_the_callers_email_domain(world: World) -> None:
    people = await world.ws.directory("alice", "Alice@Example.com")
    emails = {p.email for p in people}
    assert emails == {"bob@example.com", "carol@example.com", "mallory@example.com"}
    gmail = {p.email for p in await world.ws.directory("gina", "gina@gmail.com")}
    assert gmail == {"gus@gmail.com"}  # one "company" per domain; never sees example.com


async def test_directory_excludes_self_and_unverified(world: World) -> None:
    emails = {p.email for p in await world.ws.directory("alice", "alice@example.com")}
    assert "Alice@Example.com" not in emails
    assert "pending@example.com" not in emails


async def test_directory_search_and_exact_cross_domain_match(world: World) -> None:
    assert [p.email for p in await world.ws.directory("alice", "alice@example.com", "car")] == ["carol@example.com"]
    # A partial cross-domain query finds nothing...
    assert await world.ws.directory("alice", "alice@example.com", "gina") == []
    # ...but the exact address reaches that one account.
    exact = await world.ws.directory("alice", "alice@example.com", "Gina@Gmail.com")
    assert [p.email for p in exact] == ["gina@gmail.com"]


async def test_directory_search_treats_wildcards_literally(world: World) -> None:
    assert await world.ws.directory("alice", "alice@example.com", "%") == []


async def test_admin_directory_sees_every_domain(world: World) -> None:
    emails = {p.email for p in await world.ws.directory("alice", "alice@example.com", is_admin=True)}
    assert {"gina@gmail.com", "gus@gmail.com", "bob@example.com"} <= emails


async def test_directory_marks_members_and_pending_invites(world: World) -> None:
    ws_id = await _team(world)
    await world.ws.invite("alice", ws_id, "carol@example.com")
    status = {p.email: p.status for p in await world.ws.directory("alice", "alice@example.com", workspace_id=ws_id)}
    assert status["bob@example.com"] == "member"
    assert status["carol@example.com"] == "invited"
    assert status["mallory@example.com"] is None


async def test_unknown_email_is_rejected_with_actionable_message(world: World) -> None:
    ws_id = await _team(world)
    with pytest.raises(WorkspaceValidationError, match="finish signing up"):
        await world.ws.invite("alice", ws_id, "nobody@example.com")


async def test_duplicate_member_conflicts(world: World) -> None:
    ws_id = await _team(world)
    with pytest.raises(WorkspaceConflictError, match="already in this workspace"):
        await world.ws.invite("alice", ws_id, "bob@example.com")


async def test_role_change_takes_effect(world: World) -> None:
    ws_id = await _team(world)
    await world.ws.change_role("alice", ws_id, "bob", "admin")
    assert await world.ws.role_of("bob", ws_id) == "admin"
    # Bob can now do admin things.
    await world.ws.invite("bob", ws_id, "carol@example.com")


async def test_invalid_role_rejected(world: World) -> None:
    ws_id = await _team(world)
    with pytest.raises(WorkspaceValidationError):
        await world.ws.change_role("alice", ws_id, "bob", "owner")


# ── Permissions ──────────────────────────────────────────────────────────────


async def test_member_cannot_manage_members(world: World) -> None:
    ws_id = await _team(world)
    with pytest.raises(WorkspacePermissionError):
        await world.ws.invite("bob", ws_id, "carol@example.com")


async def test_non_member_gets_not_found_not_forbidden(world: World) -> None:
    ws_id = await _team(world)
    with pytest.raises(WorkspaceNotFoundError):
        await world.ws.get("mallory", ws_id)
    with pytest.raises(WorkspaceNotFoundError):
        await world.ws.invite("mallory", ws_id, "carol@example.com")


async def test_last_admin_cannot_be_demoted_or_removed(world: World) -> None:
    ws_id = await _team(world)
    with pytest.raises(WorkspaceValidationError, match="at least one admin"):
        await world.ws.change_role("alice", ws_id, "alice", "member")
    with pytest.raises(WorkspaceValidationError, match="last admin"):
        await world.ws.remove_member("alice", ws_id, "alice")


async def test_member_can_leave_but_not_remove_others(world: World) -> None:
    ws_id = await _team(world)
    await _join(world, "alice", ws_id, "carol@example.com", "carol")
    with pytest.raises(WorkspacePermissionError):
        await world.ws.remove_member("bob", ws_id, "carol")
    await world.ws.remove_member("bob", ws_id, "bob")
    assert await world.ws.role_of("bob", ws_id) is None


async def test_list_only_shows_own_workspaces(world: World) -> None:
    await _team(world)
    assert [w.name for w in await world.ws.list_for_user("bob")] == ["Analytics"]
    assert await world.ws.list_for_user("mallory") == []


# ── Connection sharing ───────────────────────────────────────────────────────


async def _shared_connection(w: World, ws_id: str) -> str:
    info = await w.conns.create("alice", "Warehouse", _DSN)
    await w.ws.set_connection_workspace("alice", info.connection_id, ws_id)
    return info.connection_id


async def test_member_sees_shared_connection_without_dsn_preview(world: World) -> None:
    ws_id = await _team(world)
    cid = await _shared_connection(world, ws_id)
    shared = [c for c in await world.conns.list_connections("bob") if c.connection_id == cid]
    assert len(shared) == 1
    info = shared[0]
    assert (info.access, info.workspace_name, info.owner_email) == ("member", "Analytics", "Alice@Example.com")
    assert info.url_preview is None


async def test_non_member_cannot_see_or_use_shared_connection(world: World) -> None:
    ws_id = await _team(world)
    cid = await _shared_connection(world, ws_id)
    assert cid not in {c.connection_id for c in await world.conns.list_connections("mallory")}
    assert await world.conns.get_client("mallory", cid) is None
    with pytest.raises(ConnectionNotFoundError):
        await world.conns.set_default("mallory", cid)


async def test_member_can_select_shared_connection_and_falls_back_when_removed(world: World) -> None:
    ws_id = await _team(world)
    cid = await _shared_connection(world, ws_id)
    await world.conns.set_default("bob", cid)
    assert await world.conns.get_active_connection_id("bob") == cid
    assert await world.conns.get_client("bob", cid) is not None

    await world.ws.remove_member("alice", ws_id, "bob")
    fallback = await world.conns.get_active_connection_id("bob")
    assert fallback != cid  # revoked share → own default, pointer cleared
    assert await world.conns.get_client("bob", cid) is None


async def test_selecting_own_connection_clears_shared_pointer(world: World) -> None:
    ws_id = await _team(world)
    cid = await _shared_connection(world, ws_id)
    await world.conns.set_default("bob", cid)
    own = (await world.conns.list_connections("bob"))[0]
    assert own.access == "owner"
    await world.conns.set_default("bob", own.connection_id)
    assert await world.conns.get_active_connection_id("bob") == own.connection_id


async def test_only_admins_can_share_and_server_default_cannot_be_shared(world: World) -> None:
    ws_id = await _team(world)
    bobs = await world.conns.create("bob", "Bob DB", _DSN)
    with pytest.raises(WorkspacePermissionError):
        await world.ws.set_connection_workspace("bob", bobs.connection_id, ws_id)
    server_default = next(c for c in await world.conns.list_connections("alice") if not c.has_dsn)
    with pytest.raises(WorkspaceValidationError, match="Server Default"):
        await world.ws.set_connection_workspace("alice", server_default.connection_id, ws_id)


async def test_member_cannot_share_someone_elses_connection(world: World) -> None:
    ws_id = await _team(world)
    cid = await _shared_connection(world, ws_id)
    await world.ws.change_role("alice", ws_id, "bob", "admin")
    with pytest.raises(ConnectionNotFoundError):
        await world.ws.set_connection_workspace("bob", cid, None)


async def test_deleting_workspace_unshares_connections(world: World) -> None:
    ws_id = await _team(world)
    cid = await _shared_connection(world, ws_id)
    await world.ws.delete("alice", ws_id)
    assert await world.conns.get_client("bob", cid) is None
    owned = next(c for c in await world.conns.list_connections("alice") if c.connection_id == cid)
    assert owned.workspace_id is None


async def test_removed_admins_own_connections_leave_with_them(world: World) -> None:
    ws_id = await _team(world)
    await world.ws.change_role("alice", ws_id, "bob", "admin")
    bobs = await world.conns.create("bob", "Bob DB", _DSN)
    await world.ws.set_connection_workspace("bob", bobs.connection_id, ws_id)
    await world.ws.remove_member("alice", ws_id, "bob")
    detail = await world.ws.get("alice", ws_id)
    assert bobs.connection_id not in {c.connection_id for c in detail.connections}


# ── Access policy ────────────────────────────────────────────────────────────


async def test_policy_is_owner_only_and_wraps_every_client(world: World) -> None:
    ws_id = await _team(world)
    cid = await _shared_connection(world, ws_id)
    with pytest.raises(ConnectionNotFoundError):
        await world.conns.update_policy("bob", cid, ["orders"], [])

    await world.conns.update_policy("alice", cid, ["orders", "customers"], ["customers.email"])
    fallback = AsyncDatabaseClient.__new__(AsyncDatabaseClient)  # never used for SQL here
    for user in ("alice", "bob"):
        client = await world.conns.get_scoped_client(user, cid, fallback)
        assert isinstance(client, PolicyEnforcingClient)
        assert client.policy.violations("SELECT email FROM customers")


async def test_empty_allow_list_and_bad_column_format_rejected(world: World) -> None:
    info = await world.conns.create("alice", "Warehouse", _DSN)
    from nl_to_sql.core.exceptions import ConnectionValidationError

    with pytest.raises(ConnectionValidationError):
        await world.conns.update_policy("alice", info.connection_id, [], [])
    with pytest.raises(ConnectionValidationError):
        await world.conns.update_policy("alice", info.connection_id, None, ["no_dot"])


async def test_no_policy_returns_unwrapped_client(world: World) -> None:
    info = await world.conns.create("alice", "Warehouse", _DSN)
    fallback = AsyncDatabaseClient.__new__(AsyncDatabaseClient)
    client = await world.conns.get_scoped_client("alice", info.connection_id, fallback)
    assert not isinstance(client, PolicyEnforcingClient)
