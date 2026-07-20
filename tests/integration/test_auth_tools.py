"""Integration tests for the runtime auth tools (garmin_login etc.)."""
import json

import pytest
from unittest.mock import Mock, patch
from mcp.server.fastmcp import FastMCP

from garmin_mcp import auth


class FakeGarmin:
    """Stand-in for garminconnect.Garmin driven by class-level knobs."""

    needs_mfa = False
    login_error = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.client = Mock()  # .dump(path) is a no-op mock
        self.resumed = None

    def login(self):
        if FakeGarmin.login_error:
            raise FakeGarmin.login_error
        if FakeGarmin.needs_mfa:
            return ("needs_mfa", {"state": 1})
        return (None, None)

    def resume_login(self, state, code):
        self.resumed = (state, code)
        return (None, None)

    def get_full_name(self):
        return "Test User"


@pytest.fixture
def auth_app():
    """FastMCP app with auth tools registered and callbacks captured."""
    FakeGarmin.needs_mfa = False
    FakeGarmin.login_error = None
    calls = {"authenticated": [], "unauthenticated": 0}

    def apply_auth(garmin):
        calls["authenticated"].append(garmin)

    def apply_unauth():
        calls["unauthenticated"] += 1

    auth.configure_auth(
        tokenstore="/tmp/nonexistent-garmin-tokens",
        tokenstore_base64="/tmp/nonexistent-garmin-tokens_b64",
        is_cn=False,
        apply_authenticated=apply_auth,
        apply_unauthenticated=apply_unauth,
        initial_client=None,
    )
    app = FastMCP("Test Auth")
    app = auth.register_tools(app)
    return app, calls


async def _call(app, name, args=None):
    result = await app.call_tool(name, args or {})
    return json.loads(result[0][0].text)


@pytest.mark.asyncio
async def test_login_success_no_mfa(auth_app):
    app, calls = auth_app
    with patch.object(auth, "Garmin", FakeGarmin), \
         patch.object(auth, "_secure_token_dir"):
        data = await _call(app, "garmin_login", {"email": "a@b.c", "password": "pw"})

    assert data["status"] == "ok"
    assert data["authenticated"] is True
    assert data["account"] == "Test User"
    assert len(calls["authenticated"]) == 1  # tools repointed at live client


@pytest.mark.asyncio
async def test_login_mfa_two_phase(auth_app):
    app, calls = auth_app
    FakeGarmin.needs_mfa = True
    with patch.object(auth, "Garmin", FakeGarmin), \
         patch.object(auth, "_secure_token_dir"):
        phase1 = await _call(app, "garmin_login", {"email": "a@b.c", "password": "pw"})
        assert phase1["status"] == "mfa_required"
        assert calls["authenticated"] == []  # not activated yet

        phase2 = await _call(app, "garmin_login_mfa", {"mfa_code": "123456"})

    assert phase2["status"] == "ok"
    assert phase2["authenticated"] is True
    assert len(calls["authenticated"]) == 1


@pytest.mark.asyncio
async def test_login_mfa_without_pending_errors(auth_app):
    app, _ = auth_app
    data = await _call(app, "garmin_login_mfa", {"mfa_code": "000000"})
    assert data["status"] == "error"
    assert "garmin_login first" in data["message"]


@pytest.mark.asyncio
async def test_login_failure_surfaces_error(auth_app):
    app, calls = auth_app
    from garminconnect import GarminConnectAuthenticationError
    FakeGarmin.login_error = GarminConnectAuthenticationError("bad creds")
    with patch.object(auth, "Garmin", FakeGarmin):
        data = await _call(app, "garmin_login", {"email": "a@b.c", "password": "pw"})
    assert data["status"] == "error"
    assert "bad creds" in data["message"]
    assert calls["authenticated"] == []


@pytest.mark.asyncio
async def test_auth_status_unauthenticated(auth_app):
    app, _ = auth_app  # initial_client=None
    data = await _call(app, "garmin_auth_status")
    assert data["authenticated"] is False


@pytest.mark.asyncio
async def test_auth_status_authenticated_after_login(auth_app):
    app, _ = auth_app
    with patch.object(auth, "Garmin", FakeGarmin), \
         patch.object(auth, "_secure_token_dir"):
        await _call(app, "garmin_login", {"email": "a@b.c", "password": "pw"})
        data = await _call(app, "garmin_auth_status")
    assert data["authenticated"] is True
    assert data["account"] == "Test User"


@pytest.mark.asyncio
async def test_logout_removes_tokens_and_repoints(auth_app):
    app, calls = auth_app
    with patch.object(auth, "Garmin", FakeGarmin), \
         patch.object(auth, "_secure_token_dir"):
        await _call(app, "garmin_login", {"email": "a@b.c", "password": "pw"})

    with patch.object(auth.token_utils, "remove_tokens") as rm:
        data = await _call(app, "garmin_logout")

    rm.assert_called_once()
    assert data["authenticated"] is False
    assert calls["unauthenticated"] == 1  # tools repointed at sentinel

    status = await _call(app, "garmin_auth_status")
    assert status["authenticated"] is False
