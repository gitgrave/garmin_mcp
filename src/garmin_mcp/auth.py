"""Runtime authentication MCP tools.

Lets a client log in to Garmin Connect at runtime — the credentials are used
only to mint OAuth tokens (written to the token store, e.g. ~/.garminconnect);
they are never stored. Garmin has no app-passwords / third-party OAuth, so the
account password is required. Pass credentials over a trusted transport only.

Supports MFA as a two-step flow: garmin_login may return status "mfa_required",
after which garmin_login_mfa completes it. On success, all tools are repointed
at the live session without restarting the server.
"""
import base64
import json
import os

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from garmin_mcp import token_utils
from garmin_mcp.auth_cli import _secure_token_dir

# --- configured by main() ---------------------------------------------------
_tokenstore = None
_tokenstore_base64 = None
_is_cn = False
_apply_authenticated = None    # callable(garmin) -> repoint every module at the live client
_apply_unauthenticated = None  # callable() -> repoint every module at the sentinel

# --- runtime state ----------------------------------------------------------
_client = None    # live authenticated Garmin instance, or None
_pending = None   # (Garmin, client_state) awaiting an MFA code


def configure_auth(
    *,
    tokenstore,
    tokenstore_base64,
    is_cn,
    apply_authenticated,
    apply_unauthenticated,
    initial_client=None,
):
    """Wire the auth module to the server's token store and client-swap callbacks."""
    global _tokenstore, _tokenstore_base64, _is_cn
    global _apply_authenticated, _apply_unauthenticated, _client
    _tokenstore = tokenstore
    _tokenstore_base64 = tokenstore_base64
    _is_cn = is_cn
    _apply_authenticated = apply_authenticated
    _apply_unauthenticated = apply_unauthenticated
    _client = initial_client


def _persist_and_activate(garmin) -> None:
    """Write tokens to disk (secured), then repoint all tools at the live client."""
    global _client, _pending
    garmin.client.dump(_tokenstore)
    _secure_token_dir(os.path.expanduser(_tokenstore))
    # Mirror the base64 token file, matching the CLI / startup behavior.
    try:
        token_json = os.path.join(os.path.expanduser(_tokenstore), "garmin_tokens.json")
        with open(token_json) as fh:
            data = fh.read()
        with open(os.path.expanduser(_tokenstore_base64), "w") as fh:
            fh.write(base64.b64encode(data.encode()).decode())
    except Exception:
        pass  # base64 mirror is best-effort; the directory store is authoritative
    _client = garmin
    _pending = None
    _apply_authenticated(garmin)


def _account_name(garmin):
    try:
        return garmin.get_full_name()
    except Exception:
        return None


def register_tools(app):
    """Register the runtime auth tools with the MCP app."""

    @app.tool()
    async def garmin_login(email: str, password: str) -> str:
        """Log in to Garmin Connect and persist OAuth tokens for the server to use.

        The password is used only to obtain tokens (written to the token store);
        it is not stored. If Garmin requires MFA, this returns
        {"status": "mfa_required"} — then call garmin_login_mfa with the code
        Garmin sent (email/SMS). On success every tool starts using the new
        session immediately (no restart).
        """
        global _pending
        try:
            garmin = Garmin(
                email=email,
                password=password,
                is_cn=_is_cn,
                prompt_mfa=lambda: "",
                return_on_mfa=True,
            )
            result1, result2 = garmin.login()
            if result1 == "needs_mfa":
                _pending = (garmin, result2)
                return json.dumps({
                    "status": "mfa_required",
                    "message": "Garmin sent an MFA code. Call garmin_login_mfa with it.",
                })
            _persist_and_activate(garmin)
            return json.dumps({
                "status": "ok",
                "authenticated": True,
                "account": _account_name(garmin),
            })
        except (
            GarminConnectAuthenticationError,
            GarminConnectConnectionError,
            GarminConnectTooManyRequestsError,
        ) as exc:
            return json.dumps({"status": "error", "message": str(exc)})
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)})
        finally:
            password = None  # noqa: F841 — best-effort scrub of the local reference

    @app.tool()
    async def garmin_login_mfa(mfa_code: str) -> str:
        """Complete a login that returned "mfa_required" by supplying the MFA code."""
        global _pending
        if not _pending:
            return json.dumps({
                "status": "error",
                "message": "No login awaiting MFA. Call garmin_login first.",
            })
        garmin, state = _pending
        try:
            garmin.resume_login(state, mfa_code)
            _persist_and_activate(garmin)
            return json.dumps({
                "status": "ok",
                "authenticated": True,
                "account": _account_name(garmin),
            })
        except Exception as exc:
            _pending = None
            return json.dumps({"status": "error", "message": str(exc)})

    @app.tool()
    async def garmin_auth_status() -> str:
        """Report whether the server currently holds a valid Garmin session."""
        if _client is None:
            return json.dumps({"authenticated": False})
        name = _account_name(_client)
        return json.dumps({"authenticated": name is not None, "account": name})

    @app.tool()
    async def garmin_logout() -> str:
        """Log out: delete the stored tokens and drop the active session."""
        global _client, _pending
        try:
            token_utils.remove_tokens()
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)})
        _client = None
        _pending = None
        if _apply_unauthenticated:
            _apply_unauthenticated()
        return json.dumps({"status": "ok", "authenticated": False})

    return app
