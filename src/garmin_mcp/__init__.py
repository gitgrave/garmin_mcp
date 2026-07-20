"""
Modular MCP Server for Garmin Connect Data
"""

import os
import re
import sys
import base64

import requests
from mcp.server.fastmcp import FastMCP

from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectConnectionError, GarminConnectNotFoundError, GarminConnectTooManyRequestsError

# Import all modules
from garmin_mcp import token_utils
from garmin_mcp import activity_management
from garmin_mcp import health_wellness
from garmin_mcp import user_profile
from garmin_mcp import devices
from garmin_mcp import gear_management
from garmin_mcp import weight_management
from garmin_mcp import challenges
from garmin_mcp import training
from garmin_mcp import workouts
from garmin_mcp import workout_templates
from garmin_mcp import data_management
from garmin_mcp import womens_health
from garmin_mcp import nutrition
from garmin_mcp import workout_builders
from garmin_mcp import courses
from garmin_mcp import activity_analysis
from garmin_mcp import auth


def is_interactive_terminal() -> bool:
    """Detect if running in interactive terminal vs MCP subprocess.

    Returns:
        bool: True if running in an interactive terminal, False otherwise
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def get_mfa() -> str:
    """Get MFA code from user input.

    Raises:
        RuntimeError: If running in non-interactive environment
    """
    if not is_interactive_terminal():
        print(
            "\nERROR: MFA code required but no interactive terminal available.\n"
            "Please run 'garmin-mcp-auth' in your terminal first.\n"
            "See: https://github.com/Taxuspt/garmin_mcp#mfa-setup\n",
            file=sys.stderr,
        )
        raise RuntimeError("MFA required but non-interactive environment")

    print(
        "\nGarmin Connect MFA required. Please check your email/phone for the code.",
        file=sys.stderr,
    )
    return input("Enter MFA code: ")


# Get credentials from environment
email = os.environ.get("GARMIN_EMAIL")
email_file = os.environ.get("GARMIN_EMAIL_FILE")
if email and email_file:
    raise ValueError(
        "Must only provide one of GARMIN_EMAIL and GARMIN_EMAIL_FILE, got both"
    )
elif email_file:
    with open(email_file, "r") as email_file:
        email = email_file.read().rstrip()

password = os.environ.get("GARMIN_PASSWORD")
password_file = os.environ.get("GARMIN_PASSWORD_FILE")
if password and password_file:
    raise ValueError(
        "Must only provide one of GARMIN_PASSWORD and GARMIN_PASSWORD_FILE, got both"
    )
elif password_file:
    with open(password_file, "r") as password_file:
        password = password_file.read().rstrip()

tokenstore = os.getenv("GARMINTOKENS") or "~/.garminconnect"
tokenstore_base64 = os.getenv("GARMINTOKENS_BASE64") or "~/.garminconnect_base64"
is_cn = os.getenv("GARMIN_IS_CN", "false").lower() in ("true", "1", "yes")


# --- Tool filtering ---------------------------------------------------------
# Optionally expose only a subset of tools, to reduce the context an LLM must
# carry. No modules are removed; tools are simply not registered when filtered.
#   GARMIN_ENABLED_TOOLS  - comma-separated allowlist; if set, ONLY these register
#   GARMIN_DISABLED_TOOLS - comma-separated denylist; ignored if an allowlist is set
# Tool names are case-insensitive. Unset = all tools register (default behaviour).
def _parse_tool_set(value):
    if not value:
        return set()
    return {name.strip().lower() for name in value.split(",") if name.strip()}


enabled_tools = _parse_tool_set(os.getenv("GARMIN_ENABLED_TOOLS"))
disabled_tools = _parse_tool_set(os.getenv("GARMIN_DISABLED_TOOLS"))


_VALID_TRANSPORTS = ("stdio", "streamable-http", "sse")


class GarminNotFoundError(Exception):
    """A requested Garmin resource does not exist (HTTP 404).

    Distinct from a connectivity failure — e.g. deleting an already-deleted
    workout, or fetching an id that isn't there.
    """


class GarminClientError(Exception):
    """Garmin rejected the request with a non-404 4xx client error.

    A deterministic, caller-actionable failure (bad request, forbidden, …) —
    not a network/connectivity problem.
    """


class _GarminProxy:
    """Wraps the Garmin client to translate known runtime exceptions into clear messages.

    Without this, token expiry or rate-limiting during a tool call surfaces raw
    library tracebacks to the MCP client. The proxy intercepts each attribute
    access and, if the result is callable, wraps the call so that known Garmin
    exceptions become user-friendly strings rather than server errors.
    """

    _MESSAGES = {
        GarminConnectAuthenticationError: (
            "Garmin authentication expired. "
            "Re-run 'garmin-mcp-auth' to refresh your tokens and restart the server."
        ),
        GarminConnectTooManyRequestsError: (
            "Garmin rate limit hit. Wait a few minutes before retrying."
        ),
        GarminConnectConnectionError: (
            "Garmin Connect is unreachable. Check your network connection or try again later."
        ),
    }

    # The library raises GarminConnectConnectionError for *all* HTTP failures,
    # including 4xx client errors, with messages like "API Error 404 - ...".
    _STATUS_RE = re.compile(r"(?:API Error|Error|HTTP)\s*(\d{3})")

    def __init__(self, client):
        self._client = client

    @classmethod
    def _status_code(cls, exc):
        """Best-effort HTTP status from a Garmin exception (attr or message)."""
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        if isinstance(status, int):
            return status
        match = cls._STATUS_RE.search(str(exc))
        return int(match.group(1)) if match else None

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def _call(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except GarminConnectNotFoundError:
                # The library types 404s distinctly (subclass of the connection
                # error). Surface a clear not-found rather than "unreachable".
                raise GarminNotFoundError(
                    "Not found on Garmin Connect: the requested item does not "
                    "exist (it may have already been deleted)."
                ) from None
            except tuple(self._MESSAGES) as exc:
                # Other 4xx are deterministic client errors, not connectivity
                # failures — surface the status instead of the blanket message.
                if isinstance(exc, GarminConnectConnectionError):
                    status = self._status_code(exc)
                    if status is not None and 400 <= status < 500:
                        raise GarminClientError(
                            f"Garmin Connect rejected the request (HTTP {status})."
                        ) from None
                for exc_type, msg in self._MESSAGES.items():
                    if isinstance(exc, exc_type):
                        error_details = str(exc)
                        full_msg = f"{msg} (Details: {error_details})" if error_details else msg
                        raise type(exc)(full_msg) from None
                raise

        return _call


def _parse_transport_config() -> tuple[str, str, int]:
    """Read and validate HTTP transport env vars. Raises ValueError on bad input."""
    transport = os.getenv("GARMIN_MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(
            f"Invalid GARMIN_MCP_TRANSPORT {transport!r}; "
            f"expected one of {', '.join(_VALID_TRANSPORTS)}"
        )
    # Bind to loopback by default: the HTTP transport performs no authentication,
    # so a 0.0.0.0 default would expose full read/write access to the user's
    # Garmin account to the whole network. Opt in explicitly with GARMIN_MCP_HOST.
    http_host = os.getenv("GARMIN_MCP_HOST", "127.0.0.1")
    http_port = int(os.getenv("GARMIN_MCP_PORT", "8000"))
    return transport, http_host, http_port


class _ToolFilter:
    """Wraps a FastMCP app to conditionally register tools by function name.

    Modules register via ``@app.tool()``; we intercept that decorator and skip
    registration for any tool not permitted by the env-var filter. All other
    attribute access (``run``, ``resource``, ...) passes through to the app.
    """

    def __init__(self, app, enabled, disabled):
        self._app = app
        self._enabled = enabled
        self._disabled = disabled
        self._seen = set()  # tool names encountered, for typo detection

    def _allowed(self, name):
        name = name.lower()
        if self._enabled:
            return name in self._enabled
        return name not in self._disabled

    def tool(self, *args, **kwargs):
        decorator = self._app.tool(*args, **kwargs)
        # Prefer the explicit registered name if given (@app.tool(name="x")),
        # so the env-var filter matches what the user actually configures.
        explicit = kwargs.get("name") or (
            args[0] if args and isinstance(args[0], str) else None
        )

        def wrapper(fn):
            name = explicit or getattr(fn, "__name__", "")
            self._seen.add(name.lower())
            if self._allowed(name):
                return decorator(fn)
            return fn  # skip registration; tool never reaches the LLM

        return wrapper

    def unknown_filter_names(self):
        """Configured names that never matched a real tool (likely typos)."""
        configured = self._enabled or self._disabled
        return sorted(configured - self._seen)

    def __getattr__(self, item):
        return getattr(self._app, item)
# ---------------------------------------------------------------------------


class _UnauthenticatedClient:
    """Stand-in used before login. Any Garmin call raises a clear, actionable error.

    Lets the server start without tokens so the garmin_login tool is reachable;
    all other tools report that a login is required rather than crashing.
    """

    def __getattr__(self, name):
        def _needs_login(*args, **kwargs):
            raise GarminConnectAuthenticationError(
                "Not logged in to Garmin Connect. Call garmin_login(email, password) first."
            )
        return _needs_login


def init_api(email, password):
    """Initialize Garmin API with your credentials."""
    import io

    try:
        # Using Oauth1 and OAuth2 token files from directory
        print(
            f"Trying to login to Garmin Connect using token data from directory '{tokenstore}'...\n",
            file=sys.stderr,
        )

        # Using Oauth1 and Oauth2 tokens from base64 encoded string
        # print(
        #     f"Trying to login to Garmin Connect using token data from file '{tokenstore_base64}'...\n"
        # )
        # dir_path = os.path.expanduser(tokenstore_base64)
        # with open(dir_path, "r") as token_file:
        #     tokenstore = token_file.read()

        # Suppress stderr for token validation to avoid confusing library errors
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            garmin = Garmin(is_cn=is_cn)
            garmin.login(tokenstore)
        finally:
            sys.stderr = old_stderr

    except (FileNotFoundError, GarminConnectConnectionError, GarminConnectTooManyRequestsError, GarminConnectAuthenticationError):
        # Session is expired. You'll need to log in again

        # Check if we're in a non-interactive environment without credentials
        if not is_interactive_terminal() and (not email or not password):
            print(
                "ERROR: OAuth tokens not found and no interactive terminal available.\n"
                "Please authenticate first:\n"
                "  1. Run: garmin-mcp-auth\n"
                "  2. Enter your credentials and MFA code\n"
                "  3. Restart your MCP client\n"
                f"Tokens will be saved to: {tokenstore}\n",
                file=sys.stderr,
            )
            return None

        print(
            "Login tokens not present, login with your Garmin Connect credentials to generate them.\n"
            f"They will be stored in '{tokenstore}' for future use.\n",
            file=sys.stderr,
        )
        try:
            garmin = Garmin(
                email=email, password=password, is_cn=is_cn, prompt_mfa=get_mfa, return_on_mfa=True
            )
            result1, result2 = garmin.login()
            if result1 == "needs_mfa":
                mfa_code = get_mfa()
                garmin.resume_login(result2, mfa_code)
            # Save Oauth1 and Oauth2 token files to directory for next login
            garmin.client.dump(tokenstore)
            # Restrict the freshly written tokens to owner-only. These are
            # ~6-month bearer credentials; the default umask would otherwise
            # leave them world-readable on multi-user hosts.
            token_utils.secure_token_dir(tokenstore)
            print(
                f"Oauth tokens stored in '{tokenstore}' directory for future use. (first method)\n",
                file=sys.stderr,
            )
            # Encode Oauth1 and Oauth2 tokens to base64 string and save to file for next login (alternative way)
            expanded_tokenstore = os.path.expanduser(tokenstore)
            token_json_path = os.path.join(expanded_tokenstore, "garmin_tokens.json")
            with open(token_json_path, "r") as f:
                token_data = f.read()
            token_base64 = base64.b64encode(token_data.encode()).decode()
            dir_path = os.path.expanduser(tokenstore_base64)
            with open(dir_path, "w") as token_file:
                token_file.write(token_base64)
            os.chmod(dir_path, 0o600)
            print(
                f"Oauth tokens encoded as base64 string and saved to '{dir_path}' file for future use. (second method)\n",
                file=sys.stderr,
            )
        except (
            FileNotFoundError,
            GarminConnectConnectionError,
            GarminConnectTooManyRequestsError,
            GarminConnectAuthenticationError,
            requests.exceptions.HTTPError,
        ) as err:
            error_msg = str(err)

            # Provide clean, actionable error messages
            print("\nAuthentication failed.", file=sys.stderr)

            if isinstance(err, GarminConnectAuthenticationError):
                if "MFA" in error_msg or "code" in error_msg.lower():
                    print("MFA code may be incorrect or expired.", file=sys.stderr)
                else:
                    print("Invalid email or password.", file=sys.stderr)
            elif isinstance(err, GarminConnectTooManyRequestsError):
                print(
                    "Too many requests. Please wait and try again.", file=sys.stderr
                )
            elif isinstance(err, GarminConnectConnectionError):
                if "401" in error_msg or "Unauthorized" in error_msg:
                    print(
                        "Invalid credentials. Please check your email and password.",
                        file=sys.stderr,
                    )
                elif "500" in error_msg or "503" in error_msg:
                    print(
                        "Garmin Connect service issue. Please try again later.",
                        file=sys.stderr,
                    )
                else:
                    print(f"Error: {error_msg.split(':')[0]}", file=sys.stderr)
            elif isinstance(err, requests.exceptions.HTTPError):
                print("Network error. Please check your connection.", file=sys.stderr)
            else:
                print(f"Error: {error_msg.split(':')[0]}", file=sys.stderr)

            print(
                f"\nTip: Run 'garmin-mcp-auth' to authenticate interactively.",
                file=sys.stderr,
            )
            return None

    return garmin


def main():
    """Initialize the MCP server and register all tools"""

    # On Windows, stdout runs in text mode and translates \n to \r\n, which
    # breaks the MCP stdio framing that Claude Desktop and other clients expect.
    # Force binary-transparent newlines so JSON messages arrive intact.
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, newline="\n")

    # --- Transport configuration --------------------------------------------
    # By default the server speaks stdio (Claude Desktop, MCP Inspector, etc.).
    # Set GARMIN_MCP_TRANSPORT=streamable-http (or sse) to serve over HTTP.
    #   GARMIN_MCP_TRANSPORT - stdio (default) | streamable-http | sse
    #   GARMIN_MCP_HOST      - bind address for HTTP transports (default 127.0.0.1)
    #   GARMIN_MCP_PORT      - bind port for HTTP transports (default 8000)
    try:
        transport, http_host, http_port = _parse_transport_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # All tool modules that hold a `garmin_client` global via configure().
    modules = [
        activity_management, health_wellness, user_profile, devices,
        gear_management, weight_management, challenges, training, workouts,
        data_management, womens_health, nutrition, workout_builders, courses,
        activity_analysis,
    ]

    def _apply_authenticated(garmin):
        """Point every module at a live, proxy-wrapped Garmin client."""
        proxy = _GarminProxy(garmin)
        for module in modules:
            module.configure(proxy)

    def _apply_unauthenticated():
        """Point every module at the sentinel so calls report 'login required'."""
        sentinel = _UnauthenticatedClient()
        for module in modules:
            module.configure(sentinel)

    # Initialize Garmin client. Unlike before, a missing/expired token store no
    # longer aborts startup — the server comes up unauthenticated so the
    # garmin_login tool is reachable and can populate the token store at runtime.
    raw_client = init_api(email, password)
    if raw_client:
        _apply_authenticated(raw_client)
        print("Garmin Connect client initialized successfully.", file=sys.stderr)
    else:
        _apply_unauthenticated()
        print(
            "Garmin not authenticated. Use the garmin_login tool to log in; "
            "tokens will be saved to the token store.",
            file=sys.stderr,
        )

    # Wire the runtime auth tools to the token store and client-swap callbacks.
    auth.configure_auth(
        tokenstore=tokenstore,
        tokenstore_base64=tokenstore_base64,
        is_cn=is_cn,
        apply_authenticated=_apply_authenticated,
        apply_unauthenticated=_apply_unauthenticated,
        initial_client=raw_client,
    )

    # Create the MCP app, wrapped so the env-var filter can drop tools.
    # host/port only matter for the HTTP transports; stdio ignores them.
    fastmcp = FastMCP("Garmin Connect v1.0", host=http_host, port=http_port)
    app = _ToolFilter(fastmcp, enabled_tools, disabled_tools)
    if enabled_tools:
        print(f"Tool filter: allowlist of {len(enabled_tools)} tool(s).", file=sys.stderr)
    elif disabled_tools:
        print(f"Tool filter: denylist of {len(disabled_tools)} tool(s).", file=sys.stderr)

    # Register tools from all modules
    app = activity_management.register_tools(app)
    app = health_wellness.register_tools(app)
    app = user_profile.register_tools(app)
    app = devices.register_tools(app)
    app = gear_management.register_tools(app)
    app = weight_management.register_tools(app)
    app = challenges.register_tools(app)
    app = training.register_tools(app)
    app = workouts.register_tools(app)
    app = data_management.register_tools(app)
    app = womens_health.register_tools(app)
    app = nutrition.register_tools(app)
    app = workout_builders.register_tools(app)
    app = courses.register_tools(app)
    app = activity_analysis.register_tools(app)
    app = auth.register_tools(app)

    # Register resources (workout templates)
    app = workout_templates.register_resources(app)

    # Warn about filter entries that matched no tool (most likely typos)
    unknown = app.unknown_filter_names()
    if unknown:
        print(
            f"Tool filter: warning — name(s) not found and ignored: {', '.join(unknown)}",
            file=sys.stderr,
        )

    # When serving over HTTP, expose a plain health endpoint for k8s probes.
    # The MCP endpoint itself requires a handshake and isn't probe-friendly.
    if transport != "stdio":
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse

        @fastmcp.custom_route("/healthz", methods=["GET"])
        async def healthz(_request: "Request") -> "PlainTextResponse":
            return PlainTextResponse("ok")

        print(
            f"Serving MCP over {transport} on {http_host}:{http_port}",
            file=sys.stderr,
        )

    # Run the MCP server
    app.run(transport=transport)


if __name__ == "__main__":
    main()
