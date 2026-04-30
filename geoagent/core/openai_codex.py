"""ChatGPT/Codex OAuth helpers for Python scripts and notebooks."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

OPENAI_CODEX_AUTHORIZATION_URL = "https://auth.openai.com/oauth/authorize"
# Public OAuth endpoint URL, not a credential.
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"  # nosec B105
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_SCOPE = "openid profile email offline_access"
OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENAI_CODEX_CALLBACK_PORT = 1455
OPENAI_CODEX_CALLBACK_PATH = "/auth/callback"
OPENAI_CODEX_AUTH_EXTRA_PARAMS = {  # nosec B105
    "id_token_add_organizations": "true",
    "codex_cli_simplified_flow": "true",
    "originator": "pi",
}
REFRESH_SKEW_SECONDS = 300
TOKEN_FILE_ENV = "GEOAGENT_CODEX_TOKEN_FILE"


@dataclass
class OAuthLoopbackFlow:
    """Pending localhost callback flow."""

    authorization_url: str
    redirect_uri: str
    state: str
    code_verifier: str
    server: HTTPServer
    collector: dict[str, Any]


def default_token_file() -> Path:
    """Return the default file used for non-QGIS Codex OAuth token storage."""
    override = os.environ.get(TOKEN_FILE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "geoagent" / "openai_codex_oauth.json"


def generate_pkce_pair() -> tuple[str, str]:
    """Return an OAuth PKCE verifier and S256 challenge."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def generate_state() -> str:
    """Return a cryptographically random OAuth state value."""
    return secrets.token_urlsafe(32)


def token_expires_soon(
    expires_at: float | int | str | None,
    *,
    now: float | None = None,
    skew_seconds: int = REFRESH_SKEW_SECONDS,
) -> bool:
    """Return True when a token is missing or near expiry."""
    if not expires_at:
        return True
    try:
        expiry = float(expires_at)
    except (TypeError, ValueError):
        return True
    current = time.time() if now is None else now
    return expiry <= current + skew_seconds


def decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode an unsigned JWT payload for local claim extraction."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        data = base64.urlsafe_b64decode((payload + padding).encode("ascii"))
        decoded = json.loads(data.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def extract_chatgpt_account_id(token_payload: dict[str, Any]) -> str:
    """Extract the ChatGPT account id from OAuth token claims when present."""
    for token_key in ("access_token", "id_token"):
        claims = decode_jwt_payload(str(token_payload.get(token_key, "")))
        direct = claims.get("https://api.openai.com/auth.chatgpt_account_id")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        auth_claim = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claim, dict):
            nested = auth_claim.get("chatgpt_account_id")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        for fallback_key in ("chatgpt_account_id", "account_id"):
            fallback = claims.get(fallback_key)
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
    return ""


def build_authorization_url(
    authorization_url: str,
    *,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    scope: str = "",
    extra_params: dict[str, str] | None = None,
) -> str:
    """Build an OAuth authorization-code URL with PKCE parameters."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if scope:
        params["scope"] = scope
    if extra_params:
        params.update(extra_params)
    separator = "&" if urllib.parse.urlparse(authorization_url).query else "?"
    return authorization_url + separator + urllib.parse.urlencode(params)


def extract_authorization_code(callback_url: str, expected_state: str) -> str:
    """Validate a callback URL and return its authorization code."""
    parsed = urllib.parse.urlparse(callback_url)
    query = urllib.parse.parse_qs(parsed.query)
    state = query.get("state", [""])[0]
    if state != expected_state:
        raise ValueError("OAuth callback state did not match the login request.")
    error = query.get("error", [""])[0]
    if error:
        description = query.get("error_description", [""])[0]
        raise ValueError(description or f"OAuth authorization failed: {error}")
    code = query.get("code", [""])[0]
    if not code:
        raise ValueError("OAuth callback did not include an authorization code.")
    return code


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Capture a single OAuth callback."""

    def do_GET(self):  # noqa: N802
        """Handle the OAuth redirect."""
        collector = self.server.collector  # type: ignore[attr-defined]
        collector["url"] = f"http://127.0.0.1:{self.server.server_port}{self.path}"
        body = (
            "<html><body><h3>GeoAgent login complete</h3>"
            "<p>You can close this browser tab and return to Python.</p>"
            "</body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        """Suppress callback server logging."""
        return


def start_loopback_flow(
    authorization_url: str = OPENAI_CODEX_AUTHORIZATION_URL,
    *,
    client_id: str = OPENAI_CODEX_CLIENT_ID,
    scope: str = OPENAI_CODEX_SCOPE,
    bind_host: str = "127.0.0.1",
    redirect_host: str = "localhost",
    port: int = OPENAI_CODEX_CALLBACK_PORT,
    callback_path: str = OPENAI_CODEX_CALLBACK_PATH,
    extra_params: dict[str, str] | None = None,
    fallback_port: bool = False,
) -> OAuthLoopbackFlow:
    """Start a localhost callback server and return the pending OAuth flow."""
    verifier, challenge = generate_pkce_pair()
    state = generate_state()
    collector: dict[str, Any] = {}
    try:
        server = HTTPServer((bind_host, port), _OAuthCallbackHandler)
    except OSError as exc:
        if port == 0 or not fallback_port:
            if port:
                raise RuntimeError(
                    f"OAuth callback port {port} is not available."
                ) from exc
            raise
        server = HTTPServer((bind_host, 0), _OAuthCallbackHandler)
    server.timeout = 1
    server.collector = collector  # type: ignore[attr-defined]
    if not callback_path.startswith("/"):
        callback_path = "/" + callback_path
    redirect_uri = f"http://{redirect_host}:{server.server_port}{callback_path}"
    url = build_authorization_url(
        authorization_url,
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=challenge,
        state=state,
        scope=scope,
        extra_params=extra_params or OPENAI_CODEX_AUTH_EXTRA_PARAMS,
    )
    return OAuthLoopbackFlow(url, redirect_uri, state, verifier, server, collector)


def complete_loopback_flow(
    flow: OAuthLoopbackFlow,
    *,
    token_url: str = OPENAI_CODEX_TOKEN_URL,
    client_id: str = OPENAI_CODEX_CLIENT_ID,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Wait for the browser callback and exchange the code for tokens."""
    deadline = time.time() + timeout_seconds
    try:
        while time.time() < deadline and "url" not in flow.collector:
            flow.server.handle_request()
        if "url" not in flow.collector:
            raise TimeoutError("Timed out waiting for the OAuth browser callback.")
        code = extract_authorization_code(flow.collector["url"], flow.state)
        return exchange_authorization_code(
            token_url,
            client_id=client_id,
            code=code,
            redirect_uri=flow.redirect_uri,
            code_verifier=flow.code_verifier,
        )
    finally:
        flow.server.server_close()


def _post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
    """POST URL-encoded form data and return decoded JSON."""
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OAuth token request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OAuth token request failed: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OAuth token endpoint did not return JSON.") from exc


def _normalize_token_response(response: dict[str, Any]) -> dict[str, Any]:
    """Validate and add an absolute expiry timestamp to a token response."""
    access_token = str(response.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("OAuth token endpoint did not return an access token.")
    normalized = dict(response)
    try:
        expires_in = int(normalized.get("expires_in", 3600))
    except (TypeError, ValueError):
        expires_in = 3600
    normalized["expires_at"] = int(time.time()) + max(expires_in, 0)
    normalized["token_type"] = str(normalized.get("token_type") or "Bearer")
    account_id = extract_chatgpt_account_id(normalized)
    if account_id:
        normalized["account_id"] = account_id
    return normalized


def exchange_authorization_code(
    token_url: str,
    *,
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    """Exchange an authorization code for an OAuth token payload."""
    response = _post_form(
        token_url,
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
    )
    return _normalize_token_response(response)


def refresh_oauth_token(
    token_url: str = OPENAI_CODEX_TOKEN_URL,
    *,
    client_id: str = OPENAI_CODEX_CLIENT_ID,
    refresh_token: str,
    scope: str = OPENAI_CODEX_SCOPE,
) -> dict[str, Any]:
    """Refresh an OAuth access token."""
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    if scope:
        data["scope"] = scope
    response = _post_form(token_url, data)
    normalized = _normalize_token_response(response)
    if "refresh_token" not in normalized:
        normalized["refresh_token"] = refresh_token
    return normalized


def save_token_payload(
    token_payload: dict[str, Any],
    *,
    token_file: str | os.PathLike[str] | None = None,
) -> Path:
    """Store Codex OAuth tokens in a local user-only JSON file."""
    path = (
        Path(token_file).expanduser()
        if token_file is not None
        else default_token_file()
    )
    if path.is_symlink():
        raise RuntimeError(f"Refusing to write Codex tokens through a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(token_payload, indent=2, sort_keys=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(tmp_path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(payload)
            file.write("\n")
        try:
            tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def load_token_payload(
    *,
    token_file: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load the locally stored Codex OAuth token payload."""
    path = (
        Path(token_file).expanduser()
        if token_file is not None
        else default_token_file()
    )
    if not path.exists():
        raise RuntimeError(
            "OpenAI Codex is not logged in. Run `geoagent codex login` or call "
            "`geoagent.login_openai_codex()` first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Stored Codex token file is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stored Codex token file is invalid: {path}")
    return payload


def clear_token_payload(
    *,
    token_file: str | os.PathLike[str] | None = None,
) -> None:
    """Remove the locally stored Codex OAuth token payload."""
    path = (
        Path(token_file).expanduser()
        if token_file is not None
        else default_token_file()
    )
    try:
        path.unlink()
    except FileNotFoundError:
        return


def export_openai_codex_environment(
    token_payload: dict[str, Any],
    *,
    base_url: str | None = None,
) -> None:
    """Export token payload values to environment variables for model creation."""
    access_token = str(token_payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("Stored OpenAI Codex token payload has no access token.")
    os.environ["OPENAI_CODEX_ACCESS_TOKEN"] = access_token
    os.environ["OPENAI_CODEX_BASE_URL"] = (
        base_url or os.environ.get("OPENAI_CODEX_BASE_URL") or OPENAI_CODEX_BASE_URL
    )
    account_id = str(token_payload.get("account_id", "")).strip()
    if not account_id:
        account_id = extract_chatgpt_account_id(token_payload)
    if account_id:
        os.environ["OPENAI_CODEX_ACCOUNT_ID"] = account_id


def ensure_openai_codex_environment(
    *,
    token_file: str | os.PathLike[str] | None = None,
    refresh: bool = True,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Load or refresh Codex OAuth tokens and export env vars for GeoAgent."""
    env_token = os.environ.get("OPENAI_CODEX_ACCESS_TOKEN", "").strip()
    if env_token:
        payload = {"access_token": env_token}
        account_id = os.environ.get("OPENAI_CODEX_ACCOUNT_ID", "").strip()
        if account_id:
            payload["account_id"] = account_id
        export_openai_codex_environment(payload, base_url=base_url)
        return payload

    payload = load_token_payload(token_file=token_file)
    if refresh and token_expires_soon(payload.get("expires_at")):
        refresh_token = str(payload.get("refresh_token", "")).strip()
        if not refresh_token:
            raise RuntimeError(
                "OpenAI Codex token expired and no refresh token is stored. "
                "Run `geoagent codex login` again."
            )
        payload = refresh_oauth_token(refresh_token=refresh_token)
        save_token_payload(payload, token_file=token_file)
    export_openai_codex_environment(payload, base_url=base_url)
    return payload


def login_openai_codex(
    *,
    open_browser: bool = True,
    timeout_seconds: int = 180,
    token_file: str | os.PathLike[str] | None = None,
    save: bool = True,
    set_environment: bool = True,
) -> dict[str, Any]:
    """Run the ChatGPT/Codex browser login flow for scripts or notebooks."""
    flow = start_loopback_flow()
    if open_browser:
        opened = webbrowser.open(flow.authorization_url)
        if not opened:
            print("Open this URL to finish ChatGPT login:")
            print(flow.authorization_url)
    else:
        print("Open this URL to finish ChatGPT login:")
        print(flow.authorization_url)
    payload = complete_loopback_flow(flow, timeout_seconds=timeout_seconds)
    if save:
        save_token_payload(payload, token_file=token_file)
    if set_environment:
        export_openai_codex_environment(payload)
    return payload


def is_openai_codex_logged_in(
    *,
    token_file: str | os.PathLike[str] | None = None,
) -> bool:
    """Return True when a local Codex OAuth token payload exists."""
    try:
        payload = load_token_payload(token_file=token_file)
    except RuntimeError:
        return False
    return bool(str(payload.get("access_token", "")).strip())
