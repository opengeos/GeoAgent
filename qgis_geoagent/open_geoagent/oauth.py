"""OAuth helpers for the OpenGeoAgent QGIS plugin.

The plugin supports OAuth through an external OpenAI-compatible proxy. Direct
OpenAI API calls still use API keys.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

SETTINGS_PREFIX = "OpenGeoAgent/"
OPENAI_CODEX_AUTHORIZATION_URL = "https://auth.openai.com/oauth/authorize"
# Public OAuth endpoint URL, not a credential.
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"  # nosec B105
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_SCOPE = "openid profile email offline_access"
OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENAI_CODEX_CALLBACK_PORT = 1455
OPENAI_CODEX_CALLBACK_PATH = "/auth/callback"
# OAuth authorize-request flag values below are protocol parameters, not credentials.
OPENAI_CODEX_AUTH_EXTRA_PARAMS = {  # nosec B105
    "id_token_add_organizations": "true",
    "codex_cli_simplified_flow": "true",
    "originator": "pi",
}
AUTHCFG_KEY = "openai_oauth_authcfg"
EXPIRES_AT_KEY = "openai_oauth_expires_at"
# QSettings key name (where the token type is stored), not a credential.
TOKEN_TYPE_KEY = "openai_oauth_token_type"  # nosec B105
REFRESH_SKEW_SECONDS = 300

OAUTH_CONFIG_KEYS = (
    "openai_oauth_authorization_url",
    "openai_oauth_token_url",
    "openai_oauth_client_id",
    "openai_oauth_scope",
    "openai_oauth_base_url",
)
OAUTH_TOKEN_SETTING_KEYS = (
    AUTHCFG_KEY,
    EXPIRES_AT_KEY,
    TOKEN_TYPE_KEY,
)
CODEX_DEFAULT_CONFIG = {
    "authorization_url": OPENAI_CODEX_AUTHORIZATION_URL,
    "token_url": OPENAI_CODEX_TOKEN_URL,
    "client_id": OPENAI_CODEX_CLIENT_ID,
    "scope": OPENAI_CODEX_SCOPE,
    "base_url": OPENAI_CODEX_BASE_URL,
}


@dataclass
class OAuthLoopbackFlow:
    """Pending localhost callback flow."""

    authorization_url: str
    redirect_uri: str
    state: str
    code_verifier: str
    server: HTTPServer
    collector: dict[str, Any]


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
    except (ValueError, json.JSONDecodeError):
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
            "<html><body><h3>OpenGeoAgent login complete</h3>"
            "<p>You can close this browser tab and return to QGIS.</p>"
            "</body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        """Suppress callback server logging in QGIS."""
        return


def start_loopback_flow(
    authorization_url: str,
    *,
    client_id: str,
    scope: str = "",
    bind_host: str = "127.0.0.1",
    redirect_host: str = "127.0.0.1",
    port: int = 0,
    callback_path: str = "/callback",
    extra_params: dict[str, str] | None = None,
    fallback_port: bool = True,
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
        extra_params=extra_params,
    )
    return OAuthLoopbackFlow(url, redirect_uri, state, verifier, server, collector)


def complete_loopback_flow(
    flow: OAuthLoopbackFlow,
    *,
    token_url: str,
    client_id: str,
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
    token_url: str,
    *,
    client_id: str,
    refresh_token: str,
    scope: str = "",
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


def _qgis_auth_manager():
    """Return the QGIS auth manager or raise a clear setup error."""
    try:
        from qgis.core import QgsApplication
    except Exception as exc:
        raise RuntimeError(
            "QGIS Auth Manager is required for OAuth token storage."
        ) from exc
    manager = QgsApplication.authManager()
    if manager is None:
        raise RuntimeError("QGIS Auth Manager is not available.")
    return manager


def _store_secret_payload(payload: dict[str, Any], existing_authcfg: str = "") -> str:
    """Store OAuth tokens in QGIS Auth Manager and return the auth config id."""
    try:
        from qgis.core import QgsAuthMethodConfig
    except Exception as exc:
        raise RuntimeError(
            "QGIS Auth Manager is required for OAuth token storage."
        ) from exc

    manager = _qgis_auth_manager()
    config = QgsAuthMethodConfig()
    if existing_authcfg:
        try:
            manager.loadAuthenticationConfig(existing_authcfg, config, True)
        except TypeError:
            manager.loadAuthenticationConfig(existing_authcfg, config)
    config.setName("OpenGeoAgent ChatGPT Login")
    config.setMethod("Basic")
    config.setConfig("username", "openai-codex")
    config.setConfig("password", json.dumps(payload))
    if existing_authcfg:
        ok = manager.updateAuthenticationConfig(config)
        authcfg = existing_authcfg
    else:
        ok = manager.storeAuthenticationConfig(config)
        authcfg = config.id()
    if not ok:
        raise RuntimeError("Failed to store OAuth tokens in QGIS Auth Manager.")
    return authcfg


def _load_secret_payload(authcfg: str) -> dict[str, Any]:
    """Load OAuth tokens from QGIS Auth Manager."""
    try:
        from qgis.core import QgsAuthMethodConfig
    except Exception as exc:
        raise RuntimeError(
            "QGIS Auth Manager is required for OAuth token storage."
        ) from exc

    manager = _qgis_auth_manager()
    config = QgsAuthMethodConfig()
    try:
        ok = manager.loadAuthenticationConfig(authcfg, config, True)
    except TypeError:
        ok = manager.loadAuthenticationConfig(authcfg, config)
    if not ok:
        raise RuntimeError("OpenAI OAuth tokens were not found in QGIS Auth Manager.")
    raw = config.config("password")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Stored OpenAI OAuth token payload is invalid.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Stored OpenAI OAuth token payload is invalid.")
    return payload


def _remove_secret_payload(authcfg: str) -> None:
    """Remove a QGIS Auth Manager token payload."""
    if not authcfg:
        return
    manager = _qgis_auth_manager()
    manager.removeAuthenticationConfig(authcfg)


def store_token_payload(
    settings,
    token_payload: dict[str, Any],
    *,
    prefix: str = SETTINGS_PREFIX,
) -> str:
    """Store OAuth tokens and persist only the auth reference in QSettings."""
    existing_authcfg = settings.value(f"{prefix}{AUTHCFG_KEY}", "", type=str)
    authcfg = _store_secret_payload(token_payload, str(existing_authcfg or ""))
    settings.setValue(f"{prefix}{AUTHCFG_KEY}", authcfg)
    settings.setValue(
        f"{prefix}{EXPIRES_AT_KEY}", str(token_payload.get("expires_at", ""))
    )
    settings.setValue(
        f"{prefix}{TOKEN_TYPE_KEY}", str(token_payload.get("token_type", "Bearer"))
    )
    return authcfg


def load_token_payload(settings, *, prefix: str = SETTINGS_PREFIX) -> dict[str, Any]:
    """Load the OAuth token payload referenced by QSettings."""
    authcfg = settings.value(f"{prefix}{AUTHCFG_KEY}", "", type=str)
    if not str(authcfg).strip():
        raise RuntimeError("OpenAI OAuth is not logged in.")
    return _load_secret_payload(str(authcfg))


def clear_token_payload(settings, *, prefix: str = SETTINGS_PREFIX) -> None:
    """Remove OAuth tokens and their QSettings references."""
    authcfg = settings.value(f"{prefix}{AUTHCFG_KEY}", "", type=str)
    if str(authcfg).strip():
        _remove_secret_payload(str(authcfg))
    for key in OAUTH_TOKEN_SETTING_KEYS:
        settings.remove(f"{prefix}{key}")


def ensure_openai_oauth_environment(
    settings,
    *,
    prefix: str = SETTINGS_PREFIX,
    codex: bool = True,
) -> None:
    """Refresh tokens if needed and export env vars for GeoAgent model creation."""
    base_url = settings.value(f"{prefix}openai_oauth_base_url", "", type=str).strip()
    if not base_url:
        base_url = os.environ.get("OPENAI_CODEX_BASE_URL", "").strip()
    if not base_url:
        base_url = OPENAI_CODEX_BASE_URL
    if not base_url:
        raise RuntimeError("OpenAI Codex base URL is not configured.")

    authcfg = settings.value(f"{prefix}{AUTHCFG_KEY}", "", type=str)
    if not str(authcfg).strip():
        access_token = os.environ.get("OPENAI_CODEX_ACCESS_TOKEN", "").strip()
        if access_token:
            os.environ["OPENAI_CODEX_ACCESS_TOKEN"] = access_token
            os.environ["OPENAI_CODEX_BASE_URL"] = base_url
            account_id = os.environ.get("OPENAI_CODEX_ACCOUNT_ID", "").strip()
            if not account_id:
                account_id = extract_chatgpt_account_id({"access_token": access_token})
            if account_id:
                os.environ["OPENAI_CODEX_ACCOUNT_ID"] = account_id
            return

    payload = load_token_payload(settings, prefix=prefix)
    if token_expires_soon(payload.get("expires_at")):
        refresh_token = str(payload.get("refresh_token", "")).strip()
        if not refresh_token:
            raise RuntimeError(
                "OpenAI OAuth token expired and no refresh token is stored."
            )
        token_url = settings.value(
            f"{prefix}openai_oauth_token_url", "", type=str
        ).strip()
        client_id = settings.value(
            f"{prefix}openai_oauth_client_id", "", type=str
        ).strip()
        scope = settings.value(f"{prefix}openai_oauth_scope", "", type=str).strip()
        token_url = token_url or OPENAI_CODEX_TOKEN_URL
        client_id = client_id or OPENAI_CODEX_CLIENT_ID
        scope = scope or OPENAI_CODEX_SCOPE
        if not token_url or not client_id:
            raise RuntimeError("OpenAI OAuth token URL and client ID are required.")
        payload = refresh_oauth_token(
            token_url,
            client_id=client_id,
            refresh_token=refresh_token,
            scope=scope,
        )
        store_token_payload(settings, payload, prefix=prefix)

    access_token = str(payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("Stored OpenAI OAuth token payload has no access token.")
    os.environ["OPENAI_CODEX_ACCESS_TOKEN"] = access_token
    os.environ["OPENAI_CODEX_BASE_URL"] = base_url
    account_id = str(payload.get("account_id", "")).strip()
    if account_id:
        os.environ["OPENAI_CODEX_ACCOUNT_ID"] = account_id
