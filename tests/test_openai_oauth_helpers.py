"""Tests for OpenGeoAgent OpenAI OAuth helper behavior."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import pathlib
import sys
import urllib.parse

import pytest

OAUTH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "qgis_geoagent"
    / "open_geoagent"
    / "oauth.py"
)
SPEC = importlib.util.spec_from_file_location("open_geoagent_oauth", OAUTH_PATH)
oauth = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = oauth
SPEC.loader.exec_module(oauth)


class FakeSettings:
    """Small QSettings stand-in for OAuth helper tests."""

    def __init__(self):
        self.values = {}

    def value(self, key, default="", type=str):  # noqa: A002
        """Return a stored value."""
        value = self.values.get(key, default)
        if type is str and value is not None:
            return str(value)
        return value

    def setValue(self, key, value):  # noqa: N802
        """Store a value."""
        self.values[key] = value

    def remove(self, key):
        """Remove a value."""
        self.values.pop(key, None)


def test_generate_pkce_pair_uses_s256_challenge() -> None:
    """Verify PKCE challenge generation."""
    verifier, challenge = oauth.generate_pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )

    assert len(verifier) >= 43
    assert challenge == expected


def test_extract_authorization_code_validates_state() -> None:
    """Verify callback state validation."""
    code = oauth.extract_authorization_code(
        "http://127.0.0.1:49152/callback?code=abc&state=expected",
        "expected",
    )

    assert code == "abc"
    with pytest.raises(ValueError, match="state"):
        oauth.extract_authorization_code(
            "http://127.0.0.1:49152/callback?code=abc&state=other",
            "expected",
        )


def test_codex_authorization_url_matches_registered_flow_shape() -> None:
    """Verify Codex login uses the expected redirect and auth parameters."""
    url = oauth.build_authorization_url(
        oauth.OPENAI_CODEX_AUTHORIZATION_URL,
        client_id=oauth.OPENAI_CODEX_CLIENT_ID,
        redirect_uri="http://localhost:1455/auth/callback",
        code_challenge="challenge",
        state="state",
        scope=oauth.OPENAI_CODEX_SCOPE,
        extra_params=oauth.OPENAI_CODEX_AUTH_EXTRA_PARAMS,
    )
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.openai.com"
    assert params["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert params["client_id"] == [oauth.OPENAI_CODEX_CLIENT_ID]
    assert params["scope"] == ["openid profile email offline_access"]
    assert params["id_token_add_organizations"] == ["true"]
    assert params["codex_cli_simplified_flow"] == ["true"]
    assert params["originator"] == ["pi"]


def test_token_expires_soon_with_skew() -> None:
    """Verify expiry refresh decisions."""
    assert oauth.token_expires_soon(None, now=1000)
    assert oauth.token_expires_soon(1200, now=1000, skew_seconds=300)
    assert not oauth.token_expires_soon(1400, now=1000, skew_seconds=300)


def test_extract_chatgpt_account_id_from_access_token() -> None:
    """Verify account id extraction from a JWT payload."""
    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "account-123",
        }
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    token = "header." + encoded.rstrip("=") + ".signature"

    assert oauth.extract_chatgpt_account_id({"access_token": token}) == "account-123"


def test_store_load_and_clear_token_payload(monkeypatch) -> None:
    """Verify token storage uses only an auth reference in settings."""
    settings = FakeSettings()
    stored = {}
    removed = []

    def fake_store(payload, existing_authcfg=""):
        stored["payload"] = payload
        stored["existing_authcfg"] = existing_authcfg
        return "authcfg-1"

    def fake_load(authcfg):
        stored["loaded_authcfg"] = authcfg
        return stored["payload"]

    def fake_remove(authcfg):
        removed.append(authcfg)

    monkeypatch.setattr(oauth, "_store_secret_payload", fake_store)
    monkeypatch.setattr(oauth, "_load_secret_payload", fake_load)
    monkeypatch.setattr(oauth, "_remove_secret_payload", fake_remove)

    token = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_at": 2000,
        "token_type": "Bearer",
    }
    authcfg = oauth.store_token_payload(settings, token)

    assert authcfg == "authcfg-1"
    assert settings.value("OpenGeoAgent/openai_oauth_authcfg") == "authcfg-1"
    assert "access_token" not in settings.values
    assert oauth.load_token_payload(settings) == token

    oauth.clear_token_payload(settings)

    assert removed == ["authcfg-1"]
    assert "OpenGeoAgent/openai_oauth_authcfg" not in settings.values
    assert "OpenGeoAgent/openai_oauth_expires_at" not in settings.values


def test_ensure_environment_allows_codex_env_token(monkeypatch) -> None:
    """Verify Codex auth can use built-in base URL with an env token."""
    settings = FakeSettings()
    monkeypatch.setenv("OPENAI_CODEX_ACCESS_TOKEN", "codex-token")
    monkeypatch.delenv("OPENAI_CODEX_BASE_URL", raising=False)

    oauth.ensure_openai_oauth_environment(settings, codex=True)

    assert oauth.os.environ["OPENAI_CODEX_ACCESS_TOKEN"] == "codex-token"
    assert (
        oauth.os.environ["OPENAI_CODEX_BASE_URL"]
        == "https://chatgpt.com/backend-api/codex"
    )
