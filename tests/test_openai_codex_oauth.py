"""Tests for package-level ChatGPT/Codex OAuth helpers."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import pathlib
import sys
import stat
import urllib.parse

import pytest

OPENAI_CODEX_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "geoagent"
    / "core"
    / "openai_codex.py"
)
SPEC = importlib.util.spec_from_file_location(
    "geoagent_openai_codex",
    OPENAI_CODEX_PATH,
)
openai_codex = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = openai_codex
SPEC.loader.exec_module(openai_codex)


def test_generate_pkce_pair_uses_s256_challenge() -> None:
    """Verify PKCE challenge generation."""
    verifier, challenge = openai_codex.generate_pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )

    assert len(verifier) >= 43
    assert challenge == expected


def test_codex_authorization_url_matches_registered_flow_shape() -> None:
    """Verify Codex login uses the expected redirect and auth parameters."""
    url = openai_codex.build_authorization_url(
        openai_codex.OPENAI_CODEX_AUTHORIZATION_URL,
        client_id=openai_codex.OPENAI_CODEX_CLIENT_ID,
        redirect_uri="http://localhost:1455/auth/callback",
        code_challenge="challenge",
        state="state",
        scope=openai_codex.OPENAI_CODEX_SCOPE,
        extra_params=openai_codex.OPENAI_CODEX_AUTH_EXTRA_PARAMS,
    )
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.openai.com"
    assert params["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert params["client_id"] == [openai_codex.OPENAI_CODEX_CLIENT_ID]
    assert params["scope"] == ["openid profile email offline_access"]
    assert params["id_token_add_organizations"] == ["true"]
    assert params["codex_cli_simplified_flow"] == ["true"]
    assert params["originator"] == ["pi"]


def test_save_load_and_clear_token_payload(tmp_path) -> None:
    """Verify local token file storage uses user-only permissions."""
    token_file = tmp_path / "codex.json"
    token = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_at": 4102444800,
    }

    saved = openai_codex.save_token_payload(token, token_file=token_file)

    assert saved == token_file
    assert openai_codex.load_token_payload(token_file=token_file) == token
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600

    openai_codex.clear_token_payload(token_file=token_file)

    assert not token_file.exists()


def test_ensure_environment_refreshes_stored_token(monkeypatch, tmp_path) -> None:
    """Verify expired stored tokens refresh and export environment variables."""
    token_file = tmp_path / "codex.json"
    token_file.write_text(
        json.dumps(
            {
                "access_token": "old-access",
                "refresh_token": "refresh",
                "expires_at": 1,
            }
        ),
        encoding="utf-8",
    )
    refreshed = {
        "access_token": "new-access",
        "refresh_token": "refresh",
        "expires_at": 4102444800,
        "account_id": "account-123",
    }
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        openai_codex,
        "refresh_oauth_token",
        lambda **kwargs: refreshed,
    )

    payload = openai_codex.ensure_openai_codex_environment(token_file=token_file)

    assert payload == refreshed
    assert openai_codex.os.environ["OPENAI_CODEX_ACCESS_TOKEN"] == "new-access"
    assert openai_codex.os.environ["OPENAI_CODEX_ACCOUNT_ID"] == "account-123"
    assert openai_codex.load_token_payload(token_file=token_file) == refreshed


def test_load_token_payload_missing_file_raises(tmp_path) -> None:
    """Verify missing local login has a clear error."""
    with pytest.raises(RuntimeError, match="geoagent codex login"):
        openai_codex.load_token_payload(token_file=tmp_path / "missing.json")
