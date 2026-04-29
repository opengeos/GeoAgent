"""Tests for model provider configuration and resolution."""

from __future__ import annotations

import sys
import types

import pytest

from geoagent.core.config import GeoAgentConfig
from geoagent.core.model import resolve_model


def _install_fake_openai_responses_model(monkeypatch):
    """Install a fake Strands OpenAI Responses module."""

    class FakeOpenAIResponsesModel:
        """Capture OpenAI Responses constructor arguments."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = types.ModuleType("strands.models.openai_responses")
    module.OpenAIResponsesModel = FakeOpenAIResponsesModel
    monkeypatch.setitem(sys.modules, "strands", types.ModuleType("strands"))
    monkeypatch.setitem(
        sys.modules,
        "strands.models",
        types.ModuleType("strands.models"),
    )
    monkeypatch.setitem(sys.modules, "strands.models.openai_responses", module)
    return FakeOpenAIResponsesModel


def test_litellm_config_is_valid() -> None:
    """Verify that LiteLLM is accepted as a configured provider."""
    cfg = GeoAgentConfig(provider="litellm", model="openai/gpt-5.5")

    assert cfg.provider == "litellm"
    assert cfg.model == "openai/gpt-5.5"


def test_openai_codex_config_is_valid() -> None:
    """Verify that OpenAI Codex OAuth is accepted as a provider."""
    cfg = GeoAgentConfig(provider="openai-codex", model="gpt-5.5")

    assert cfg.provider == "openai-codex"
    assert cfg.model == "gpt-5.5"


def test_openai_codex_is_default_provider_without_env(monkeypatch) -> None:
    """Verify OpenAI Codex is the default provider."""
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_CODEX_ACCESS_TOKEN",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_HOST",
        "USE_OLLAMA",
        "LITELLM_API_KEY",
        "LITELLM_MODEL",
        "LITELLM_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    assert GeoAgentConfig().provider == "openai-codex"


def test_openai_codex_can_be_selected_from_environment(monkeypatch) -> None:
    """Verify that Codex OAuth token selects OpenAI Codex by default."""
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_HOST",
        "USE_OLLAMA",
        "LITELLM_API_KEY",
        "LITELLM_MODEL",
        "LITELLM_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_CODEX_ACCESS_TOKEN", "codex-token")

    assert GeoAgentConfig().provider == "openai-codex"


def test_litellm_can_be_selected_from_environment(monkeypatch) -> None:
    """Verify that LiteLLM environment variables select LiteLLM by default."""
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_CODEX_ACCESS_TOKEN",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_HOST",
        "USE_OLLAMA",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm.example.test")

    assert GeoAgentConfig().provider == "litellm"


def test_resolve_openai_codex_model(monkeypatch) -> None:
    """Verify OpenAI Codex resolves to the Responses model."""
    fake_model = _install_fake_openai_responses_model(monkeypatch)
    monkeypatch.setenv("OPENAI_CODEX_ACCESS_TOKEN", "codex-token")
    monkeypatch.setenv("OPENAI_CODEX_ACCOUNT_ID", "account-123")

    model = resolve_model(
        GeoAgentConfig(
            provider="openai-codex",
            model="gpt-5.5",
            temperature=0.1,
            max_tokens=2048,
        )
    )

    assert isinstance(model, fake_model)
    assert model.kwargs == {
        "client_args": {
            "api_key": "codex-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "default_headers": {
                "User-Agent": "codex-cli",
                "ChatGPT-Account-Id": "account-123",
            },
        },
        "model_id": "gpt-5.5",
        "params": {},
    }


def test_openai_codex_requires_token(monkeypatch) -> None:
    """Verify missing Codex OAuth token gives a clear error."""
    _install_fake_openai_responses_model(monkeypatch)
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)

    with pytest.raises(ValueError, match="ChatGPT OAuth access token"):
        resolve_model(GeoAgentConfig(provider="openai-codex"))


def test_resolve_litellm_model(monkeypatch) -> None:
    """Verify that LiteLLM config resolves to the Strands LiteLLM model."""

    class FakeLiteLLMModel:
        """Capture LiteLLM constructor arguments."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = types.ModuleType("strands.models.litellm")
    module.LiteLLMModel = FakeLiteLLMModel
    monkeypatch.setitem(sys.modules, "strands", types.ModuleType("strands"))
    monkeypatch.setitem(
        sys.modules,
        "strands.models",
        types.ModuleType("strands.models"),
    )
    monkeypatch.setitem(sys.modules, "strands.models.litellm", module)
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm.example.test")

    model = resolve_model(
        GeoAgentConfig(
            provider="litellm",
            model="anthropic/claude-3-7-sonnet-20250219",
            temperature=0.2,
            max_tokens=1024,
        )
    )

    assert isinstance(model, FakeLiteLLMModel)
    assert model.kwargs == {
        "client_args": {
            "api_key": "test-key",
            "base_url": "https://litellm.example.test",
        },
        "model_id": "anthropic/claude-3-7-sonnet-20250219",
        "params": {"temperature": 0.2, "max_tokens": 1024},
    }
