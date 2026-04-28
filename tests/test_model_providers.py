"""Tests for model provider configuration and resolution."""

from __future__ import annotations

import sys
import types

from geoagent.core.config import GeoAgentConfig
from geoagent.core.model import resolve_model


def test_litellm_config_is_valid() -> None:
    """Verify that LiteLLM is accepted as a configured provider."""
    cfg = GeoAgentConfig(provider="litellm", model="openai/gpt-5.5")

    assert cfg.provider == "litellm"
    assert cfg.model == "openai/gpt-5.5"


def test_litellm_can_be_selected_from_environment(monkeypatch) -> None:
    """Verify that LiteLLM environment variables select LiteLLM by default."""
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_HOST",
        "USE_OLLAMA",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm.example.test")

    assert GeoAgentConfig().provider == "litellm"


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
