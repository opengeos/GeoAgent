"""Tests for settings diagnostics and installer selection helpers."""

from __future__ import annotations

import types
from importlib import util
from pathlib import Path

from open_geoagent.dialogs.settings_dock import (
    ProviderTestWorker,
    SETTINGS_PREFIX,
    collect_diagnostics,
    _model_requires_default_temperature,
)


class _FakeSettings:
    """Small QSettings stand-in for diagnostics tests."""

    def __init__(self, values):
        self.values = dict(values)

    def value(self, key, default="", type=str):  # noqa: A002
        value = self.values.get(key, default)
        if type is bool:
            return bool(value)
        if type is int:
            return int(value)
        return value


def test_collect_diagnostics_redacts_credentials(monkeypatch, tmp_path) -> None:
    """Diagnostics expose credential presence only, never secret values."""
    from open_geoagent import deps_manager, uv_manager

    monkeypatch.setattr(deps_manager, "check_dependencies", lambda: [])
    monkeypatch.setattr(deps_manager, "venv_exists", lambda: True)
    monkeypatch.setattr(deps_manager, "get_venv_dir", lambda: "/tmp/venv")
    monkeypatch.setattr(
        deps_manager, "get_venv_site_packages", lambda: "/tmp/venv/site-packages"
    )
    monkeypatch.setattr(uv_manager, "get_uv_path", lambda: "/tmp/uv")
    monkeypatch.setattr(uv_manager, "verify_uv", lambda: (True, "uv ok"))

    settings = _FakeSettings(
        {
            f"{SETTINGS_PREFIX}provider": "openai",
            f"{SETTINGS_PREFIX}model": "gpt-test",
            f"{SETTINGS_PREFIX}openai_api_key": "sk-secret",
        }
    )
    (tmp_path / "metadata.txt").write_text("version=1.2.3\n", encoding="utf-8")

    diagnostics = collect_diagnostics(settings, str(tmp_path))
    text = str(diagnostics)

    assert diagnostics["credential_presence"]["openai_api_key"]["saved"] is True
    assert diagnostics["model"]["provider"] == "openai"
    assert "sk-secret" not in text


def test_uv_usable_requires_successful_verification(monkeypatch) -> None:
    """A stale uv file should not be treated as usable."""
    from open_geoagent import deps_manager

    monkeypatch.setattr(
        deps_manager,
        "uv_manager",
        types.SimpleNamespace(uv_exists=lambda: True, verify_uv=lambda: (False, "bad")),
        raising=False,
    )

    # Patch the relative import target through sys.modules by monkeypatching the
    # imported module functions directly.
    import open_geoagent.uv_manager as uv_manager

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: True)
    monkeypatch.setattr(uv_manager, "verify_uv", lambda: (False, "bad"))

    assert deps_manager._uv_usable() is False


def test_provider_test_worker_uses_ollama_safe_smoke_prompt(monkeypatch) -> None:
    """Ollama smoke tests should avoid GeoAgent's full prompt/token budget."""
    import sys

    captured = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    def _resolve_model(config):
        captured["resolved_config"] = config
        return "model"

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs

        def __call__(self, prompt):
            captured["prompt"] = prompt
            return "ok"

    geoagent_module = types.ModuleType("geoagent")
    geoagent_module.GeoAgentConfig = _FakeConfig
    model_module = types.ModuleType("geoagent.core.model")
    model_module.resolve_model = _resolve_model
    strands_module = types.ModuleType("strands")
    strands_module.Agent = _FakeAgent
    monkeypatch.setitem(sys.modules, "geoagent", geoagent_module)
    monkeypatch.setitem(sys.modules, "geoagent.core", types.ModuleType("geoagent.core"))
    monkeypatch.setitem(sys.modules, "geoagent.core.model", model_module)
    monkeypatch.setitem(sys.modules, "strands", strands_module)

    worker = ProviderTestWorker("ollama", "qwen3.5:4b", 256, _FakeSettings({}))
    emitted = {}
    worker.finished.connect(lambda result: emitted.setdefault("result", result))

    worker.run()

    assert emitted["result"]["success"] is True
    assert captured["config"]["max_tokens"] == 4096
    assert captured["agent_kwargs"]["tools"] == []
    assert "provider connectivity test" in captured["agent_kwargs"]["system_prompt"]
    assert captured["prompt"].startswith("/no_think")


def test_openai_new_models_use_max_completion_tokens(monkeypatch) -> None:
    """OpenAI gpt-5 style model ids should not send legacy max_tokens."""
    import sys

    captured = {}

    class _FakeConfig:
        provider = "openai"
        model = "gpt-5.5"
        temperature = 0
        max_tokens = 2048
        client_args = {}

        def model_copy(self, update=None):
            return self

    class _FakeOpenAIModel:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    config_module = types.ModuleType("geoagent.core.config")
    config_module.GeoAgentConfig = lambda **kwargs: _FakeConfig()
    config_module.ProviderName = str
    openai_module = types.ModuleType("strands.models.openai")
    openai_module.OpenAIModel = _FakeOpenAIModel

    monkeypatch.setitem(sys.modules, "geoagent", types.ModuleType("geoagent"))
    monkeypatch.setitem(sys.modules, "geoagent.core", types.ModuleType("geoagent.core"))
    monkeypatch.setitem(sys.modules, "geoagent.core.config", config_module)
    monkeypatch.setitem(sys.modules, "strands", types.ModuleType("strands"))
    monkeypatch.setitem(
        sys.modules,
        "strands.models",
        types.ModuleType("strands.models"),
    )
    monkeypatch.setitem(sys.modules, "strands.models.openai", openai_module)

    module_path = Path(__file__).resolve().parents[2] / "geoagent" / "core" / "model.py"
    spec = util.spec_from_file_location("_geoagent_model_under_test", module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.resolve_model(_FakeConfig())

    params = captured["kwargs"]["params"]
    assert params["max_completion_tokens"] == 2048
    assert "max_tokens" not in params
    assert "temperature" not in params


def test_provider_test_worker_uses_default_temperature_for_openai_gpt5(
    monkeypatch,
) -> None:
    """Provider smoke tests should not send temperature=0 to GPT-5 models."""
    import sys

    captured = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    def _resolve_model(config):
        return "model"

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs

        def __call__(self, prompt):
            captured["prompt"] = prompt
            return "ok"

    geoagent_module = types.ModuleType("geoagent")
    geoagent_module.GeoAgentConfig = _FakeConfig
    model_module = types.ModuleType("geoagent.core.model")
    model_module.resolve_model = _resolve_model
    strands_module = types.ModuleType("strands")
    strands_module.Agent = _FakeAgent
    monkeypatch.setitem(sys.modules, "geoagent", geoagent_module)
    monkeypatch.setitem(sys.modules, "geoagent.core", types.ModuleType("geoagent.core"))
    monkeypatch.setitem(sys.modules, "geoagent.core.model", model_module)
    monkeypatch.setitem(sys.modules, "strands", strands_module)

    worker = ProviderTestWorker("openai", "gpt-5.5", 1024, _FakeSettings({}))
    emitted = {}
    worker.finished.connect(lambda result: emitted.setdefault("result", result))

    worker.run()

    assert emitted["result"]["success"] is True
    assert captured["config"]["temperature"] == 1
    assert _model_requires_default_temperature("openai", "gpt-5.5") is True


def test_litellm_openai_gpt5_omits_temperature(monkeypatch) -> None:
    """LiteLLM OpenAI GPT-5 routes should not send unsupported temperature."""
    import sys

    captured = {}

    class _FakeConfig:
        provider = "litellm"
        model = "openai/gpt-5.5"
        temperature = 0
        max_tokens = 2048
        client_args = {}
        litellm_base_url = None

        def model_copy(self, update=None):
            return self

    class _FakeLiteLLMModel:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    config_module = types.ModuleType("geoagent.core.config")
    config_module.GeoAgentConfig = lambda **kwargs: _FakeConfig()
    config_module.ProviderName = str
    litellm_module = types.ModuleType("strands.models.litellm")
    litellm_module.LiteLLMModel = _FakeLiteLLMModel

    monkeypatch.setitem(sys.modules, "geoagent", types.ModuleType("geoagent"))
    monkeypatch.setitem(sys.modules, "geoagent.core", types.ModuleType("geoagent.core"))
    monkeypatch.setitem(sys.modules, "geoagent.core.config", config_module)
    monkeypatch.setitem(sys.modules, "strands", types.ModuleType("strands"))
    monkeypatch.setitem(
        sys.modules,
        "strands.models",
        types.ModuleType("strands.models"),
    )
    monkeypatch.setitem(sys.modules, "strands.models.litellm", litellm_module)

    module_path = Path(__file__).resolve().parents[2] / "geoagent" / "core" / "model.py"
    spec = util.spec_from_file_location(
        "_geoagent_model_under_test_litellm", module_path
    )
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.resolve_model(_FakeConfig())

    params = captured["kwargs"]["params"]
    assert params == {"max_tokens": 2048}
