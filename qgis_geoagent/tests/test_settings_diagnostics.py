"""Tests for settings diagnostics and installer selection helpers."""

from __future__ import annotations

import types
from importlib import util
from pathlib import Path

from open_geoagent.dialogs.settings_dock import (
    ProviderTestWorker,
    SETTINGS_PREFIX,
    VOICE_SHORTCUT_SETTING,
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
            f"{SETTINGS_PREFIX}transcription_model": "gpt-4o-transcribe",
            f"{SETTINGS_PREFIX}image_model": "gpt-image-2",
            f"{SETTINGS_PREFIX}{VOICE_SHORTCUT_SETTING}": "Alt+M",
            f"{SETTINGS_PREFIX}openai_api_key": "sk-secret",
            f"{SETTINGS_PREFIX}openai_org_id": "org-secret",
            f"{SETTINGS_PREFIX}openai_project_id": "proj-secret",
        }
    )
    (tmp_path / "metadata.txt").write_text("version=1.2.3\n", encoding="utf-8")

    diagnostics = collect_diagnostics(settings, str(tmp_path))
    text = str(diagnostics)

    assert diagnostics["credential_presence"]["openai_api_key"]["saved"] is True
    assert diagnostics["credential_presence"]["openai_org_id"]["saved"] is True
    assert diagnostics["credential_presence"]["openai_project_id"]["saved"] is True
    assert diagnostics["model"]["provider"] == "openai"
    assert diagnostics["model"]["transcription_model"] == "gpt-4o-transcribe"
    assert diagnostics["model"]["image_model"] == "gpt-image-2"
    assert diagnostics["model"]["voice_shortcut"] == "Alt+M"
    assert "sk-secret" not in text
    assert "org-secret" not in text
    assert "proj-secret" not in text


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


def test_find_python_executable_uses_macos_base_executable(
    monkeypatch, tmp_path
) -> None:
    """macOS QGIS exposes the GUI binary as sys.executable in some builds."""
    import sys

    from open_geoagent import deps_manager

    macos_dir = tmp_path / "QGIS.app" / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True)
    qgis_binary = macos_dir / "QGIS"
    python_binary = macos_dir / (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    qgis_binary.write_text("", encoding="utf-8")
    python_binary.write_text("", encoding="utf-8")

    monkeypatch.setattr(deps_manager.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(deps_manager.sys, "platform", "darwin")
    monkeypatch.setattr(deps_manager.sys, "executable", str(qgis_binary))
    monkeypatch.setattr(
        deps_manager.sys, "_base_executable", str(python_binary), raising=False
    )
    monkeypatch.setattr(
        deps_manager,
        "_python_executable_usable",
        lambda path: ((True, "") if path == str(python_binary) else (False, "broken")),
    )

    assert deps_manager._find_python_executable() == str(python_binary)


def test_find_python_executable_finds_macos_bundle_python(
    monkeypatch, tmp_path
) -> None:
    """The resolver should find Python inside QGIS.app, not return QGIS."""
    import sys

    from open_geoagent import deps_manager

    macos_dir = tmp_path / "QGIS.app" / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True)
    qgis_binary = macos_dir / "QGIS"
    python_binary = macos_dir / (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    qgis_binary.write_text("", encoding="utf-8")
    python_binary.write_text("", encoding="utf-8")

    monkeypatch.setattr(deps_manager.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(deps_manager.sys, "platform", "darwin")
    monkeypatch.setattr(deps_manager.sys, "executable", str(qgis_binary))
    monkeypatch.setattr(
        deps_manager.sys, "_base_executable", str(qgis_binary), raising=False
    )
    monkeypatch.setattr(deps_manager.sys, "_base_prefix", str(macos_dir), raising=False)
    monkeypatch.setattr(deps_manager.sys, "prefix", str(macos_dir))
    monkeypatch.setattr(deps_manager.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        deps_manager,
        "_python_executable_usable",
        lambda path: ((True, "") if path == str(python_binary) else (False, "broken")),
    )

    assert deps_manager._find_python_executable() == str(python_binary)


def test_python_executable_usable_rejects_qgis_app_bundle_python(
    monkeypatch, tmp_path
) -> None:
    """A startable QGIS.app Python wrapper still should not be used for venv."""
    from open_geoagent import deps_manager

    python_binary = tmp_path / "QGIS-final-4_0_2.app" / "Contents" / "MacOS" / "python"
    python_binary.parent.mkdir(parents=True)
    python_binary.write_text("", encoding="utf-8")

    monkeypatch.setattr(deps_manager.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(deps_manager.sys, "platform", "darwin")
    monkeypatch.setattr(
        deps_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )

    usable, reason = deps_manager._python_executable_usable(str(python_binary))

    assert usable is False
    assert "QGIS app-bundle Python" in reason


def test_find_python_executable_skips_unstartable_macos_bundle_python(
    monkeypatch, tmp_path
) -> None:
    """The official macOS app python may exist but fail before importing encodings."""
    import sys

    from open_geoagent import deps_manager

    macos_dir = tmp_path / "QGIS.app" / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True)
    qgis_binary = macos_dir / "QGIS"
    broken_python = macos_dir / (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    path_python = (
        tmp_path / "bin" / (f"python{sys.version_info.major}.{sys.version_info.minor}")
    )
    path_python.parent.mkdir()
    qgis_binary.write_text("", encoding="utf-8")
    broken_python.write_text("", encoding="utf-8")
    path_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(deps_manager.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(deps_manager.sys, "platform", "darwin")
    monkeypatch.setattr(deps_manager.sys, "executable", str(qgis_binary))
    monkeypatch.setattr(
        deps_manager.sys, "_base_executable", str(broken_python), raising=False
    )
    monkeypatch.setattr(deps_manager.sys, "_base_prefix", str(macos_dir), raising=False)
    monkeypatch.setattr(deps_manager.sys, "base_prefix", str(tmp_path), raising=False)
    monkeypatch.setattr(
        deps_manager.sys, "base_exec_prefix", str(tmp_path), raising=False
    )
    monkeypatch.setattr(deps_manager.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        deps_manager.shutil,
        "which",
        lambda name: (
            str(path_python)
            if name == f"python{sys.version_info.major}.{sys.version_info.minor}"
            else None
        ),
    )
    monkeypatch.setattr(
        deps_manager,
        "_python_executable_usable",
        lambda path: (
            path == str(path_python),
            "No module named 'encodings'" if path == str(broken_python) else "",
        ),
    )

    assert deps_manager._find_python_executable() == str(path_python)


def test_find_python_executable_refuses_non_python_without_candidate(
    monkeypatch, tmp_path
) -> None:
    """The installer must fail clearly instead of running a QGIS binary."""
    from open_geoagent import deps_manager

    qgis_binary = tmp_path / "QGIS"
    qgis_binary.write_text("", encoding="utf-8")

    monkeypatch.setattr(deps_manager.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(deps_manager.sys, "platform", "darwin")
    monkeypatch.setattr(deps_manager.sys, "executable", str(qgis_binary))
    monkeypatch.setattr(
        deps_manager.sys, "_base_executable", str(qgis_binary), raising=False
    )
    monkeypatch.setattr(deps_manager.sys, "_base_prefix", str(tmp_path), raising=False)
    monkeypatch.setattr(deps_manager.sys, "base_prefix", str(tmp_path), raising=False)
    monkeypatch.setattr(
        deps_manager.sys, "base_exec_prefix", str(tmp_path), raising=False
    )
    monkeypatch.setattr(deps_manager.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(deps_manager.shutil, "which", lambda _name: None)

    try:
        deps_manager._find_python_executable()
    except RuntimeError as exc:
        assert "cannot safely run QGIS itself" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_create_venv_lets_uv_resolve_python_version_when_bundle_python_is_broken(
    monkeypatch, tmp_path
) -> None:
    """uv can create the matching venv when app-bundle Python cannot start."""
    from open_geoagent import deps_manager
    import open_geoagent.uv_manager as uv_manager

    commands = []
    venv_dir = str(tmp_path / "venv")
    expected_python = deps_manager.get_venv_python_path(venv_dir)

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        if cmd[:2] == ["/tmp/uv", "venv"]:
            Path(expected_python).parent.mkdir(parents=True, exist_ok=True)
            Path(expected_python).write_text("", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deps_manager, "python_runtime_supported", lambda: True)
    monkeypatch.setattr(deps_manager, "_uv_usable", lambda: True)
    monkeypatch.setattr(uv_manager, "get_uv_path", lambda: "/tmp/uv")
    monkeypatch.setattr(
        deps_manager,
        "_find_python_executable",
        lambda: (_ for _ in ()).throw(RuntimeError("No module named 'encodings'")),
    )
    monkeypatch.setattr(deps_manager.subprocess, "run", fake_run)

    assert deps_manager.create_venv(venv_dir) == expected_python
    assert commands[0] == [
        "/tmp/uv",
        "venv",
        "--managed-python",
        "--python",
        deps_manager._python_version_spec(),
        venv_dir,
    ]


def test_ensure_usable_venv_recreates_existing_broken_venv(
    monkeypatch, tmp_path
) -> None:
    """A stale venv with a broken Python executable should not be reused."""
    from open_geoagent import deps_manager

    venv_dir = str(tmp_path / "venv")
    python_path = Path(deps_manager.get_venv_python_path(venv_dir))
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    stale_marker = Path(venv_dir) / "stale.txt"
    stale_marker.write_text("stale", encoding="utf-8")
    progress = []

    def fake_create_venv(path):
        assert path == venv_dir
        recreated_python = Path(deps_manager.get_venv_python_path(path))
        recreated_python.parent.mkdir(parents=True, exist_ok=True)
        recreated_python.write_text("", encoding="utf-8")
        return str(recreated_python)

    monkeypatch.setattr(
        deps_manager,
        "venv_python_usable",
        lambda _path: (False, "No module named 'encodings'"),
    )
    monkeypatch.setattr(deps_manager, "create_venv", fake_create_venv)

    assert deps_manager._ensure_usable_venv(
        venv_dir, progress_callback=lambda p, m: progress.append((p, m))
    ) == str(python_path)
    assert not stale_marker.exists()
    assert any("unusable" in message for _percent, message in progress)


def test_install_packages_falls_back_to_pip_when_uv_install_fails(
    monkeypatch, tmp_path
) -> None:
    """Dependency installation should retry with pip when uv pip fails."""
    from open_geoagent import deps_manager
    import open_geoagent.uv_manager as uv_manager

    venv_dir = str(tmp_path / "venv")
    python_path = Path(deps_manager.get_venv_python_path(venv_dir))
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    commands = []
    progress = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        if cmd[:3] == ["/tmp/uv", "pip", "install"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="uv failed")
        return types.SimpleNamespace(returncode=0, stdout="pip ok", stderr="")

    monkeypatch.setattr(deps_manager, "_uv_usable", lambda: True)
    monkeypatch.setattr(uv_manager, "get_uv_path", lambda: "/tmp/uv")
    monkeypatch.setattr(deps_manager, "venv_python_usable", lambda _path: (True, ""))
    monkeypatch.setattr(deps_manager, "_verify_pip_and_return", lambda path: path)
    monkeypatch.setattr(deps_manager.subprocess, "run", fake_run)

    success, message = deps_manager.install_packages(
        venv_dir,
        ["GeoAgent[providers]>=1.8.0"],
        progress_callback=lambda p, m: progress.append((p, m)),
    )

    assert success is True
    assert message == "Packages installed successfully."
    assert commands[0][:3] == ["/tmp/uv", "pip", "install"]
    assert commands[1][:4] == [str(python_path), "-m", "pip", "install"]
    assert any("retrying with pip" in message for _percent, message in progress)


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
    assert captured["config"]["max_tokens"] == 256
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
